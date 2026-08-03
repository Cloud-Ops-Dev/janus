# Janus — Operations (Phase 1)

How to deploy, verify, and operate the Janus gateway. Janus serves the same
broker through two front doors:

| Surface | Invocation | For |
|---|---|---|
| MCP / stdio | `python -m janus --stdio` | Per-client spawn (Claude Code, Codex) |
| MCP / HTTP | `python -m janus --serve` (`/mcp/`) | Authenticated networked MCP clients |
| REST | `python -m janus --serve` (`/v1/`) | Hermes Desktop + `bin/janus` CLI |
| MCP / HTTP only | `python -m janus --mcp-http` | Standalone networked MCP process |

The stable agent-facing surface is 7 broker tools: `capability_search`,
`capability_describe`, `capability_call`, `server_list`, `server_health`,
`policy_explain`, `audit_recent`, plus the optional dynamic-exposure controls
`capability_expose` and `capability_unexpose`.

## Configure

1. **Endpoints + secrets** — copy `config/janus.env.template` to
   `~/.config/systemd/user/janus.env` (mode `0600`) and fill in real downstream
   URLs and per-host `JANUS_TOKENS`. Real secret *values* never live in this
   repo; they are env values here or `op://` refs resolved by the credential
   broker (needs `OP_SERVICE_ACCOUNT_TOKEN`).
2. **op token** — create `~/.config/systemd/user/op-creds.env` (mode `0600`)
   with `OP_SERVICE_ACCOUNT_TOKEN=...` (only if any server uses an `op://` ref).
3. **Validate** — `python -m janus --check`. It exits non-zero and prints each
   missing endpoint/secret/token. This is the unit's `ExecStartPre`, so a
   half-configured Janus never starts silently degraded (constitution §12).

## Deploy (systemd --user)

```bash
cp systemd/janus.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now janus.service
loginctl enable-linger "$USER"          # survive logout — required for the Logout Test
```

### Logout Test (constitution §12)

Janus must come back on its own with no interactive shell. Verify:

```bash
systemctl --user is-enabled janus.service      # enabled
loginctl show-user "$USER" -p Linger            # Linger=yes
# then: log out fully (or reboot) and confirm the REST /v1/health responds:
curl -s http://127.0.0.1:8088/v1/health
```

The unit declares all its env via `EnvironmentFile=` and depends on no login
shell, so it passes provided linger is enabled and the two env files exist.

### Liveness watchdog — `Type=notify` + `WatchdogSec=45` (infra-gn2q)

Since 2026-08-03 the unit is `Type=notify`, not `Type=simple`. **If you edit the
unit, keep it that way**, and be aware of what it implies:

- The gateway sends `READY=1` only once the socket is genuinely servable. Under
  `Type=notify`, a unit that reaches `active (running)` has therefore *proved* it
  bound — that state is no longer reachable by merely having a live process.
- It then heartbeats `WATCHDOG=1` every `WatchdogSec/2` from a task **on the
  asyncio event loop**. If the loop stalls for more than `WatchdogSec`, systemd
  kills the process and `Restart=always` brings it back.

**Why it exists.** On 2026-08-03 the gateway wedged for ~20 minutes while this unit
reported `active (running)` the entire time: a downstream 503 livelocked the event
loop at 100% CPU, so the process was alive and served nothing, and SIGTERM could
not even be handled (signal delivery needs the loop to turn) — systemd had to
SIGKILL it at the stop timeout. Incident `infra-8r1x`.

**Operational consequences:**

- A `Failed with result 'watchdog'` in the journal means *the event loop stalled
  past 45s*, not that the process crashed. Treat it as a hang, and go looking for
  a livelock or a blocking call on the loop.
- Before restarting a suspected-wedged gateway, **dump its stacks first** —
  `py-spy dump --pid $(systemctl --user show janus.service -p MainPID --value)`
  (needs `sudo` when `/proc/sys/kernel/yama/ptrace_scope` is `1`). The restart
  destroys the only evidence of *why* it wedged.
- Running `python -m janus --serve` by hand is unaffected: every notification is
  inert when `NOTIFY_SOCKET` is unset.

Check it is actually armed and heartbeating:

```bash
systemctl --user show janus.service -p Type -p WatchdogUSec
systemctl --user show janus.service -p WatchdogTimestamp   # must advance ~every 22.5s
```

### Downstreams do not gate startup

`always_on` downstreams connect in a **background** task; the listener binds
regardless of their health. A dead or slow downstream degrades the brokered
surface but must never stop the gateway serving — that coupling is what let one
remote 503 take six healthy local downstreams offline. Do not reintroduce an
`await gateway.connect()` before the lifespan `yield`.

## Connect agents (run alongside existing MCP — do not rip out)

- **Claude Code / Codex (stdio):** add an MCP server that runs
  `~/IDE/projects/janus/.venv/bin/python -m janus --stdio` with the gateway env.
- **Hermes Desktop / SSH:** use `bin/janus` (set `JANUS_URL` + `JANUS_TOKEN`),
  or POST to `/v1/capability/{search,describe,call}`.
- **Networked MCP:** connect to `http://HOST:8088/mcp/` with one of the bearer
  tokens mapped in `JANUS_TOKENS`. The token supplies the profile/audit identity;
  `Mcp-Session-Id` isolates trifecta and exposed-tool state. Session state expires
  after `JANUS_MCP_SESSION_TTL_SECONDS` (default `3600`).

Cut over per host once a host answers read questions through Janus reliably;
keep the direct MCP configs as documented break-glass.

### Exposed-tool naming — `JANUS_EXPOSED_SEPARATOR` (per-client)

Auto-exposed capabilities (`JANUS_AUTO_EXPOSE`) are surfaced as native tools
named `cap<sep><capability_id with '.' -> sep>`, where `<sep>` defaults to `__`:

    open_brain.search_thoughts  ->  cap__open_brain__search_thoughts

**Do not change the default.** Those exact names are referenced by Claude and
Codex hooks, runbooks and memory as `mcp__janus__cap__open_brain__search_thoughts`;
a global rename breaks two working agents.

`JANUS_EXPOSED_SEPARATOR` overrides `<sep>` **for one client**. It exists because
**Grok Build uses `__` as its own server/tool separator** — its `events.jsonl`
shows `call_id: "janus__capability_search"` — so `janus__cap__open_brain__search_thoughts`
is ambiguous and Grok **silently drops the tool with no warning**. That is the
whole symptom: Grok saw 9 janus tools where Claude saw 11, and the two missing
were exactly the auto-exposed natives, the only two whose names contain `__`
(bead `infra-smy1`).

Grok's wiring therefore sets `JANUS_EXPOSED_SEPARATOR=_`, yielding
`cap_open_brain_search_thoughts` — no `__` anywhere, including the stem, which is
joined with the same separator precisely so a hardcoded `cap__` prefix cannot
smuggle one back in. Janus never parses the name back (the exposer keeps a
`name -> capability_id` dict), so the single underscore costs nothing.

Set by `~/IDE/infra/bin/install-grok-build`, which is the git-tracked source of
truth for Grok's Janus wiring. An empty or structurally invalid value falls back
to `__` — a broken separator degrades to today's behaviour, never to a tool name
a client would reject.

Verify after any change by listing tools per client: **both** modes must report
the same count, and the Grok mode must contain no `__`.

## bin/doctor integration (operator, infra repo)

Once Janus is deployed and classified production, add a check to
`~/IDE/infra/bin/doctor` (and, if applicable, register it in
`governance/production-systems.yaml`) that asserts `janus.service` is
active/enabled and `/v1/health` responds. This is intentionally deferred until
deploy so `bin/doctor` does not hard-fail on an un-deployed service.

## Discovery, approval & drift (Phase 2)

Janus tracks each downstream tool's descriptor and refuses to broker one it
hasn't reviewed. Runtime lifecycle state (approved / quarantined + the reviewed
descriptor/schema *baseline* hashes) lives in `<JANUS_DATA_DIR>/janus-registry.db`
— the same SQLite the live gateway reads, so operator actions take effect
immediately, no restart. Manage it with `bin/janus-admin` (host-local,
human-only; never a network endpoint). Output is JSON.

> **Which registry am I editing?** `JANUS_DATA_DIR` / `JANUS_CONFIG_DIR` decide,
> and they are usually **not** the in-repo `data/` + `config/`. On a deployed host
> both `janus.service` and the co-located stdio broker
> (`~/.config/janus/janus-mcp-stdio.sh`) load them from
> `~/.config/systemd/user/janus.env` — e.g. `~/.local/share/janus/data`.
> `bin/janus-admin` now sources that same file when the caller hasn't pinned the
> dirs, and prints `config_dir=` / `data_dir=` on stderr every run so the target
> is never a guess. **Read the printed `data_dir` before trusting any output.**
>
> This matters because the in-repo `data/` is a public seed copy: on 2026-07-30 it
> reported "0 of 27 capabilities quarantined" while the live registry held "1 of
> 134 quarantined — `open_brain.search_thoughts`, descriptor drift". A bare
> `janus-admin list` run from the repo therefore hid a real quarantine, and an
> `approve` written there would not have reached the running broker. Tracked as
> `infra-6mlu`.

```bash
bin/janus-admin discover                 # crawl downstreams, refresh observations,
                                         #   AUTO-QUARANTINE drifted approved caps + alert
bin/janus-admin list                     # every capability's lifecycle state
bin/janus-admin pending                  # capabilities awaiting first approval (uncallable)
bin/janus-admin approve <capability_id>  # approve + lock observed descriptor as baseline
bin/janus-admin quarantine-capability <id> [--reason ...]
bin/janus-admin quarantine-server <id>   [--reason ...]
bin/janus-admin diff <id> [--fetch]      # baseline-vs-observed hash delta; --fetch also
                                         #   prints the LIVE raw descriptor (operator eyes only)
```

**Lifecycle.** A capability marked `approved: true` in the git-tracked
`config/capabilities.yaml` (the human review) adopts its first observed descriptor
as the trusted baseline on the first `discover`. A capability left unapproved is
`pending` and uncallable until `approve`. Re-`approve` accepts a new descriptor as
the baseline (this is how you clear a drift quarantine).

**Drift = supply-chain defense (design §5.8).** Each `discover`, an *approved*
capability whose raw description or input schema hash diverges from its baseline
is auto-quarantined (uncallable) and an alert fires; it stays quarantined until a
human re-approves. Raw descriptor text never enters model context — only sha256
hashes travel; `diff --fetch` is the one path that shows raw text, and only to the
operator's terminal.

**Alerts.** Set `JANUS_DISCORD_WEBHOOK_URL` in `janus.env` (resolve the
`claude-channel-webhook` value from 1Password once; never commit it). Unset =
drift is still quarantined + logged, just not pinged. Schedule periodic discovery
with a `systemd --user` timer running `bin/janus-admin discover` (Phase 4 will fold
this into the service loop).

## Audit

Every brokered call (allow/confirm/deny) is one row in `data/janus.db`
(`invocations`) and one line in `data/janus.jsonl`. `data/` is gitignored and
must not be in any file-sync tree (constitution §15). Query recent activity via
the `audit_recent` tool or `GET /v1/audit/recent`.
