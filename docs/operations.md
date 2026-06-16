# Janus — Operations (Phase 1)

How to deploy, verify, and operate the Janus gateway. Janus serves the same
broker through two front doors:

| Surface | Invocation | For |
|---|---|---|
| MCP / stdio | `python -m janus --stdio` | Per-client spawn (Claude Code, Codex) |
| MCP / HTTP | `python -m janus --mcp-http` | Networked MCP clients |
| REST | `python -m janus --serve` | Hermes Desktop + `bin/janus` CLI |

The agent-facing tool surface is exactly 7 tools: `capability_search`,
`capability_describe`, `capability_call`, `server_list`, `server_health`,
`policy_explain`, `audit_recent`.

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

## Connect agents (run alongside existing MCP — do not rip out)

- **Claude Code / Codex (stdio):** add an MCP server that runs
  `~/IDE/projects/janus/.venv/bin/python -m janus --stdio` with the gateway env.
- **Hermes Desktop / SSH:** use `bin/janus` (set `JANUS_URL` + `JANUS_TOKEN`),
  or POST to `/v1/capability/{search,describe,call}`.

Cut over per host once a host answers read questions through Janus reliably;
keep the direct MCP configs as documented break-glass.

## bin/doctor integration (operator, infra repo)

Once Janus is deployed and classified production, add a check to
`~/IDE/infra/bin/doctor` (and, if applicable, register it in
`governance/production-systems.yaml`) that asserts `janus.service` is
active/enabled and `/v1/health` responds. This is intentionally deferred until
deploy so `bin/doctor` does not hard-fail on an un-deployed service.

## Discovery, approval & drift (Phase 2)

Janus tracks each downstream tool's descriptor and refuses to broker one it
hasn't reviewed. Runtime lifecycle state (approved / quarantined + the reviewed
descriptor/schema *baseline* hashes) lives in `data/janus-registry.db` — the same
SQLite the live gateway reads, so operator actions take effect immediately, no
restart. Manage it with `bin/janus-admin` (host-local, human-only; never a
network endpoint). Output is JSON.

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
