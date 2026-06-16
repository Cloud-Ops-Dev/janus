---
owner: Clayton
last_reviewed: 2026-06-16
source_of_truth_for: Janus MCP gateway operating contract for agents — extends IDE constitution
supersedes: null
---

# Janus — Contract (STRICT)

> **Why this exists.** Entry point for any agent (Claude Code, Codex) working in the Janus repo. Inherits the IDE constitution; Janus-specific notes below.

## Inheritance

Inherits the IDE constitution (`~/IDE/CLAUDE.md` / `~/IDE/AGENTS.md`). The constitution wins on conflict. Structural rules, execution interface, Defect Mode, Open Brain, and memory authority are all inherited — not repeated here.

## Scope

Janus is an MCP gateway / capability broker: agents load one small, stable tool surface (`capability.search` / `describe` / `call`, `server.list` / `health`, `policy.explain`, `audit.recent`) instead of every downstream MCP server. It owns a capability registry, a deny-by-default policy engine (risk tiers + env gates), a credential broker, a result sanitizer, descriptor/schema drift detection, and an audit log. Dual interface: MCP (capable hosts) + REST/CLI (fallback).

Design + phased plan: tracked in Open Brain (`c455fed9`, `38cd4933`), the internal Notion *Master MCP Design* / *Implementation Plan* pages, and the build epic **infra-22q** in beads.

## Governance

- **Tier: RM** (registered in `~/IDE/infra/governance/repos.yaml` + the push guard). No push without a DEPLOYED RM release whose validated SHA equals HEAD. Universal git hooks installed via `bin/repo-manifest install-hooks --apply`.
- **Public repo discipline:** this repo is intended to be public. **No credentials, secrets, tokens, internal hostnames/IPs, or `op://` values in committed code.** Secrets are resolved at runtime from an external secret manager and loaded via systemd `EnvironmentFile`; nothing secret at rest.
- **Issue tracking:** beads, in the infra store (single-tracker doctrine). No separate `.beads/` here.
- **Commit convention:** `[charter:<phase>] <kind>: <slug> — <summary>` per constitution §14.

## Self-sufficiency

Any production Janus service must pass the Logout Test (constitution §12): run after reboot with no interactive shell. Secrets via `EnvironmentFile=`, fail loudly if absent.
