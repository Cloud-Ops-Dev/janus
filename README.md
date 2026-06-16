# Janus

> **Draft — scaffold only.** This README will be expanded as the design and implementation land.

**Janus** is an MCP ([Model Context Protocol](https://modelcontextprotocol.io)) **gateway / capability broker**.

Instead of loading every downstream MCP server into every agent session, an agent loads **one** small, stable tool surface and discovers downstream capabilities on demand:

```
capability.search(query)      → a short, ranked list of relevant capabilities (no full schemas)
capability.describe(id)       → the schema + risk + policy for one capability
capability.call(id, args)     → a policy-checked, audited invocation
server.list / server.health   → downstream inventory + liveness
policy.explain                → why an action is allowed / denied / needs confirmation
```

Everything else — the dozens of downstream tools — stays an implementation detail behind a registry, policy engine, credential broker, result sanitizer, and audit log.

## Why

- **Small fixed tool surface** instead of N servers per session → less context cost, better tool selection.
- **Staged just-in-time discovery** (search → describe → call) rather than dumping every tool schema up front.
- **Deny-by-default policy** with risk tiers and environment (dev/test/prod) gates.
- **Credentials owned by the gateway** and resolved from an external secret manager at call time — never exposed to the model.
- **Untrusted tool metadata neutralized** (human-reviewed summaries; descriptor/schema drift detection).
- **Every invocation audited.**
- **Dual interface** — MCP for capable hosts, REST/CLI as a fallback.

## Status

Early development. This is the initial scaffold; architecture and the phased build plan are being finalized.

## License

TBD.
