# Security Policy

Janus sits in front of every downstream MCP server as a policy and credential boundary,
so we take security reports seriously.

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues,
discussions, or pull requests.**

Instead, use GitHub's **private vulnerability reporting**:

1. Go to the [Security tab](https://github.com/Cloud-Ops-Dev/janus/security) of this
   repository.
2. Click **Report a vulnerability**.
3. Describe the issue, the affected version/commit, and a reproduction if you have one.

This opens a private advisory visible only to you and the maintainers.

We aim to acknowledge a report within **3 business days** and to provide a remediation
timeline after triage. Please give us a reasonable window to fix and release before any
public disclosure.

## Scope

In scope:

- Policy-engine bypass (a capability reachable that deny-by-default should have blocked).
- Credential leakage across the broker boundary (a downstream secret exposed to a client).
- Sanitizer bypass (unsanitized downstream content reaching a client).
- Audit-log tampering or omission of security-relevant events.
- Authentication/authorization flaws in the gateway surface.

Out of scope:

- Vulnerabilities in downstream MCP servers themselves (report those upstream).
- Issues requiring a trusted operator to have already misconfigured policy.
- Denial of service from an already-authorized client exceeding sane limits.

## Supported versions

Janus is under active development; security fixes target the `main` branch. If you are
running a pinned commit, note it in your report.
