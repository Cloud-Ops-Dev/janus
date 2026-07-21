# Contributing to Janus

Thanks for your interest in Janus — an MCP gateway and capability broker. This guide
explains how to file issues, propose changes, and get a pull request merged.

## Ways to help

- **Report a bug** — open a [Bug report](https://github.com/Cloud-Ops-Dev/janus/issues/new?template=bug.yml).
- **Request a feature** — open a [Feature request](https://github.com/Cloud-Ops-Dev/janus/issues/new?template=feature.yml).
- **Improve the docs** — open a [Docs issue](https://github.com/Cloud-Ops-Dev/janus/issues/new?template=docs.yml) or send a PR.
- **Report a vulnerability** — do **not** open a public issue. See [`SECURITY.md`](SECURITY.md).

Please search existing issues before opening a new one.

## Development setup

Janus targets **Python 3.11+** and ships with a project virtualenv.

```bash
git clone https://github.com/Cloud-Ops-Dev/janus.git
cd janus
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

Run the quality gates locally before you push — these are the same checks the release
pipeline enforces:

```bash
.venv/bin/ruff check .          # lint (incl. bandit security rules)
.venv/bin/mypy src              # strict type-check
.venv/bin/python -m pytest tests/ -x -q   # tests
```

A change is not mergeable unless all three pass.

## Making a change

1. **Open (or claim) an issue first** for anything beyond a trivial fix, so work isn't
   duplicated and scope is agreed.
2. **Branch** off `main` using a `type/short-slug` name:
   - `feat/capability-cache`
   - `fix/policy-deny-default`
   - `docs/contributing-guide`
3. Keep changes focused. One logical change per PR.
4. Add or update tests. New behavior without a test will be asked to add one.
5. Update docs (`README.md`, `docs/`) when behavior or interfaces change.

## Pull requests

**All changes land through pull requests** — including those from maintainers. The
default branch is protected; direct pushes to `main` are not accepted.

Your PR should:

- Fill in the [pull request template](.github/PULL_REQUEST_TEMPLATE.md).
- Reference the issue it resolves with `Closes #<n>` in the description.
- Pass lint, types, and tests.
- Have a clear title describing what it delivers.

> **Maintainers / automated agents:** keep the internal work-item reference in the PR
> body as `Bead: <id>` (e.g. `Bead: infra-abc`). The internal tracker (beads) remains
> the source of truth for execution; GitHub Issues/PRs are the external collaboration
> surface. External contributors do **not** need a bead reference.

Review + merge:

- A maintainer reviews for correctness, scope, tests, and docs.
- Once approved and green, a maintainer merges. External PRs are never auto-merged.
- Merging a PR with `Closes #<n>` closes the linked issue.

## Code style

- Formatting and lint are enforced by **ruff** (config in `pyproject.toml`).
- Types are enforced by **mypy --strict**. Public functions are fully typed.
- Prefer small, explicit, well-named functions over cleverness — Janus is a security
  boundary; readability is a feature.

## Scope

In scope: the broker tool surface, registry, policy engine, credential isolation,
sanitizer, and audit log. Out of scope: forking downstream MCP servers, or turning
Janus into a general-purpose framework. When unsure, open a feature request and ask.

## Code of conduct

Be respectful and constructive. Harassment or abuse is not tolerated. Maintainers may
close or block on conduct grounds.

## License

By contributing, you agree that your contributions are licensed under the same terms as
this project (see `LICENSE`).
