"""Capability registry — load and validate Janus's git-tracked YAML registry.

The registry is the human-curated source of truth for which downstream MCP
servers exist and which of their tools (*capabilities*) Janus may broker. It is
deliberately split from the SQLite operational cache (see ``schema_store.py``):
this module owns *validation* of the declarative config; the schema store owns
*hashes, verification timestamps, and drift detection*.

Security note (design §5.1 / §11): the model-visible ``Capability.summary`` is
the only descriptive text ever shown to a model. Raw downstream tool
descriptions are untrusted and are **never** stored as a model field here — only
their sha256 hash (``raw_description_hash``) is tracked, so a poisoned
third-party descriptor can never reach the model context through the registry.

Public-repo safety: this loader and its seed config carry no secrets, tokens,
``op://`` references, or internal URLs/IPs. Connection endpoints and credentials
are named via environment variables (``endpoint_env`` / ``auth.secret_env``) and
resolved at runtime by the credential broker.
"""

from __future__ import annotations

import enum
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class RegistryError(ValueError):
    """Raised when the registry config is missing, malformed, or inconsistent."""


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class Transport(enum.StrEnum):
    """How Janus connects to a downstream server."""

    STDIO = "stdio"
    HTTP = "http"
    SSE = "sse"
    STREAMABLE_HTTP = "streamable_http"
    # In-process handler. No endpoint, command, or credential. The broker
    # dispatches ``Capability.handler`` itself; the MCP client never connects.
    NATIVE = "native"


# Handler ids the registry will load. Anything else is rejected at load so a
# typo cannot register a capability that fails open or reaches the network.
NATIVE_HANDLERS: frozenset[str] = frozenset({"github_public.issue_get"})

_OWNER_REPO_RE = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,98})/[A-Za-z0-9](?:[A-Za-z0-9._-]{0,98})$"
)
_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class TrustLevel(enum.StrEnum):
    FIRST_PARTY = "first_party"
    THIRD_PARTY = "third_party"


class Lifecycle(enum.StrEnum):
    ALWAYS_ON = "always_on"
    LAZY = "lazy"


class AuthType(enum.StrEnum):
    NONE = "none"
    BEARER = "bearer"


class EnvScope(enum.StrEnum):
    """First-class environment dimension for policy (design §5.3)."""

    DEV = "dev"
    TEST = "test"
    PROD_SAFE = "prod_safe"
    PROD = "prod"


class RiskTier(enum.StrEnum):
    """Capability risk tiers (design §5.2, copied from Open Edison)."""

    READ_ONLY = "read_only"
    PROD_READ = "prod_read"
    LOCAL_WRITE = "local_write"
    EXTERNAL_WRITE = "external_write"
    NETWORK_EGRESS = "network_egress"
    HUMAN_MESSAGE_SEND = "human_message_send"
    PROD_WRITE = "prod_write"
    CREDENTIAL_ACCESS = "credential_access"
    FINANCIAL_OR_BILLING = "financial_or_billing"
    DESTRUCTIVE = "destructive"


# Severity ordering used for the per-server ``risk_ceiling`` cap and (later) by
# the policy engine. Higher = more dangerous. Orthogonal "sensitive" tiers
# (credential/financial) are ranked above ordinary writes so a read-only ceiling
# can never silently admit them.
RISK_SEVERITY: dict[RiskTier, int] = {
    RiskTier.READ_ONLY: 10,
    RiskTier.PROD_READ: 20,
    RiskTier.LOCAL_WRITE: 30,
    RiskTier.EXTERNAL_WRITE: 40,
    RiskTier.NETWORK_EGRESS: 50,
    RiskTier.HUMAN_MESSAGE_SEND: 55,
    RiskTier.PROD_WRITE: 60,
    RiskTier.CREDENTIAL_ACCESS: 70,
    RiskTier.FINANCIAL_OR_BILLING: 80,
    RiskTier.DESTRUCTIVE: 90,
}


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class ServerAuth(BaseModel):
    """How Janus authenticates to a downstream server.

    ``secret_env`` names an environment variable holding the secret
    (public-repo-safe). ``secret_ref`` is an ``op://`` reference resolved by the
    credential broker — used only in gitignored local overlays, never committed.

    ``extra_headers`` supports downstreams that need more than the single
    ``Authorization: Bearer`` header (e.g. Open Brain wants both a bearer token
    and an ``x-brain-key`` header). It maps a header NAME to the NAME of the
    environment variable holding that header's value — public-repo-safe, since
    only names are committed; the values are resolved at runtime and redacted
    from logs by the credential broker.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: AuthType = AuthType.NONE
    secret_env: str | None = None
    secret_ref: str | None = None
    extra_headers: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_secret(self) -> ServerAuth:
        if self.type is AuthType.NONE:
            if self.secret_env or self.secret_ref:
                raise ValueError("auth type 'none' must not declare a secret")
        elif not (self.secret_env or self.secret_ref):
            raise ValueError(
                f"auth type '{self.type.value}' requires secret_env or secret_ref"
            )
        for header_name, env_name in self.extra_headers.items():
            if not header_name or not env_name:
                raise ValueError(
                    "extra_headers entries require a non-empty header name and "
                    "a non-empty env-var name"
                )
        return self


class Server(BaseModel):
    """A declared downstream server.

    MCP transports (stdio, http, sse, streamable_http) are connected by the
    client manager. ``native`` is an in-process handler: it has no endpoint,
    command, or credential, and the broker dispatches it directly.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    display_name: str
    transport: Transport

    # HTTP-family connection (env var NAME holding the URL).
    endpoint_env: str | None = None

    # stdio connection. ``command`` is a portable binary name; ``command_env``
    # names an env var holding an absolute path (for internal wrapper scripts,
    # keeping host paths out of a public repo).
    command: str | None = None
    command_env: str | None = None
    args: list[str] = Field(default_factory=list)

    # stdio env-injection (infra-b7g) — pass environment into the child so a
    # downstream MCP can run without a bespoke wrapper. ``env`` is a static map
    # (host-specific paths/ids; gitignored overlay only, never the public seed).
    # ``env_passthrough`` names env vars copied from Janus's own process env at
    # connect time (public-repo-safe: only NAMES are committed). The MCP stdio
    # client otherwise inherits only HOME/PATH/USER/... so secrets like the op
    # token never reach the child unless passed through here.
    env: dict[str, str] = Field(default_factory=dict)
    env_passthrough: list[str] = Field(default_factory=list)

    auth: ServerAuth = Field(default_factory=ServerAuth)
    lifecycle: Lifecycle = Lifecycle.ALWAYS_ON
    trust_level: TrustLevel = TrustLevel.FIRST_PARTY
    risk_ceiling: RiskTier = RiskTier.READ_ONLY
    default_env_scope: list[EnvScope] = Field(default_factory=lambda: [EnvScope.DEV])
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_transport(self) -> Server:
        if not self.default_env_scope:
            raise ValueError(f"server '{self.id}': default_env_scope must be non-empty")
        if self.transport is Transport.STDIO:
            if not (self.command or self.command_env):
                raise ValueError(
                    f"server '{self.id}': stdio transport requires command or command_env"
                )
            if self.endpoint_env:
                raise ValueError(
                    f"server '{self.id}': stdio transport must not set endpoint_env"
                )
        elif self.transport is Transport.NATIVE:
            if (
                self.endpoint_env
                or self.command
                or self.command_env
                or self.args
                or self.env
                or self.env_passthrough
            ):
                raise ValueError(
                    f"server '{self.id}': native transport must not declare a "
                    "connection (no endpoint, command, args, or env)"
                )
            if (
                self.auth.type is not AuthType.NONE
                or self.auth.secret_env
                or self.auth.secret_ref
                or self.auth.extra_headers
            ):
                raise ValueError(
                    f"server '{self.id}': native transport must not declare auth or secrets"
                )
        else:
            if not self.endpoint_env:
                raise ValueError(
                    f"server '{self.id}': {self.transport.value} transport requires endpoint_env"
                )
            if self.command or self.command_env or self.args:
                raise ValueError(
                    f"server '{self.id}': {self.transport.value} transport "
                    "must not set command/command_env/args"
                )
            if self.env or self.env_passthrough:
                raise ValueError(
                    f"server '{self.id}': {self.transport.value} transport "
                    "must not set env/env_passthrough (stdio only)"
                )
        return self


class Capability(BaseModel):
    """One brokered downstream tool, after human review/approval.

    ``summary`` is the only model-visible description. The untrusted raw
    downstream description is never a field here — only its hash is tracked.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    server_id: str
    downstream_tool_name: str
    title: str
    summary: str
    risk: RiskTier
    env_scope: list[EnvScope]
    requires_confirmation: bool = False
    approved: bool = False
    quarantined: bool = False
    raw_description_hash: str | None = None
    input_schema_hash: str | None = None
    last_verified: str | None = None
    tags: list[str] = Field(default_factory=list)
    # In-process handler id (see ``NATIVE_HANDLERS``). Unset for MCP tools.
    handler: str | None = None
    # Empty: every identity that passes profile policy may use the capability.
    # Non-empty: only these principal labels (HostIdentity.label) may search or
    # call it. Not MCP session keys. Native handlers must set this. Describe
    # still returns the policy decision so a denied identity sees an explicit deny.
    allowed_identities: list[str] = Field(default_factory=list)
    # Public ``owner/name`` repositories a native read handler may touch.
    # MCP capabilities leave this empty. Entries are the canonical spelling.
    repo_allowlist: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_env_scope(self) -> Capability:
        if not self.env_scope:
            raise ValueError(f"capability '{self.id}': env_scope must be non-empty")
        if self.handler is not None:
            if self.handler not in NATIVE_HANDLERS:
                raise ValueError(
                    f"capability '{self.id}': unknown handler '{self.handler}'"
                )
            if not self.allowed_identities:
                raise ValueError(
                    f"capability '{self.id}': handler '{self.handler}' requires "
                    "allowed_identities (deny by default)"
                )
        elif self.allowed_identities:
            raise ValueError(
                f"capability '{self.id}': allowed_identities requires a handler"
            )
        if self.handler == "github_public.issue_get" and not self.repo_allowlist:
            raise ValueError(
                f"capability '{self.id}': github issue lookup requires a "
                "non-empty repo_allowlist"
            )
        seen_ids: set[str] = set()
        for identity in self.allowed_identities:
            if not _IDENTITY_RE.fullmatch(identity):
                raise ValueError(
                    f"capability '{self.id}': invalid allowed identity '{identity}'"
                )
            if identity in seen_ids:
                raise ValueError(
                    f"capability '{self.id}': duplicate allowed identity '{identity}'"
                )
            seen_ids.add(identity)
        seen_repos: set[str] = set()
        for repo in self.repo_allowlist:
            if not _OWNER_REPO_RE.fullmatch(repo) or ".." in repo:
                raise ValueError(
                    f"capability '{self.id}': invalid repo allowlist entry '{repo}'"
                )
            key = repo.casefold()
            if key in seen_repos:
                raise ValueError(
                    f"capability '{self.id}': duplicate repo allowlist entry '{repo}'"
                )
            seen_repos.add(key)
        if self.repo_allowlist and self.handler != "github_public.issue_get":
            raise ValueError(
                f"capability '{self.id}': repo_allowlist is only valid for "
                "github_public.issue_get"
            )
        return self

    @property
    def callable(self) -> bool:
        """A capability is brokerable only when approved and not quarantined."""
        return self.approved and not self.quarantined


class Registry(BaseModel):
    """The validated registry: servers + capabilities with cross-references."""

    model_config = ConfigDict(frozen=True)

    servers: dict[str, Server]
    capabilities: dict[str, Capability]

    @model_validator(mode="after")
    def _cross_validate(self) -> Registry:
        for cap_id, cap in self.capabilities.items():
            server = self.servers.get(cap.server_id)
            if server is None:
                raise ValueError(
                    f"capability '{cap_id}': unknown server_id '{cap.server_id}'"
                )
            if RISK_SEVERITY[cap.risk] > RISK_SEVERITY[server.risk_ceiling]:
                raise ValueError(
                    f"capability '{cap_id}': risk '{cap.risk.value}' exceeds server "
                    f"'{server.id}' risk_ceiling '{server.risk_ceiling.value}'"
                )
            outside = set(cap.env_scope) - set(server.default_env_scope)
            if outside:
                names = sorted(scope.value for scope in outside)
                raise ValueError(
                    f"capability '{cap_id}': env_scope {names} outside server "
                    f"'{server.id}' default_env_scope"
                )
        return self

    def capabilities_for_server(self, server_id: str) -> list[Capability]:
        return [c for c in self.capabilities.values() if c.server_id == server_id]

    def callable_capabilities(self) -> list[Capability]:
        return [c for c in self.capabilities.values() if c.callable]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_yaml_mapping(path: Path, top_key: str) -> dict[str, object]:
    if not path.exists():
        raise RegistryError(f"registry file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RegistryError(f"{path}: invalid YAML: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise RegistryError(f"{path}: top-level YAML must be a mapping")
    section = raw.get(top_key)
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise RegistryError(f"{path}: '{top_key}' must be a mapping")
    return section


def _load_servers(path: Path) -> dict[str, Server]:
    section = _read_yaml_mapping(path, "servers")
    out: dict[str, Server] = {}
    for raw_id, body in section.items():
        sid = str(raw_id)
        if not isinstance(body, dict):
            raise RegistryError(f"server '{sid}': entry must be a mapping")
        try:
            out[sid] = Server.model_validate({"id": sid, **body})
        except ValidationError as exc:
            raise RegistryError(f"invalid server '{sid}': {exc}") from exc
    return out


def _load_capabilities(path: Path) -> dict[str, Capability]:
    section = _read_yaml_mapping(path, "capabilities")
    out: dict[str, Capability] = {}
    for raw_id, body in section.items():
        cid = str(raw_id)
        if not isinstance(body, dict):
            raise RegistryError(f"capability '{cid}': entry must be a mapping")
        try:
            out[cid] = Capability.model_validate({"id": cid, **body})
        except ValidationError as exc:
            raise RegistryError(f"invalid capability '{cid}': {exc}") from exc
    return out


def load_registry(config_dir: Path | str) -> Registry:
    """Load and validate ``servers.yaml`` + ``capabilities.yaml`` from a directory."""
    config_dir = Path(config_dir)
    servers = _load_servers(config_dir / "servers.yaml")
    capabilities = _load_capabilities(config_dir / "capabilities.yaml")
    try:
        return Registry(servers=servers, capabilities=capabilities)
    except ValidationError as exc:
        raise RegistryError(str(exc)) from exc
