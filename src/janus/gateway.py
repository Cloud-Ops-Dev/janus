"""Composition root + serving entrypoints (infra-22q.8).

Wires the registry, credential broker, downstream client manager, policy engine,
audit log, sanitizer, broker, and the two front doors (MCP + REST) from one
:class:`GatewayConfig`. Construction (``Gateway.build``) is separated from
connection (``Gateway.connect``) so the manager's downstream sessions open inside
the serving event loop.

Self-sufficiency (constitution §12): :func:`check_environment` validates that
every endpoint/secret a server needs is present and is run as the systemd
unit's ``ExecStartPre`` (``--check``). Resolution fails loudly; nothing here
depends on an interactive shell.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from janus.audit.audit_log import SqliteAuditLog
from janus.discovery.alerts import build_alerter
from janus.downstream.client_manager import DownstreamClientManager
from janus.exposure import DynamicToolExposer
from janus.policy.engine import ProfilePolicyEngine
from janus.policy.profiles import load_profiles
from janus.policy.trifecta import TrifectaGuard
from janus.registry.registry import EnvScope, load_registry
from janus.registry.schema_store import SchemaStore
from janus.search.ranker import BlendedRanker
from janus.security.credential_broker import CredentialBroker
from janus.security.output_sanitizer import OutputSanitizer
from janus.server_mcp import build_mcp_server
from janus.server_rest import BrokerDeps, HostIdentity, create_rest_app

DEFAULT_REST_PORT = 8088
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Runtime lifecycle/registry cache (approval, quarantine, descriptor hashes).
# Distinct from the audit DB (janus.db) — different schema + lifecycle. The admin
# CLI opens the SAME file, so both the live service and the operator see one
# source of truth for approval/quarantine state.
REGISTRY_DB_NAME = "janus-registry.db"


@dataclass(frozen=True)
class GatewayConfig:
    config_dir: Path
    data_dir: Path
    rest_host: str = "127.0.0.1"
    rest_port: int = DEFAULT_REST_PORT
    default_env: EnvScope = EnvScope.PROD_SAFE
    op_path: str = "op"
    mcp_session: str = "mcp"
    mcp_profile: str = "default_assistant"
    # Phase 4 — lazy lifecycle. idle_after_seconds=0 disables idle reaping (every
    # server stays as before); >0 shuts down LAZY servers idle past the timeout.
    idle_after_seconds: float = 0.0
    reaper_interval_seconds: float = 60.0
    # Phase 6 — startup auto-expose. Capability ids surfaced as native MCP tools
    # when an MCP surface (stdio / mcp-http) starts, so daily tools keep native
    # ergonomics after a client drops its direct MCP. Empty = expose nothing
    # (capability_call/capability_expose remain available). Policy still applies:
    # a capability the session may not call is silently skipped.
    auto_expose: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> GatewayConfig:
        env = environ if environ is not None else os.environ
        root = Path(env.get("JANUS_HOME", str(_REPO_ROOT)))
        auto_expose = tuple(
            item.strip()
            for item in env.get("JANUS_AUTO_EXPOSE", "").split(",")
            if item.strip()
        )
        return cls(
            config_dir=Path(env.get("JANUS_CONFIG_DIR", str(root / "config"))),
            data_dir=Path(env.get("JANUS_DATA_DIR", str(root / "data"))),
            rest_host=env.get("JANUS_REST_HOST", "127.0.0.1"),
            rest_port=int(env.get("JANUS_REST_PORT", str(DEFAULT_REST_PORT))),
            default_env=EnvScope(env.get("JANUS_DEFAULT_ENV", "prod_safe")),
            op_path=env.get("JANUS_OP_PATH", "op"),
            mcp_session=env.get("JANUS_MCP_SESSION", "mcp"),
            mcp_profile=env.get("JANUS_MCP_PROFILE", "default_assistant"),
            idle_after_seconds=float(env.get("JANUS_IDLE_AFTER_SECONDS", "0")),
            reaper_interval_seconds=float(env.get("JANUS_REAPER_INTERVAL_SECONDS", "60")),
            auto_expose=auto_expose,
        )


def parse_tokens(environ: Mapping[str, str]) -> dict[str, HostIdentity]:
    """Parse ``JANUS_TOKENS`` (``token=label:profile;...``) into host identities.

    Tokens come from the environment / op refs — never committed.
    """
    raw = environ.get("JANUS_TOKENS", "")
    tokens: dict[str, HostIdentity] = {}
    for part in (p.strip() for p in raw.split(";")):
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"invalid JANUS_TOKENS entry (need token=label:profile): {part}")
        token, spec = part.split("=", 1)
        label, _, profile = spec.partition(":")
        tokens[token.strip()] = HostIdentity(
            label=label.strip() or token.strip(),
            profile=profile.strip() or "default_assistant",
        )
    return tokens


def check_environment(config: GatewayConfig, environ: Mapping[str, str]) -> list[str]:
    """Return a list of problems that would stop Janus serving correctly."""
    problems: list[str] = []
    for name in ("servers.yaml", "capabilities.yaml"):
        if not (config.config_dir / name).exists():
            problems.append(f"missing config file: {config.config_dir / name}")
    if problems:
        return problems

    registry = load_registry(config.config_dir)
    needs_op = False
    for sid, server in registry.servers.items():
        if server.endpoint_env and not environ.get(server.endpoint_env):
            problems.append(f"server '{sid}': endpoint env '{server.endpoint_env}' is unset")
        if server.auth.secret_env and not environ.get(server.auth.secret_env):
            problems.append(f"server '{sid}': secret env '{server.auth.secret_env}' is unset")
        if server.auth.secret_ref:
            needs_op = True
        for header_name, env_name in server.auth.extra_headers.items():
            if not environ.get(env_name):
                problems.append(
                    f"server '{sid}': extra-header env '{env_name}' "
                    f"(for header '{header_name}') is unset"
                )
    if needs_op and not environ.get("OP_SERVICE_ACCOUNT_TOKEN"):
        problems.append("op:// secret refs declared but OP_SERVICE_ACCOUNT_TOKEN is unset")
    if not parse_tokens(environ):
        problems.append("JANUS_TOKENS is unset — the REST API would have no authorized hosts")
    return problems


@dataclass
class Gateway:
    config: GatewayConfig
    manager: DownstreamClientManager
    deps: BrokerDeps
    tokens: dict[str, HostIdentity]
    store: SchemaStore
    _audit: SqliteAuditLog
    _connected: bool = field(default=False)

    @classmethod
    def build(cls, config: GatewayConfig, environ: Mapping[str, str] | None = None) -> Gateway:
        env = dict(environ if environ is not None else os.environ)
        registry = load_registry(config.config_dir)

        profiles_path = config.config_dir / "profiles.yaml"
        policy = ProfilePolicyEngine(
            load_profiles(profiles_path) if profiles_path.exists() else None
        )
        credential = CredentialBroker(env, op_path=config.op_path)
        manager = DownstreamClientManager(
            registry.servers, credential, idle_after=config.idle_after_seconds
        )
        audit = SqliteAuditLog(config.data_dir / "janus.db")
        # The store is the runtime authority for approval/quarantine (Phase 2).
        # Mirror the YAML registry into it; re-sync preserves runtime lifecycle.
        store = SchemaStore(config.data_dir / REGISTRY_DB_NAME)
        store.sync_from_registry(registry)
        sanitizer = OutputSanitizer(credential.redactor)
        # Phase 5: build the blended ranker once and precompute capability vectors.
        ranker = BlendedRanker()
        ranker.prepare(registry.capabilities)
        deps = BrokerDeps(
            registry=registry,
            manager=manager,
            policy=policy,
            audit=audit,
            sanitizer=sanitizer,
            state=store,
            trifecta=TrifectaGuard(),
            alerter=build_alerter(env),
            ranker=ranker,
            default_env=config.default_env,
        )
        return cls(
            config=config,
            manager=manager,
            deps=deps,
            tokens=parse_tokens(env),
            store=store,
            _audit=audit,
        )

    async def connect(self) -> list[str]:
        await self.manager.__aenter__()
        self._connected = True
        return await self.manager.connect_all()

    async def aclose(self) -> None:
        if self._connected:
            await self.manager.__aexit__(None, None, None)
            self._connected = False
        self.store.close()
        self._audit.close()

    def rest_app(self, lifespan: object | None = None) -> FastAPI:
        return create_rest_app(self.deps, self.tokens, lifespan=lifespan)  # type: ignore[arg-type]

    def mcp_server(self) -> tuple[Any, DynamicToolExposer | None]:
        identity = HostIdentity(
            label=self.config.mcp_session, profile=self.config.mcp_profile
        )
        return build_mcp_server(self.deps.broker_for(identity))


# --------------------------------------------------------------------------- #
# Serving entrypoints
# --------------------------------------------------------------------------- #
async def _apply_auto_expose(
    exposer: DynamicToolExposer | None, capability_ids: tuple[str, ...]
) -> dict[str, Any] | None:
    """Surface the configured hot capabilities as native MCP tools at startup.

    No-op (returns ``None``) when dynamic exposure is off or nothing is
    configured. Policy-denied capabilities are skipped inside ``expose`` (never
    raised), so a tightened profile degrades to the generic ``capability_call``
    rather than failing to start.
    """
    if exposer is None or not capability_ids:
        return None
    return await exposer.expose(list(capability_ids))


async def serve_stdio(config: GatewayConfig) -> None:
    """Serve the MCP surface over stdio (per-client spawn; Phase-1 acceptance)."""
    gateway = Gateway.build(config)
    await gateway.connect()
    server, exposer = gateway.mcp_server()
    await _apply_auto_expose(exposer, config.auto_expose)
    try:
        await server.run_stdio_async(show_banner=False)
    finally:
        await gateway.aclose()


async def serve_rest(config: GatewayConfig) -> None:
    """Serve the REST API (always-on networked surface). Operator-deployed."""
    import uvicorn

    gateway = Gateway.build(config)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await gateway.connect()
        reaper: asyncio.Task[None] | None = None
        if config.idle_after_seconds > 0:
            reaper = asyncio.create_task(
                gateway.manager.run_idle_reaper(
                    interval=config.reaper_interval_seconds
                )
            )
        try:
            yield
        finally:
            if reaper is not None:
                reaper.cancel()
                with suppress(asyncio.CancelledError):
                    await reaper
            await gateway.aclose()

    app = gateway.rest_app(lifespan=lifespan)
    server = uvicorn.Server(
        uvicorn.Config(app, host=config.rest_host, port=config.rest_port, log_level="info")
    )
    await server.serve()


async def serve_mcp_http(config: GatewayConfig) -> None:
    """Serve the MCP surface over streamable-HTTP (networked MCP clients)."""
    gateway = Gateway.build(config)
    await gateway.connect()
    server, exposer = gateway.mcp_server()
    await _apply_auto_expose(exposer, config.auto_expose)
    try:
        await server.run_http_async(
            show_banner=False,
            host=config.rest_host,
            port=config.rest_port,
            path="/mcp",
        )
    finally:
        await gateway.aclose()
