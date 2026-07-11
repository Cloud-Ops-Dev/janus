"""Authenticated, isolated session state for Janus MCP-over-HTTP."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from fastmcp import FastMCP
from fastmcp.server.auth import AccessToken, TokenVerifier
from fastmcp.server.context import Context
from fastmcp.server.dependencies import get_access_token, get_context
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.tools.tool import Tool, ToolResult

from janus.broker import Broker
from janus.exposure import DynamicToolExposer
from janus.registry.registry import EnvScope
from janus.server_mcp import _parse_env, _parse_risk
from janus.server_rest import BrokerDeps, HostIdentity


class JanusTokenVerifier(TokenVerifier):
    """Reuse the REST ``JANUS_TOKENS`` identities for MCP bearer auth."""

    def __init__(self, identities: dict[str, HostIdentity]) -> None:
        super().__init__()
        self._identities = identities

    async def verify_token(self, token: str) -> AccessToken | None:
        identity = self._identities.get(token)
        if identity is None:
            return None
        return AccessToken(
            token=token,
            client_id=identity.label,
            scopes=[],
            claims={
                "janus_label": identity.label,
                "janus_profile": identity.profile,
                "janus_attended": identity.attended,
            },
        )


@dataclass
class SessionState:
    key: str
    session_id: str
    broker: Broker
    exposer: DynamicToolExposer
    last_seen: float


class McpSessionPool:
    """Broker + exposed-tool state isolated by authenticated MCP session."""

    def __init__(
        self,
        deps: BrokerDeps,
        *,
        ttl_seconds: float = 3600,
        auto_expose: tuple[str, ...] = (),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("MCP session TTL must be greater than zero")
        self._deps = deps
        self._ttl = ttl_seconds
        self._auto_expose = auto_expose
        self._clock = clock
        self._states: dict[str, SessionState] = {}
        self._owners: dict[str, str] = {}

    @staticmethod
    def identity_from_access_token(token: AccessToken | None) -> HostIdentity:
        if token is None:
            raise PermissionError("authenticated Janus identity is required")
        claims = token.claims or {}
        label = str(claims.get("janus_label") or token.client_id)
        return HostIdentity(
            label=label,
            profile=str(claims.get("janus_profile") or "default_assistant"),
            attended=bool(claims.get("janus_attended", True)),
        )

    def state_for(self, identity: HostIdentity, session_id: str) -> SessionState:
        now = self._clock()
        self._prune(now)
        owner = self._owners.get(session_id)
        if owner is not None and owner != identity.label:
            raise PermissionError("MCP session id is already bound to another identity")
        self._owners[session_id] = identity.label
        key = f"{identity.label}:mcp:{session_id}"
        state = self._states.get(key)
        if state is None:
            broker_identity = HostIdentity(
                label=key, profile=identity.profile, attended=identity.attended
            )
            broker = self._deps.broker_for(broker_identity)
            # This exposer never mutates the serving FastMCP provider. It is used
            # only as a per-session schema/tool factory and registry.
            isolated_server: FastMCP = FastMCP(f"janus-session-{session_id}")
            state = SessionState(
                key=key,
                session_id=session_id,
                broker=broker,
                exposer=DynamicToolExposer(isolated_server, broker),
                last_seen=now,
            )
            self._states[key] = state
        state.last_seen = now
        return state

    def from_context(self, context: Context | None = None) -> SessionState:
        ctx = context or get_context()
        identity = self.identity_from_access_token(get_access_token())
        return self.state_for(identity, ctx.session_id)

    async def ensure_auto_exposed(self, state: SessionState) -> None:
        missing = [
            cid
            for cid in self._auto_expose
            if DynamicToolExposer.tool_name(cid) not in state.exposer.active
        ]
        if missing:
            await state.exposer.expose(missing)

    def exposed_tools(self, state: SessionState) -> list[Tool]:
        return state.exposer.tools

    async def expose(
        self, state: SessionState, capability_ids: list[str], env: EnvScope | None
    ) -> dict[str, Any]:
        return await state.exposer.expose(capability_ids, env)

    def unexpose(self, state: SessionState, names: list[str] | None) -> dict[str, Any]:
        return state.exposer.unexpose(names)

    def find_exposed(self, state: SessionState, name: str) -> Tool | None:
        return state.exposer.get_tool(name)

    def _prune(self, now: float) -> None:
        if self._ttl <= 0:
            return
        stale = [key for key, state in self._states.items() if now - state.last_seen >= self._ttl]
        for key in stale:
            state = self._states.pop(key)
            self._owners.pop(state.session_id, None)
            if self._deps.trifecta is not None:
                self._deps.trifecta.clear_session(state.key)


class SessionExposureMiddleware(Middleware):
    def __init__(self, pool: McpSessionPool) -> None:
        self._pool = pool

    async def on_list_tools(
        self, context: MiddlewareContext[Any], call_next: Callable[..., Any]
    ) -> Sequence[Tool]:
        tools = list(await call_next(context))
        state = self._pool.from_context(context.fastmcp_context)
        await self._pool.ensure_auto_exposed(state)
        tools.extend(self._pool.exposed_tools(state))
        return tools

    async def on_call_tool(
        self, context: MiddlewareContext[Any], call_next: Callable[..., Any]
    ) -> ToolResult:
        state = self._pool.from_context(context.fastmcp_context)
        tool = self._pool.find_exposed(state, context.message.name)
        if tool is not None:
            return await tool.run(context.message.arguments or {})
        return await call_next(context)


def build_session_mcp_server(
    deps: BrokerDeps,
    tokens: dict[str, HostIdentity],
    *,
    ttl_seconds: float = 3600,
    auto_expose: tuple[str, ...] = (),
    name: str = "janus",
) -> tuple[FastMCP, McpSessionPool]:
    """Build an authenticated MCP server whose broker view is per session."""
    pool = McpSessionPool(deps, ttl_seconds=ttl_seconds, auto_expose=auto_expose)
    mcp: FastMCP = FastMCP(
        name,
        auth=JanusTokenVerifier(tokens),
        middleware=[SessionExposureMiddleware(pool)],
    )

    def current() -> SessionState:
        return pool.from_context()

    @mcp.tool
    async def capability_search(
        query: str,
        env: str | None = None,
        max_results: int = 10,
        risk_max: str | None = None,
    ) -> dict[str, Any]:
        try:
            return current().broker.capability_search(
                query, _parse_env(env), max_results, _parse_risk(risk_max)
            )
        except ValueError as exc:
            return {"error": str(exc)}

    @mcp.tool
    async def capability_describe(capability_id: str, env: str | None = None) -> dict[str, Any]:
        try:
            return await current().broker.capability_describe(capability_id, _parse_env(env))
        except ValueError as exc:
            return {"error": str(exc)}

    @mcp.tool
    async def capability_call(
        capability_id: str,
        arguments: dict[str, Any],
        reason: str,
        env: str | None = None,
        confirm: bool = False,
    ) -> dict[str, Any]:
        try:
            return await current().broker.capability_call(
                capability_id,
                arguments,
                reason,
                _parse_env(env),
                confirmed=confirm,
            )
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}

    @mcp.tool
    async def server_list() -> dict[str, Any]:
        return current().broker.server_list()

    @mcp.tool
    async def server_health(server_id: str | None = None) -> dict[str, Any]:
        return await current().broker.server_health(server_id)

    @mcp.tool
    async def policy_explain(capability_id: str, env: str | None = None) -> dict[str, Any]:
        try:
            return current().broker.policy_explain(capability_id, _parse_env(env))
        except ValueError as exc:
            return {"error": str(exc)}

    @mcp.tool
    async def audit_recent(limit: int = 20) -> dict[str, Any]:
        return current().broker.audit_recent(limit)

    @mcp.tool
    async def capability_expose(
        capability_ids: list[str], env: str | None = None
    ) -> dict[str, Any]:
        try:
            state = current()
            return await pool.expose(state, capability_ids, _parse_env(env))
        except ValueError as exc:
            return {"error": str(exc)}

    @mcp.tool
    async def capability_unexpose(tool_names: list[str] | None = None) -> dict[str, Any]:
        state = current()
        return pool.unexpose(state, tool_names)

    return mcp, pool
