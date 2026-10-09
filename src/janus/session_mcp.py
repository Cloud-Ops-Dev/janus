"""Authenticated, isolated session state for Janus MCP-over-HTTP."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from fastmcp import FastMCP
from fastmcp.server.auth import AccessToken, TokenVerifier
from fastmcp.server.context import Context
from fastmcp.server.dependencies import get_access_token, get_context
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.tools.tool import Tool, ToolResult

from janus.approval import ApprovalMode
from janus.broker import Broker
from janus.exposure import DynamicToolExposer
from janus.registry.registry import EnvScope
from janus.room_binding import (
    MALFORMED_ROOM_BINDING,
    ROOM_BINDING_MISMATCH,
    RoomBindingRejected,
    presented_room_binding,
    room_bindings_equal,
)
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
                "janus_approval_mode": identity.approval_mode.value,
            },
        )


@dataclass
class SessionState:
    key: str
    session_id: str
    broker: Broker
    exposer: DynamicToolExposer
    last_seen: float
    # Hidden from repr so a logged session object does not include the claim.
    room_binding: str | None = field(default=None, repr=False)


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
        self._lock = threading.Lock()
        # Process-local, like the approval store. Not pruned with session TTL:
        # a copied header must not open a second session after the first expires.
        self._session_bindings: dict[str, str] = {}
        self._binding_owners: dict[str, str] = {}

    @staticmethod
    def identity_from_access_token(token: AccessToken | None) -> HostIdentity:
        if token is None:
            raise PermissionError("authenticated Janus identity is required")
        claims = token.claims or {}
        label = str(claims.get("janus_label") or token.client_id)
        raw_mode = claims.get("janus_approval_mode", ApprovalMode.LEGACY.value)
        if not isinstance(raw_mode, str) or raw_mode not in {
            ApprovalMode.LEGACY.value,
            ApprovalMode.ROOM_APPROVAL.value,
        }:
            raise PermissionError("unknown Janus approval mode")
        return HostIdentity(
            label=label,
            profile=str(claims.get("janus_profile") or "default_assistant"),
            attended=bool(claims.get("janus_attended", True)),
            approval_mode=ApprovalMode(raw_mode),
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
            # ``key`` isolates audit, trifecta, and result handles. Allowlists
            # see the authenticated principal, not this key.
            broker = self._deps.broker_for(identity, session_id=key)
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
        presented: str | None = None
        if identity.approval_mode is ApprovalMode.ROOM_APPROVAL:
            # Legacy identities never reach this read, so a header on those
            # sessions cannot fail the call or stick to a grant.
            presented = presented_room_binding()
        state = self.state_for(identity, ctx.session_id)
        if identity.approval_mode is ApprovalMode.ROOM_APPROVAL:
            self._apply_room_binding(state, presented)
        return state

    def _apply_room_binding(self, state: SessionState, presented: str | None) -> None:
        """Pin ``presented`` to this MCP session, or keep the pin already made.

        A missing header does not clear a pin and does not fail the call.
        A different header for this session, or a header already pinned to
        another session, fails closed. The stored value is copied onto the
        session broker so later grants and consumes use it without reading
        tool arguments.
        """
        with self._lock:
            pinned = self._session_bindings.get(state.session_id)
            if presented is None:
                self._restore_pin(state, pinned)
                return
            if pinned is None:
                owner = self._binding_owners.get(presented)
                if owner is not None and owner != state.session_id:
                    raise RoomBindingRejected(ROOM_BINDING_MISMATCH) from None
                self._session_bindings[state.session_id] = presented
                self._binding_owners[presented] = state.session_id
                self._restore_pin(state, presented)
                return
            if not room_bindings_equal(pinned, presented):
                raise RoomBindingRejected(ROOM_BINDING_MISMATCH) from None
            self._restore_pin(state, pinned)

    @staticmethod
    def _restore_pin(state: SessionState, pinned: str | None) -> None:
        if pinned is None or state.room_binding is not None:
            return
        state.broker.pin_room_binding(pinned)
        state.room_binding = pinned

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
            # A room claim survives the state TTL. Keep its identity owner as
            # well, or a different identity could reuse this session id and
            # inherit the retired room claim without presenting the header.
            if state.session_id not in self._session_bindings:
                self._owners.pop(state.session_id, None)
            if self._deps.trifecta is not None:
                self._deps.trifecta.clear_session(state.key)


def _room_binding_tool_error(exc: RoomBindingRejected) -> ToolResult:
    """Tool result for a rejected header. The value is not copied onto it."""
    message = str(exc)
    if message not in {MALFORMED_ROOM_BINDING, ROOM_BINDING_MISMATCH}:
        message = MALFORMED_ROOM_BINDING
    return ToolResult(structured_content={"status": "error", "error": message})


class SessionExposureMiddleware(Middleware):
    def __init__(self, pool: McpSessionPool) -> None:
        self._pool = pool

    async def on_list_tools(
        self, context: MiddlewareContext[Any], call_next: Callable[..., Any]
    ) -> Sequence[Tool]:
        tools = list(await call_next(context))
        try:
            state = self._pool.from_context(context.fastmcp_context)
        except RoomBindingRejected:
            # A rejected header must not pin, and must not become an error
            # string that could be built from the header value. The base tool
            # list is unchanged; tool calls still fail closed.
            return tools
        await self._pool.ensure_auto_exposed(state)
        tools.extend(self._pool.exposed_tools(state))
        return tools

    async def on_call_tool(
        self, context: MiddlewareContext[Any], call_next: Callable[..., Any]
    ) -> ToolResult:
        try:
            state = self._pool.from_context(context.fastmcp_context)
        except RoomBindingRejected as exc:
            return _room_binding_tool_error(exc)
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
        approval_request_id: str | None = None,
    ) -> dict[str, Any]:
        """Invoke a capability. The call is policy-checked and audited. `reason`
        is your stated intent (recorded). Some calls are denied. Others return
        needs_confirmation with a preview. If that response includes
        approval_request_id, confirm=true does not authorize the call: retry the
        same capability, arguments, environment, and reason with that
        approval_request_id once an operator has approved it. Otherwise re-call
        with confirm=true to proceed."""
        try:
            return await current().broker.capability_call(
                capability_id,
                arguments,
                reason,
                _parse_env(env),
                confirmed=confirm,
                approval_request_id=approval_request_id,
            )
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}

    @mcp.tool
    async def result_read(handle: str, offset: int, limit: int = 20_000) -> dict[str, Any]:
        return current().broker.result_read(handle, offset, limit)

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
