"""FastMCP server — exposes Janus's 7 broker tools as one small MCP surface.

This is the entire agent-facing tool surface (design §3). Everything else (Open
Brain, Paperclip, Beads, ...) stays an implementation detail behind the broker.
Tool names use underscores (``capability_search``) for cross-client portability.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from janus.broker import Broker
from janus.exposure import DynamicToolExposer
from janus.registry.registry import EnvScope, RiskTier


def _parse_env(env: str | None) -> EnvScope | None:
    if env is None:
        return None
    return EnvScope(env)


def _parse_risk(risk: str | None) -> RiskTier | None:
    if risk is None:
        return None
    return RiskTier(risk)


def create_mcp_server(
    broker: Broker, *, name: str = "janus", dynamic_exposure: bool = True
) -> FastMCP:
    """Build the Janus FastMCP server bound to a broker instance.

    ``dynamic_exposure`` (Phase 6) adds ``capability_expose`` / ``capability_
    unexpose`` so clients that handle ``tools/list_changed`` can surface searched
    capabilities as native tools. It is purely additive — the core 7 tools and
    the universal ``capability_call`` fallback are unchanged.
    """
    mcp: FastMCP = FastMCP(name)
    exposer = DynamicToolExposer(mcp, broker)

    @mcp.tool
    async def capability_search(
        query: str,
        env: str | None = None,
        max_results: int = 10,
        risk_max: str | None = None,
    ) -> dict[str, Any]:
        """Search for relevant capabilities. Returns a ranked short list with
        summaries and risk tiers — NOT full schemas. Call capability_describe
        for a schema, then capability_call to invoke."""
        try:
            return broker.capability_search(
                query, _parse_env(env), max_results, _parse_risk(risk_max)
            )
        except ValueError as exc:
            return {"error": str(exc)}

    @mcp.tool
    async def capability_describe(
        capability_id: str, env: str | None = None
    ) -> dict[str, Any]:
        """Describe one capability: its input schema (fetched on demand), risk
        tier, and the policy decision for the current session/environment."""
        try:
            return await broker.capability_describe(capability_id, _parse_env(env))
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
        """Invoke a capability. The call is policy-checked and audited. `reason`
        is your stated intent (recorded). Write/prod-like calls may be denied or
        return needs_confirmation — re-call with confirm=true to proceed."""
        try:
            return await broker.capability_call(
                capability_id, arguments, reason, _parse_env(env), confirmed=confirm
            )
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}

    @mcp.tool
    async def server_list() -> dict[str, Any]:
        """List downstream servers: transport, trust, connection status, and
        capability counts."""
        return broker.server_list()

    @mcp.tool
    async def server_health(server_id: str | None = None) -> dict[str, Any]:
        """Liveness and tool counts for one or all downstream servers."""
        return await broker.server_health(server_id)

    @mcp.tool
    async def policy_explain(
        capability_id: str, env: str | None = None
    ) -> dict[str, Any]:
        """Explain why a capability is allowed, denied, or needs confirmation in
        the given environment for the current session profile."""
        try:
            return broker.policy_explain(capability_id, _parse_env(env))
        except ValueError as exc:
            return {"error": str(exc)}

    @mcp.tool
    async def audit_recent(limit: int = 20) -> dict[str, Any]:
        """Recent brokered invocations (allow/confirm/deny) for this session."""
        return broker.audit_recent(limit)

    if dynamic_exposure:

        @mcp.tool
        async def capability_expose(
            capability_ids: list[str], env: str | None = None
        ) -> dict[str, Any]:
            """Surface the given capabilities as native MCP tools (with their real
            schemas) for clients that handle tools/list_changed. Policy-denied
            capabilities are skipped. capability_call remains the fallback."""
            try:
                return await exposer.expose(capability_ids, _parse_env(env))
            except ValueError as exc:
                return {"error": str(exc)}

        @mcp.tool
        async def capability_unexpose(
            tool_names: list[str] | None = None,
        ) -> dict[str, Any]:
            """Remove dynamically-exposed native tools (all of them if tool_names
            is omitted)."""
            return exposer.unexpose(tool_names)

    return mcp
