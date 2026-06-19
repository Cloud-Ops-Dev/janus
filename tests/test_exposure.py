"""Dynamic tool exposure tests (Phase 6, infra-lxt).

Exposes searched capabilities as native MCP tools carrying the downstream's real
schema, verifies a native call routes back through the broker (policy + audit
enforced), that policy-denied capabilities are never exposed, and that unsupported
configurations (dynamic_exposure off) are unaffected.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from janus.audit import InMemoryAuditSink
from janus.broker import Broker
from janus.downstream import DownstreamClientManager
from janus.policy import Decision, PolicyContext, PolicyDecision
from janus.registry import (
    Capability,
    EnvScope,
    Registry,
    RiskTier,
    Server,
    Transport,
)
from janus.server_mcp import build_mcp_server, create_mcp_server

FAKE = str(Path(__file__).parent / "_fake_downstream.py")


class StubPolicy:
    """read_only -> ALLOW, everything else -> DENY (so writes are not exposable)."""

    def evaluate(self, ctx: PolicyContext) -> PolicyDecision:
        if ctx.capability.risk is RiskTier.READ_ONLY:
            return PolicyDecision(Decision.ALLOW, "read ok", ctx.capability.id, ctx.capability.risk)
        return PolicyDecision(Decision.DENY, "write denied", ctx.capability.id, ctx.capability.risk)


def _cap(cid: str, tool: str, risk: RiskTier) -> Capability:
    return Capability(
        id=cid, server_id="fake", downstream_tool_name=tool, title=cid,
        summary="Add two integers a and b", risk=risk,
        env_scope=[EnvScope.DEV, EnvScope.PROD_SAFE], approved=True,
    )


def _registry() -> Registry:
    fake = Server(
        id="fake", display_name="Fake", transport=Transport.STDIO,
        command=sys.executable, args=[FAKE], risk_ceiling=RiskTier.EXTERNAL_WRITE,
        default_env_scope=[EnvScope.DEV, EnvScope.PROD_SAFE],
    )
    caps = {
        "fake.add": _cap("fake.add", "add", RiskTier.READ_ONLY),
        "fake.write": _cap("fake.write", "echo", RiskTier.EXTERNAL_WRITE),
    }
    return Registry(servers={"fake": fake}, capabilities=caps)


def _broker(mgr: DownstreamClientManager) -> Broker:
    return Broker(
        _registry(), mgr, StubPolicy(), InMemoryAuditSink(),
        default_env=EnvScope.PROD_SAFE,
    )


def test_expose_adds_native_tool_with_real_schema_and_routes_through_broker() -> None:
    async def body() -> None:
        mgr = DownstreamClientManager(_registry().servers)
        async with mgr:
            await mgr.connect_all()
            broker = _broker(mgr)
            server = create_mcp_server(broker)

            res = await server.call_tool("capability_expose", {"capability_ids": ["fake.add"]})
            assert res.structured_content["exposed"] == ["cap__fake__add"]

            tools = {t.name: t for t in await server.list_tools()}
            assert "cap__fake__add" in tools
            # the native tool advertises the downstream's REAL input schema.
            assert sorted(tools["cap__fake__add"].parameters["properties"]) == ["a", "b"]

            out = await server.call_tool("cap__fake__add", {"a": 2, "b": 3})
            assert out.structured_content["status"] == "ok"
            assert out.structured_content["structured"]["result"] == 5
            # routed through the broker -> the call was policy-checked + audited.
            audit = broker.audit_recent()
            assert audit["entries"][0]["capability_id"] == "fake.add"

    asyncio.run(body())


def test_policy_denied_capability_is_not_exposed() -> None:
    async def body() -> None:
        mgr = DownstreamClientManager(_registry().servers)
        async with mgr:
            await mgr.connect_all()
            server = create_mcp_server(_broker(mgr))
            res = await server.call_tool(
                "capability_expose", {"capability_ids": ["fake.write"]}
            )
            assert res.structured_content["exposed"] == []
            assert res.structured_content["skipped"][0]["reason"] == "policy denied"
            assert "cap__fake__write" not in {t.name for t in await server.list_tools()}

    asyncio.run(body())


def test_unexpose_removes_native_tool() -> None:
    async def body() -> None:
        mgr = DownstreamClientManager(_registry().servers)
        async with mgr:
            await mgr.connect_all()
            server = create_mcp_server(_broker(mgr))
            await server.call_tool("capability_expose", {"capability_ids": ["fake.add"]})
            assert "cap__fake__add" in {t.name for t in await server.list_tools()}
            un = await server.call_tool("capability_unexpose", {})
            assert un.structured_content["unexposed"] == ["cap__fake__add"]
            assert "cap__fake__add" not in {t.name for t in await server.list_tools()}

    asyncio.run(body())


def test_expose_unknown_capability_is_skipped() -> None:
    async def body() -> None:
        mgr = DownstreamClientManager(_registry().servers)
        async with mgr:
            await mgr.connect_all()
            server = create_mcp_server(_broker(mgr))
            res = await server.call_tool("capability_expose", {"capability_ids": ["nope"]})
            assert res.structured_content["exposed"] == []
            assert res.structured_content["skipped"][0]["capability_id"] == "nope"

    asyncio.run(body())


def test_dynamic_exposure_off_hides_the_expose_tools() -> None:
    mgr = DownstreamClientManager(_registry().servers)
    server = create_mcp_server(_broker(mgr), dynamic_exposure=False)
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert "capability_expose" not in names
    assert "capability_unexpose" not in names
    assert "capability_call" in names  # the universal fallback is always present


def test_build_mcp_server_returns_exposer_handle() -> None:
    mgr = DownstreamClientManager(_registry().servers)
    # dynamic exposure on -> the serving layer gets a handle to auto-expose with.
    _server, exposer = build_mcp_server(_broker(mgr))
    assert exposer is not None
    # off -> no handle (nothing to auto-expose through).
    _server_off, none_exposer = build_mcp_server(_broker(mgr), dynamic_exposure=False)
    assert none_exposer is None


def test_create_mcp_server_shim_returns_plain_server() -> None:
    mgr = DownstreamClientManager(_registry().servers)
    server = create_mcp_server(_broker(mgr))
    # back-compat: a bare FastMCP, not the (server, exposer) tuple.
    assert not isinstance(server, tuple)
    assert "capability_call" in {t.name for t in asyncio.run(server.list_tools())}
