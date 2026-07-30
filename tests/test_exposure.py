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
from janus.exposure import (
    DynamicToolExposer,
    exposed_separator,
    exposed_tool_name,
)
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


# ── Exposed-tool naming / JANUS_EXPOSED_SEPARATOR (infra-smy1) ────────────────
# Grok Build uses "__" as its own server/tool separator, so it silently drops any
# janus tool whose name contains "__" — which was exactly the two auto-exposed
# open_brain natives. The knob below lets Grok get single-underscore names while
# Claude and Codex keep the byte-identical cap__* names their hooks/docs depend on.


def test_default_separator_is_unchanged_for_claude_and_codex() -> None:
    """REGRESSION LOCK. These exact strings are referenced by Claude/Codex hooks,
    runbooks and tool-call habit (mcp__janus__cap__open_brain__search_thoughts).
    A change here is a breaking rename for two working agents — not a refactor."""
    assert exposed_tool_name("open_brain.search_thoughts") == "cap__open_brain__search_thoughts"
    assert exposed_tool_name("open_brain.capture_thought") == "cap__open_brain__capture_thought"
    assert DynamicToolExposer.tool_name("open_brain.search_thoughts") == (
        "cap__open_brain__search_thoughts"
    )


def test_single_underscore_separator_emits_no_double_underscore() -> None:
    """The Grok configuration. The whole point is that NO '__' survives anywhere in
    the name — including in the stem, which is why the stem is joined with the
    separator instead of being hardcoded as 'cap__'."""
    name = exposed_tool_name("open_brain.search_thoughts", separator="_")
    assert name == "cap_open_brain_search_thoughts"
    assert "__" not in name
    # ... and prefixed by Grok's own separator, it is still unambiguous to Grok.
    assert "__" not in f"janus__{name}".removeprefix("janus__")


def test_separator_read_from_environment() -> None:
    assert exposed_separator({"JANUS_EXPOSED_SEPARATOR": "_"}) == "_"
    assert exposed_separator({"JANUS_EXPOSED_SEPARATOR": "-"}) == "-"


def test_invalid_or_empty_separator_falls_back_to_default() -> None:
    """Fail SAFE: a broken separator degrades to today's behaviour, never to a tool
    name a client would reject outright."""
    for bad in ("", "   ", ".", "::", "a b", "toolongseparator", "$"):
        assert exposed_separator({"JANUS_EXPOSED_SEPARATOR": bad}) == "__"
    # absent entirely
    assert exposed_separator({}) == "__"


def test_separator_applies_to_every_dot_in_a_capability_id() -> None:
    assert exposed_tool_name("a.b.c", separator="_") == "cap_a_b_c"
    assert exposed_tool_name("a.b.c", separator="__") == "cap__a__b__c"
