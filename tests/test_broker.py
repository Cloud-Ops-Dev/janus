"""Tests for the broker (the 7 tools) + the FastMCP server wiring.

A ``StubPolicy`` makes decisions deterministic so we can assert that the broker
*enforces* allow/confirm/deny correctly. Downstream calls hit the real
``_fake_downstream.py`` stdio server.
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
from janus.server_mcp import create_mcp_server

FAKE = str(Path(__file__).parent / "_fake_downstream.py")


class StubPolicy:
    """read_only -> ALLOW, local_write -> CONFIRM, everything else -> DENY."""

    def evaluate(self, ctx: PolicyContext) -> PolicyDecision:
        risk = ctx.capability.risk
        if risk is RiskTier.READ_ONLY:
            decision, reason = Decision.ALLOW, "read-only is always allowed"
        elif risk is RiskTier.LOCAL_WRITE:
            decision, reason = Decision.CONFIRM, "local write needs confirmation"
        else:
            decision, reason = Decision.DENY, f"{risk} denied by default"
        return PolicyDecision(decision, reason, ctx.capability.id, risk)


def _cap(cid: str, tool: str, risk: RiskTier, **kw: object) -> Capability:
    return Capability(
        id=cid,
        server_id="fake",
        downstream_tool_name=tool,
        title=kw.pop("title", cid),  # type: ignore[arg-type]
        summary=kw.pop("summary", cid),  # type: ignore[arg-type]
        risk=risk,
        env_scope=[EnvScope.DEV, EnvScope.PROD_SAFE],
        approved=True,
        **kw,  # type: ignore[arg-type]
    )


def _registry() -> Registry:
    fake = Server(
        id="fake",
        display_name="Fake",
        transport=Transport.STDIO,
        command=sys.executable,
        args=[FAKE],
        risk_ceiling=RiskTier.EXTERNAL_WRITE,
        default_env_scope=[EnvScope.DEV, EnvScope.PROD_SAFE],
    )
    caps = {
        "fake.echo": _cap("fake.echo", "echo", RiskTier.READ_ONLY,
                          title="Echo text", summary="Echo the text back."),
        "fake.add": _cap("fake.add", "add", RiskTier.READ_ONLY,
                         title="Add two integers", summary="Add two integers a and b."),
        "fake.write": _cap("fake.write", "echo", RiskTier.LOCAL_WRITE,
                           title="Echo write", summary="Write via echo."),
        "fake.ext": _cap("fake.ext", "echo", RiskTier.EXTERNAL_WRITE,
                        title="Add external", summary="Add via external write."),
        "fake.quar": _cap("fake.quar", "echo", RiskTier.READ_ONLY,
                         title="Quarantined", summary="Quarantined echo.",
                         quarantined=True),
    }
    return Registry(servers={"fake": fake}, capabilities=caps)


def _broker(manager: DownstreamClientManager, *, attended: bool = True) -> Broker:
    return Broker(
        _registry(),
        manager,
        StubPolicy(),
        InMemoryAuditSink(),
        attended=attended,
        default_env=EnvScope.PROD_SAFE,
        session_id="s1",
    )


# --------------------------------------------------------------------------- #
# capability_search — sync, no downstream needed
# --------------------------------------------------------------------------- #
def test_search_filters_denied_and_omits_schemas() -> None:
    mgr = DownstreamClientManager(_registry().servers)
    broker = _broker(mgr)
    out = broker.capability_search("add")
    ids = {r["capability_id"] for r in out["results"]}
    assert "fake.add" in ids
    assert "fake.ext" not in ids  # denied tools never surface (design §7)
    for row in out["results"]:
        assert "input_schema" not in row  # search returns no schemas


def test_search_empty_query_returns_all_allowed() -> None:
    mgr = DownstreamClientManager(_registry().servers)
    broker = _broker(mgr)
    out = broker.capability_search("")
    ids = {r["capability_id"] for r in out["results"]}
    # echo/add (allow) + write (confirm) included; ext denied + quar uncallable.
    assert ids == {"fake.echo", "fake.add", "fake.write"}


# --------------------------------------------------------------------------- #
# capability_describe / call — async, real downstream
# --------------------------------------------------------------------------- #
def test_describe_fetches_schema_jit_and_hides_raw_description() -> None:
    async def body() -> None:
        mgr = DownstreamClientManager(_registry().servers)
        async with mgr:
            await mgr.connect_all()
            broker = _broker(mgr)
            out = await broker.capability_describe("fake.add")
            assert out["summary"] == "Add two integers a and b."
            assert "description" not in out  # only summary is model-visible
            assert out["input_schema"] is not None
            assert "a" in out["input_schema"]["properties"]
            assert out["policy"]["decision"] == "allow"

    asyncio.run(body())


def test_call_allow_executes_and_audits_keys_only() -> None:
    async def body() -> None:
        mgr = DownstreamClientManager(_registry().servers)
        async with mgr:
            await mgr.connect_all()
            broker = _broker(mgr)
            out = await broker.capability_call(
                "fake.add", {"a": 2, "b": 3}, reason="sum two numbers"
            )
            assert out["status"] == "ok"
            assert "5" in out["text"] or (
                out["structured"] and 5 in out["structured"].values()
            )
            audit = broker.audit_recent()
            assert audit["count"] == 1
            entry = audit["entries"][0]
            assert entry["decision"] == "allow"
            assert entry["arg_keys"] == ["a", "b"]  # keys only, never values

    asyncio.run(body())


def test_call_confirm_attended_needs_confirmation() -> None:
    async def body() -> None:
        mgr = DownstreamClientManager(_registry().servers)
        async with mgr:
            await mgr.connect_all()
            broker = _broker(mgr, attended=True)
            out = await broker.capability_call("fake.write", {"text": "x"}, reason="r")
            assert out["status"] == "needs_confirmation"
            assert broker.audit_recent()["entries"][0]["decision"] == "confirm"

    asyncio.run(body())


def test_call_confirm_unattended_hard_denies() -> None:
    async def body() -> None:
        mgr = DownstreamClientManager(_registry().servers)
        async with mgr:
            await mgr.connect_all()
            broker = _broker(mgr, attended=False)
            out = await broker.capability_call("fake.write", {"text": "x"}, reason="r")
            assert out["status"] == "denied"
            assert "unattended" in out["reason"]
            assert broker.audit_recent()["entries"][0]["decision"] == "deny"

    asyncio.run(body())


def test_call_denied_tier_blocked() -> None:
    async def body() -> None:
        mgr = DownstreamClientManager(_registry().servers)
        async with mgr:
            await mgr.connect_all()
            broker = _broker(mgr)
            out = await broker.capability_call("fake.ext", {"text": "x"}, reason="r")
            assert out["status"] == "denied"

    asyncio.run(body())


def test_call_quarantined_blocked_without_policy() -> None:
    async def body() -> None:
        mgr = DownstreamClientManager(_registry().servers)
        async with mgr:
            await mgr.connect_all()
            broker = _broker(mgr)
            out = await broker.capability_call("fake.quar", {"text": "x"}, reason="r")
            assert out["status"] == "denied"
            assert "quarantined" in out["reason"]

    asyncio.run(body())


def test_call_unknown_capability() -> None:
    mgr = DownstreamClientManager(_registry().servers)
    broker = _broker(mgr)
    out = asyncio.run(broker.capability_call("nope", {}, reason="r"))
    assert out["status"] == "error"


# --------------------------------------------------------------------------- #
# server_list / server_health / policy_explain
# --------------------------------------------------------------------------- #
def test_server_list_counts() -> None:
    mgr = DownstreamClientManager(_registry().servers)
    broker = _broker(mgr)
    out = broker.server_list()
    assert out["count"] == 1
    server = out["servers"][0]
    assert server["server_id"] == "fake"
    assert server["capability_count"] == 5
    assert server["connected"] is False  # not connected in this test


def test_server_health_live() -> None:
    async def body() -> None:
        mgr = DownstreamClientManager(_registry().servers)
        async with mgr:
            await mgr.connect_all()
            broker = _broker(mgr)
            out = await broker.server_health()
            row = out["servers"][0]
            assert row["connected"] is True
            assert row["tool_count"] == 2
            assert row["capability_count"] == 5

    asyncio.run(body())


def test_policy_explain() -> None:
    mgr = DownstreamClientManager(_registry().servers)
    broker = _broker(mgr)
    assert broker.policy_explain("fake.echo")["decision"] == "allow"
    assert broker.policy_explain("fake.write")["decision"] == "confirm"
    assert broker.policy_explain("fake.ext")["decision"] == "deny"


# --------------------------------------------------------------------------- #
# FastMCP server surface
# --------------------------------------------------------------------------- #
_CORE_TOOLS = {
    "capability_search",
    "capability_describe",
    "capability_call",
    "server_list",
    "server_health",
    "policy_explain",
    "audit_recent",
}


def test_mcp_server_exposes_core_seven_tools() -> None:
    mgr = DownstreamClientManager(_registry().servers)
    server = create_mcp_server(_broker(mgr), dynamic_exposure=False)
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == _CORE_TOOLS


def test_mcp_server_with_dynamic_exposure_stays_under_ten_tools() -> None:
    mgr = DownstreamClientManager(_registry().servers)
    server = create_mcp_server(_broker(mgr))  # dynamic_exposure on by default
    names = {t.name for t in asyncio.run(server.list_tools())}
    # core 7 + capability_expose/unexpose = 9 (design: model sees < 10 tools).
    assert _CORE_TOOLS <= names
    assert names == _CORE_TOOLS | {"capability_expose", "capability_unexpose"}
    assert len(names) < 10
