"""Lethal-trifecta session guard tests (Phase 3, infra-lxx).

Two layers: (1) pure unit tests over ``legs_for`` + ``TrifectaGuard``; (2) broker
integration over the real ``_fake_downstream.py`` stdio server, asserting that
the guard escalates the *completing external-comm call* — confirm when attended,
hard-deny + alert when unattended — and that pure reads never gate.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from janus.audit import InMemoryAuditSink
from janus.broker import Broker
from janus.downstream import DownstreamClientManager
from janus.policy import (
    Decision,
    PolicyContext,
    PolicyDecision,
    TrifectaGuard,
    TrifectaLeg,
    legs_for,
)
from janus.registry import (
    Capability,
    EnvScope,
    Registry,
    RiskTier,
    Server,
    Transport,
    TrustLevel,
)

FAKE = str(Path(__file__).parent / "_fake_downstream.py")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _server(sid: str, trust: TrustLevel, ceiling: RiskTier) -> Server:
    return Server(
        id=sid,
        display_name=sid,
        transport=Transport.STDIO,
        command=sys.executable,
        args=[FAKE],
        trust_level=trust,
        risk_ceiling=ceiling,
        default_env_scope=[EnvScope.DEV, EnvScope.PROD_SAFE],
    )


def _cap(cid: str, server_id: str, tool: str, risk: RiskTier) -> Capability:
    return Capability(
        id=cid,
        server_id=server_id,
        downstream_tool_name=tool,
        title=cid,
        summary=cid,
        risk=risk,
        env_scope=[EnvScope.DEV, EnvScope.PROD_SAFE],
        approved=True,
    )


# --------------------------------------------------------------------------- #
# legs_for — risk + trust -> trifecta legs
# --------------------------------------------------------------------------- #
def test_legs_first_party_read_is_private_data() -> None:
    fp = _server("fp", TrustLevel.FIRST_PARTY, RiskTier.READ_ONLY)
    cap = _cap("fp.read", "fp", "echo", RiskTier.READ_ONLY)
    assert legs_for(cap, fp) == frozenset({TrifectaLeg.PRIVATE_DATA})


def test_legs_third_party_read_is_untrusted_content() -> None:
    tp = _server("tp", TrustLevel.THIRD_PARTY, RiskTier.READ_ONLY)
    cap = _cap("tp.read", "tp", "echo", RiskTier.READ_ONLY)
    assert legs_for(cap, tp) == frozenset({TrifectaLeg.UNTRUSTED_CONTENT})


def test_legs_external_write_is_external_comm() -> None:
    fp = _server("fp", TrustLevel.FIRST_PARTY, RiskTier.EXTERNAL_WRITE)
    cap = _cap("fp.send", "fp", "echo", RiskTier.EXTERNAL_WRITE)
    assert legs_for(cap, fp) == frozenset({TrifectaLeg.EXTERNAL_COMM})


def test_legs_third_party_egress_is_untrusted_and_external() -> None:
    tp = _server("tp", TrustLevel.THIRD_PARTY, RiskTier.NETWORK_EGRESS)
    cap = _cap("tp.fetch", "tp", "echo", RiskTier.NETWORK_EGRESS)
    assert legs_for(cap, tp) == frozenset(
        {TrifectaLeg.UNTRUSTED_CONTENT, TrifectaLeg.EXTERNAL_COMM}
    )


def test_legs_credential_access_is_private_even_third_party() -> None:
    tp = _server("tp", TrustLevel.THIRD_PARTY, RiskTier.CREDENTIAL_ACCESS)
    cap = _cap("tp.cred", "tp", "echo", RiskTier.CREDENTIAL_ACCESS)
    assert legs_for(cap, tp) == frozenset(
        {TrifectaLeg.UNTRUSTED_CONTENT, TrifectaLeg.PRIVATE_DATA}
    )


def test_legs_local_write_lights_nothing() -> None:
    fp = _server("fp", TrustLevel.FIRST_PARTY, RiskTier.LOCAL_WRITE)
    cap = _cap("fp.w", "fp", "echo", RiskTier.LOCAL_WRITE)
    assert legs_for(cap, fp) == frozenset()


# --------------------------------------------------------------------------- #
# TrifectaGuard — accumulation + gating
# --------------------------------------------------------------------------- #
def test_guard_external_alone_is_not_gated() -> None:
    guard = TrifectaGuard()
    fp = _server("fp", TrustLevel.FIRST_PARTY, RiskTier.EXTERNAL_WRITE)
    send = _cap("fp.send", "fp", "echo", RiskTier.EXTERNAL_WRITE)
    assert guard.assess("s1", send, fp).gated is False


def test_guard_gates_external_after_private_and_untrusted() -> None:
    guard = TrifectaGuard()
    fp = _server("fp", TrustLevel.FIRST_PARTY, RiskTier.EXTERNAL_WRITE)
    tp = _server("tp", TrustLevel.THIRD_PARTY, RiskTier.READ_ONLY)
    read = _cap("fp.read", "fp", "echo", RiskTier.READ_ONLY)
    untrusted = _cap("tp.read", "tp", "echo", RiskTier.READ_ONLY)
    send = _cap("fp.send", "fp", "echo", RiskTier.EXTERNAL_WRITE)

    guard.record("s1", read, fp)       # private_data
    guard.record("s1", untrusted, tp)  # untrusted_content
    assert guard.session_legs("s1") == frozenset(
        {TrifectaLeg.PRIVATE_DATA, TrifectaLeg.UNTRUSTED_CONTENT}
    )
    assessment = guard.assess("s1", send, fp)
    assert assessment.gated is True
    assert "lethal trifecta" in assessment.reason


def test_guard_pure_read_never_gated_even_when_complete() -> None:
    guard = TrifectaGuard()
    fp = _server("fp", TrustLevel.FIRST_PARTY, RiskTier.EXTERNAL_WRITE)
    tp = _server("tp", TrustLevel.THIRD_PARTY, RiskTier.EXTERNAL_WRITE)
    read = _cap("fp.read", "fp", "echo", RiskTier.READ_ONLY)
    untrusted = _cap("tp.read", "tp", "echo", RiskTier.READ_ONLY)
    send = _cap("fp.send", "fp", "echo", RiskTier.EXTERNAL_WRITE)
    # Session already holds all three legs.
    guard.record("s1", read, fp)
    guard.record("s1", untrusted, tp)
    guard.record("s1", send, fp)
    # A further pure read is not an exfil step -> not gated.
    assert guard.assess("s1", read, fp).gated is False
    # But a further external-comm send IS gated (the send is what we stop).
    assert guard.assess("s1", send, fp).gated is True


def test_guard_sessions_are_isolated() -> None:
    guard = TrifectaGuard()
    fp = _server("fp", TrustLevel.FIRST_PARTY, RiskTier.EXTERNAL_WRITE)
    tp = _server("tp", TrustLevel.THIRD_PARTY, RiskTier.READ_ONLY)
    read = _cap("fp.read", "fp", "echo", RiskTier.READ_ONLY)
    untrusted = _cap("tp.read", "tp", "echo", RiskTier.READ_ONLY)
    send = _cap("fp.send", "fp", "echo", RiskTier.EXTERNAL_WRITE)
    guard.record("s1", read, fp)
    guard.record("s1", untrusted, tp)
    # s2 is clean -> the same send is not gated there.
    assert guard.assess("s2", send, fp).gated is False


# --------------------------------------------------------------------------- #
# broker integration
# --------------------------------------------------------------------------- #
class AllowAllPolicy:
    """Every tier ALLOW — isolates the trifecta guard from the policy engine."""

    def evaluate(self, ctx: PolicyContext) -> PolicyDecision:
        return PolicyDecision(
            Decision.ALLOW, "allow-all (test)", ctx.capability.id, ctx.capability.risk
        )


class FakeAlerter:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def send(self, message: str) -> bool:
        self.messages.append(message)
        return True


def _trifecta_registry() -> Registry:
    fp = _server("fp", TrustLevel.FIRST_PARTY, RiskTier.EXTERNAL_WRITE)
    tp = _server("tp", TrustLevel.THIRD_PARTY, RiskTier.NETWORK_EGRESS)
    caps = {
        "fp.read": _cap("fp.read", "fp", "echo", RiskTier.READ_ONLY),
        "tp.read": _cap("tp.read", "tp", "echo", RiskTier.READ_ONLY),
        "fp.send": _cap("fp.send", "fp", "echo", RiskTier.EXTERNAL_WRITE),
    }
    return Registry(servers={"fp": fp, "tp": tp}, capabilities=caps)


def _trifecta_broker(
    mgr: DownstreamClientManager,
    guard: TrifectaGuard,
    *,
    attended: bool,
    alerter: FakeAlerter | None = None,
) -> Broker:
    return Broker(
        _trifecta_registry(),
        mgr,
        AllowAllPolicy(),
        InMemoryAuditSink(),
        trifecta=guard,
        alerter=alerter,
        attended=attended,
        default_env=EnvScope.PROD_SAFE,
        session_id="s1",
    )


def test_broker_trifecta_attended_escalates_to_confirmation() -> None:
    async def body() -> None:
        guard = TrifectaGuard()
        mgr = DownstreamClientManager(_trifecta_registry().servers)
        async with mgr:
            await mgr.connect_all()
            broker = _trifecta_broker(mgr, guard, attended=True)
            # 1) read private (first-party) — allowed, no gate.
            r1 = await broker.capability_call("fp.read", {"text": "secret"}, reason="r")
            assert r1["status"] == "ok"
            # 2) read untrusted (third-party) — allowed, no gate.
            r2 = await broker.capability_call("tp.read", {"text": "web"}, reason="r")
            assert r2["status"] == "ok"
            # 3) external send now COMPLETES the trifecta -> needs confirmation.
            r3 = await broker.capability_call("fp.send", {"text": "x"}, reason="r")
            assert r3["status"] == "needs_confirmation"
            assert r3["trifecta"] is True
            assert "lethal trifecta" in r3["reason"]
            # confirming lets it through.
            r4 = await broker.capability_call(
                "fp.send", {"text": "x"}, reason="r", confirmed=True
            )
            assert r4["status"] == "ok"

    asyncio.run(body())


def test_broker_trifecta_unattended_hard_denies_and_alerts() -> None:
    async def body() -> None:
        guard = TrifectaGuard()
        alerter = FakeAlerter()
        mgr = DownstreamClientManager(_trifecta_registry().servers)
        async with mgr:
            await mgr.connect_all()
            broker = _trifecta_broker(mgr, guard, attended=False, alerter=alerter)
            await broker.capability_call("fp.read", {"text": "secret"}, reason="r")
            await broker.capability_call("tp.read", {"text": "web"}, reason="r")
            out = await broker.capability_call("fp.send", {"text": "x"}, reason="r")
            assert out["status"] == "denied"
            assert out["trifecta"] is True
            assert "lethal trifecta" in out["reason"]
            assert any("lethal trifecta" in m for m in alerter.messages)

    asyncio.run(body())


def test_broker_external_send_without_taint_is_allowed() -> None:
    async def body() -> None:
        guard = TrifectaGuard()
        mgr = DownstreamClientManager(_trifecta_registry().servers)
        async with mgr:
            await mgr.connect_all()
            broker = _trifecta_broker(mgr, guard, attended=True)
            # Private read then external send (no untrusted content) -> not gated.
            await broker.capability_call("fp.read", {"text": "secret"}, reason="r")
            out = await broker.capability_call("fp.send", {"text": "x"}, reason="r")
            assert out["status"] == "ok"

    asyncio.run(body())


def test_broker_blocked_call_does_not_taint_session() -> None:
    """A capability blocked before execution must not record its trifecta legs."""

    async def body() -> None:
        guard = TrifectaGuard()
        fp = _server("fp", TrustLevel.FIRST_PARTY, RiskTier.READ_ONLY)
        quar = _cap("fp.quar", "fp", "echo", RiskTier.READ_ONLY)
        object.__setattr__(quar, "quarantined", True)  # frozen model
        registry = Registry(servers={"fp": fp}, capabilities={"fp.quar": quar})
        mgr = DownstreamClientManager(registry.servers)
        async with mgr:
            await mgr.connect_all()
            broker = Broker(
                registry,
                mgr,
                AllowAllPolicy(),
                InMemoryAuditSink(),
                trifecta=guard,
                attended=True,
                default_env=EnvScope.PROD_SAFE,
                session_id="s1",
            )
            out = await broker.capability_call("fp.quar", {"text": "x"}, reason="r")
            assert out["status"] == "denied"
            # blocked before execution -> no leg recorded.
            assert guard.session_legs("s1") == frozenset()

    asyncio.run(body())
