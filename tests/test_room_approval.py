"""Room identities cannot self-confirm. An operator grant can, once.

These tests drive the broker and the HTTP approval API the way a room and an
operator would. They assert who is allowed to cause a downstream call, not how
the grant table is stored.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi.testclient import TestClient
from fastmcp.server.auth import AccessToken

from janus.approval import ApprovalMode, ApprovalStore, ApprovalView
from janus.audit import InMemoryAuditSink
from janus.broker import Broker
from janus.downstream.client_manager import DownstreamError, DownstreamResult
from janus.policy import (
    Decision,
    PolicyContext,
    PolicyDecision,
    ProfilePolicyEngine,
    TrifectaGuard,
    TrifectaLeg,
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
from janus.security import OutputSanitizer, SecretRedactor
from janus.server_rest import BrokerDeps, HostIdentity, create_rest_app
from janus.session_mcp import JanusTokenVerifier, McpSessionPool

NEEDLE = "NEEDLE_SECRET_VALUE_9f3a"
ROOM_BEARER = "synthetic-room-bearer"
LEGACY_BEARER = "synthetic-legacy-bearer"
OPERATOR_BEARER = "synthetic-operator-bearer"


class RecordingManager:
    def __init__(self, text: str = "ok") -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.text = text

    async def call(self, server_id: str, tool: str, arguments: dict[str, Any]) -> DownstreamResult:
        self.calls.append((server_id, tool, dict(arguments)))
        return DownstreamResult(False, self.text, None)


class ExplodingManager(RecordingManager):
    async def call(self, server_id: str, tool: str, arguments: dict[str, Any]) -> DownstreamResult:
        self.calls.append((server_id, tool, dict(arguments)))
        raise DownstreamError("downstream failed")


class SlowManager(RecordingManager):
    async def call(self, server_id: str, tool: str, arguments: dict[str, Any]) -> DownstreamResult:
        self.calls.append((server_id, tool, dict(arguments)))
        await asyncio.sleep(0.05)
        return DownstreamResult(False, self.text, None)


class TierPolicy:
    """read_only allow, local_write confirm, everything else deny."""

    def evaluate(self, ctx: PolicyContext) -> PolicyDecision:
        risk = ctx.capability.risk
        if risk is RiskTier.READ_ONLY:
            decision, reason = Decision.ALLOW, "read-only is allowed"
        elif risk is RiskTier.LOCAL_WRITE:
            decision, reason = Decision.CONFIRM, "local write needs confirmation"
        else:
            decision, reason = Decision.DENY, f"{risk} denied"
        return PolicyDecision(decision, reason, ctx.capability.id, risk)


class AllowAll:
    def evaluate(self, ctx: PolicyContext) -> PolicyDecision:
        return PolicyDecision(Decision.ALLOW, "allow-all", ctx.capability.id, ctx.capability.risk)


class ScriptedPolicy:
    def __init__(self, decisions: list[Decision]) -> None:
        self.remaining = list(decisions)

    def evaluate(self, ctx: PolicyContext) -> PolicyDecision:
        decision = self.remaining.pop(0) if self.remaining else Decision.DENY
        return PolicyDecision(decision, "scripted", ctx.capability.id, ctx.capability.risk)


class MutablePolicy:
    def __init__(self, decision: Decision) -> None:
        self.decision = decision

    def evaluate(self, ctx: PolicyContext) -> PolicyDecision:
        return PolicyDecision(self.decision, "mutable", ctx.capability.id, ctx.capability.risk)


class Clocks:
    def __init__(self) -> None:
        self.mono = 0.0
        self.wall = datetime(2026, 10, 4, tzinfo=UTC)

    def monotonic(self) -> float:
        return self.mono

    def wall_clock(self) -> datetime:
        return self.wall

    def advance(self, seconds: float) -> None:
        self.mono += seconds
        self.wall += timedelta(seconds=seconds)


def _server(sid: str, trust: TrustLevel, ceiling: RiskTier) -> Server:
    return Server(
        id=sid,
        display_name=sid,
        transport=Transport.STDIO,
        command="false",
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


def _registry() -> Registry:
    server = _server("fake", TrustLevel.FIRST_PARTY, RiskTier.EXTERNAL_WRITE)
    caps = {
        "fake.read": _cap("fake.read", "fake", "read", RiskTier.READ_ONLY),
        "fake.write": _cap("fake.write", "fake", "echo", RiskTier.LOCAL_WRITE),
        "fake.other": _cap("fake.other", "fake", "other", RiskTier.LOCAL_WRITE),
        "fake.ext": _cap("fake.ext", "fake", "send", RiskTier.EXTERNAL_WRITE),
    }
    return Registry(servers={"fake": server}, capabilities=caps)


def _trifecta_registry() -> Registry:
    fp = _server("fp", TrustLevel.FIRST_PARTY, RiskTier.EXTERNAL_WRITE)
    tp = _server("tp", TrustLevel.THIRD_PARTY, RiskTier.NETWORK_EGRESS)
    caps = {
        "fp.read": _cap("fp.read", "fp", "echo", RiskTier.READ_ONLY),
        "tp.read": _cap("tp.read", "tp", "echo", RiskTier.READ_ONLY),
        "fp.send": _cap("fp.send", "fp", "echo", RiskTier.EXTERNAL_WRITE),
    }
    return Registry(servers={"fp": fp, "tp": tp}, capabilities=caps)


def _broker(
    manager: RecordingManager,
    store: ApprovalStore | None,
    *,
    registry: Registry | None = None,
    policy: TierPolicy | AllowAll | ScriptedPolicy | MutablePolicy | None = None,
    label: str = "room",
    profile: str = "default_assistant",
    session: str = "room-session",
    mode: ApprovalMode = ApprovalMode.ROOM_APPROVAL,
    attended: bool = True,
    trifecta: TrifectaGuard | None = None,
    sanitizer: OutputSanitizer | None = None,
) -> Broker:
    return Broker(
        registry or _registry(),
        manager,  # type: ignore[arg-type]
        policy or TierPolicy(),
        InMemoryAuditSink(),
        sanitizer=sanitizer,
        trifecta=trifecta,
        session_id=session,
        identity=label,
        profile=profile,
        attended=attended,
        default_env=EnvScope.PROD_SAFE,
        approval_mode=mode,
        approvals=store,
    )


def _no_secret(payload: object) -> None:
    assert NEEDLE not in json.dumps(payload)


def _args() -> dict[str, Any]:
    return {"b": 1, "note": NEEDLE, "nested": {"d": 1, "c": 2}}


def _reordered() -> dict[str, Any]:
    return {"nested": {"c": 2, "d": 1}, "note": NEEDLE, "b": 1}


def _view(store: ApprovalStore, approval_id: str) -> ApprovalView:
    view = store.get(approval_id)
    assert view is not None
    return view


async def _issue(broker: Broker, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "capability_id": "fake.write",
        "arguments": _args(),
        "reason": "file the note",
        "confirmed": True,
    }
    payload.update(overrides)
    arguments = payload.pop("arguments")
    capability_id = payload.pop("capability_id")
    reason = payload.pop("reason")
    return await broker.capability_call(capability_id, arguments, reason, **payload)


def test_room_self_confirm_never_calls_downstream() -> None:
    async def body() -> None:
        manager = RecordingManager()
        store = ApprovalStore()
        broker = _broker(manager, store)

        silent = await _issue(broker, confirmed=False)
        shouted = await _issue(broker, confirmed=True)

        assert manager.calls == []
        for response in (silent, shouted):
            assert response["status"] == "needs_confirmation"
            assert response["approval_request_id"]
            assert "arguments" not in response
            assert response["preview"]["arg_keys"] == ["b", "nested", "note"]
            assert "arguments" not in response["preview"]
            _no_secret(response)
            _no_secret(broker.audit_recent())
        assert silent["approval_request_id"] != shouted["approval_request_id"]

        pending = await _issue(
            broker, confirmed=True, approval_request_id=silent["approval_request_id"]
        )
        assert pending["status"] == "needs_confirmation"
        assert pending["approval_request_id"] == silent["approval_request_id"]
        assert manager.calls == []
        assert _view(store, silent["approval_request_id"]).status == "pending"

        unknown = await _issue(broker, approval_request_id="not-a-real-grant", confirmed=True)
        assert unknown["status"] == "denied"
        assert unknown["approval_request_id"] == "not-a-real-grant"
        assert manager.calls == []
        _no_secret(unknown)

        unavailable = _broker(manager, None)
        blocked = await _issue(unavailable, confirmed=True)
        assert blocked["status"] == "denied"
        assert manager.calls == []

    asyncio.run(body())


def test_approved_exact_call_runs_once_and_mismatch_does_not() -> None:
    async def body() -> None:
        manager = RecordingManager()
        store = ApprovalStore()
        broker = _broker(manager, store)
        issued = await _issue(broker)
        approval_id = issued["approval_request_id"]
        assert store.decide(approval_id, "approve").ok

        mismatched = await _issue(
            broker,
            arguments={"note": "different", "b": 1, "nested": {"c": 2, "d": 1}},
            approval_request_id=approval_id,
            confirmed=True,
        )
        assert mismatched["status"] == "denied"
        assert manager.calls == []
        _no_secret(mismatched)
        assert _view(store, approval_id).status == "approved"

        executed = await _issue(
            broker, arguments=_reordered(), approval_request_id=approval_id, confirmed=False
        )
        assert executed["status"] == "ok"
        assert len(manager.calls) == 1
        assert manager.calls[0][0] == "fake"
        assert manager.calls[0][1] == "echo"
        assert manager.calls[0][2]["note"] == NEEDLE
        _no_secret(broker.audit_recent())

        replay = await _issue(broker, arguments=_reordered(), approval_request_id=approval_id)
        assert replay["status"] == "denied"
        assert "consumed" in replay["reason"]
        assert len(manager.calls) == 1

    asyncio.run(body())


def test_binding_mismatches_and_policy_deny_do_not_spend_or_run() -> None:
    async def body() -> None:
        manager = RecordingManager()
        registry = _registry()
        store = ApprovalStore()
        broker = _broker(manager, store, registry=registry, session="session-a")
        approval_id = (await _issue(broker))["approval_request_id"]
        assert store.decide(approval_id, "approve").ok

        attempts = [
            _issue(broker, reason="a different reason", approval_request_id=approval_id),
            _issue(broker, env=EnvScope.DEV, approval_request_id=approval_id),
            _issue(broker, capability_id="fake.other", approval_request_id=approval_id),
            _issue(broker, arguments={"items": [2, 1]}, approval_request_id=approval_id),
            _issue(
                _broker(manager, store, registry=registry, session="session-b"),
                approval_request_id=approval_id,
            ),
            _issue(
                _broker(manager, store, registry=registry, session="session-a", label="other-room"),
                approval_request_id=approval_id,
            ),
            _issue(
                _broker(
                    manager,
                    store,
                    registry=registry,
                    session="session-a",
                    profile="infra_operator",
                ),
                approval_request_id=approval_id,
            ),
            _issue(
                broker,
                capability_id="fake.ext",
                approval_request_id=approval_id,
                confirmed=True,
            ),
        ]
        original = registry.capabilities["fake.write"]
        registry.capabilities["fake.write"] = original.model_copy(
            update={"downstream_tool_name": "renamed"}
        )
        renamed = await _issue(broker, approval_request_id=approval_id)
        registry.capabilities["fake.write"] = original
        assert renamed["status"] == "denied"
        _no_secret(renamed)

        for attempt in attempts:
            outcome = await attempt
            assert outcome["status"] == "denied"
            _no_secret(outcome)
        assert manager.calls == []
        assert _view(store, approval_id).status == "approved"

        exact = await _issue(broker, approval_request_id=approval_id)
        assert exact["status"] == "ok"
        assert len(manager.calls) == 1
        assert manager.calls[0][1] == "echo"

    asyncio.run(body())


def test_integer_and_float_arguments_are_not_the_same_grant() -> None:
    async def body() -> None:
        manager = RecordingManager()
        store = ApprovalStore()
        broker = _broker(manager, store)
        approval_id = (await _issue(broker, arguments={"n": 1}, reason="numbers"))[
            "approval_request_id"
        ]
        assert store.decide(approval_id, "approve").ok
        floated = await _issue(
            broker, arguments={"n": 1.0}, reason="numbers", approval_request_id=approval_id
        )
        assert floated["status"] == "denied"
        assert manager.calls == []
        exact = await _issue(
            broker, arguments={"n": 1}, reason="numbers", approval_request_id=approval_id
        )
        assert exact["status"] == "ok"
        assert len(manager.calls) == 1

    asyncio.run(body())


def test_deny_expiry_restart_and_downstream_error_fail_closed() -> None:
    async def body() -> None:
        clocks = Clocks()
        store = ApprovalStore(clock=clocks.monotonic, wall_clock=clocks.wall_clock)
        manager = RecordingManager()
        broker = _broker(manager, store)

        denied_id = (await _issue(broker, reason="do not"))["approval_request_id"]
        assert store.decide(denied_id, "deny").ok
        assert not store.decide(denied_id, "approve").ok
        denied = await _issue(broker, reason="do not", approval_request_id=denied_id)
        assert denied["status"] == "denied"
        assert manager.calls == []

        expiring_id = (await _issue(broker, reason="wait"))["approval_request_id"]
        clocks.advance(15 * 60 - 1)
        assert _view(store, expiring_id).status == "pending"
        clocks.advance(1)
        assert _view(store, expiring_id).status == "expired"
        assert not store.decide(expiring_id, "approve").ok
        expired = await _issue(broker, reason="wait", approval_request_id=expiring_id)
        assert expired["status"] == "denied"
        assert manager.calls == []

        fresh = ApprovalStore()
        restarted = _broker(manager, fresh)
        # The id was approved nowhere in the new process.
        carried = (await _issue(broker, reason="carry"))["approval_request_id"]
        assert store.decide(carried, "approve").ok
        lost = await _issue(restarted, reason="carry", approval_request_id=carried)
        assert lost["status"] == "denied"
        assert manager.calls == []

        boom = ExplodingManager()
        boom_store = ApprovalStore()
        boom_broker = _broker(boom, boom_store)
        boom_id = (await _issue(boom_broker, reason="boom"))["approval_request_id"]
        assert boom_store.decide(boom_id, "approve").ok
        failed = await _issue(boom_broker, reason="boom", approval_request_id=boom_id)
        assert failed["status"] == "error"
        assert len(boom.calls) == 1
        again = await _issue(boom_broker, reason="boom", approval_request_id=boom_id)
        assert again["status"] == "denied"
        assert len(boom.calls) == 1

    asyncio.run(body())


def test_operator_revoke_before_use_blocks_execution() -> None:
    async def body() -> None:
        manager = RecordingManager()
        store = ApprovalStore()
        broker = _broker(manager, store)
        approval_id = (await _issue(broker))["approval_request_id"]
        assert store.decide(approval_id, "approve").ok
        assert store.decide(approval_id, "approve").ok
        revoked = store.decide(approval_id, "deny")
        assert revoked.ok
        assert revoked.view is not None
        assert revoked.view.status == "denied"
        outcome = await _issue(broker, approval_request_id=approval_id)
        assert outcome["status"] == "denied"
        assert manager.calls == []

    asyncio.run(body())


def test_policy_deny_after_consume_does_not_call_downstream() -> None:
    async def body() -> None:
        manager = RecordingManager()
        policy = ScriptedPolicy([Decision.CONFIRM, Decision.CONFIRM, Decision.DENY])
        store = ApprovalStore()
        broker = _broker(manager, store, policy=policy)
        approval_id = (await _issue(broker))["approval_request_id"]
        assert store.decide(approval_id, "approve").ok
        outcome = await _issue(broker, approval_request_id=approval_id)
        assert outcome["status"] == "denied"
        assert manager.calls == []
        assert policy.remaining == []
        assert _view(store, approval_id).status == "consumed"

    asyncio.run(body())


def test_legacy_confirm_still_executes_without_an_approval_id() -> None:
    async def body() -> None:
        manager = RecordingManager()
        store = ApprovalStore()
        legacy = _broker(manager, store, mode=ApprovalMode.LEGACY, label="legacy-host")
        waiting = await _issue(legacy, confirmed=False)
        assert waiting["status"] == "needs_confirmation"
        assert "approval_request_id" not in waiting
        assert manager.calls == []

        proceeded = await _issue(legacy, confirmed=True)
        assert proceeded["status"] == "ok"
        assert "approval_request_id" not in proceeded
        assert len(manager.calls) == 1

    asyncio.run(body())


def test_trifecta_self_confirm_blocks_until_exact_approval() -> None:
    async def body() -> None:
        manager = RecordingManager()
        guard = TrifectaGuard()
        store = ApprovalStore()
        registry = _trifecta_registry()
        broker = _broker(
            manager,
            store,
            registry=registry,
            policy=AllowAll(),
            trifecta=guard,
        )
        assert (await broker.capability_call("fp.read", {"text": "secret"}, "r"))["status"] == "ok"
        assert (await broker.capability_call("tp.read", {"text": "web"}, "r"))["status"] == "ok"
        gated = await broker.capability_call("fp.send", {"text": "x"}, "r", confirmed=True)
        assert gated["status"] == "needs_confirmation"
        assert gated["trifecta"] is True
        assert "lethal trifecta" in gated["reason"]
        assert NEEDLE not in json.dumps(gated)
        assert len(manager.calls) == 2
        assert TrifectaLeg.EXTERNAL_COMM not in guard.session_legs("room-session")

        approval_id = gated["approval_request_id"]
        assert store.decide(approval_id, "approve").ok
        changed = await broker.capability_call(
            "fp.send", {"text": "y"}, "r", approval_request_id=approval_id, confirmed=True
        )
        assert changed["status"] == "denied"
        assert len(manager.calls) == 2

        exact = await broker.capability_call(
            "fp.send", {"text": "x"}, "r", approval_request_id=approval_id
        )
        assert exact["status"] == "ok"
        assert len(manager.calls) == 3
        assert manager.calls[2][2] == {"text": "x"}

        replay = await broker.capability_call(
            "fp.send", {"text": "x"}, "r", approval_request_id=approval_id, confirmed=True
        )
        assert replay["status"] == "denied"
        assert len(manager.calls) == 3

    asyncio.run(body())


def test_result_read_uses_consumed_grant_not_the_model_confirm_bit() -> None:
    async def body() -> None:
        text = "0123456789abcdef"
        sanitizer = OutputSanitizer(SecretRedactor(), max_chars=8)

        room_policy = MutablePolicy(Decision.ALLOW)
        room = _broker(
            RecordingManager(text),
            ApprovalStore(),
            policy=room_policy,
            sanitizer=sanitizer,
        )
        allowed = await room.capability_call("fake.read", {"q": "x"}, "read", confirmed=True)
        handle = allowed["truncation"]["handle"]
        room_policy.decision = Decision.CONFIRM
        assert room.result_read(handle, 8, 8)["status"] == "denied"

        legacy_policy = MutablePolicy(Decision.ALLOW)
        legacy_sanitizer = OutputSanitizer(SecretRedactor(), max_chars=8)
        legacy = _broker(
            RecordingManager(text),
            ApprovalStore(),
            policy=legacy_policy,
            mode=ApprovalMode.LEGACY,
            label="legacy-host",
            sanitizer=legacy_sanitizer,
        )
        legacy_allowed = await legacy.capability_call(
            "fake.read", {"q": "x"}, "read", confirmed=True
        )
        legacy_policy.decision = Decision.CONFIRM
        assert legacy.result_read(legacy_allowed["truncation"]["handle"], 8, 8)["status"] == "ok"

        confirm_policy = MutablePolicy(Decision.CONFIRM)
        confirm_sanitizer = OutputSanitizer(SecretRedactor(), max_chars=8)
        confirm_store = ApprovalStore()
        confirm_broker = _broker(
            RecordingManager(text),
            confirm_store,
            policy=confirm_policy,
            sanitizer=confirm_sanitizer,
        )
        issued = await confirm_broker.capability_call(
            "fake.write", {"text": "v"}, "write", confirmed=True
        )
        assert issued["status"] == "needs_confirmation"
        assert "truncation" not in issued
        approval_id = issued["approval_request_id"]
        assert confirm_store.decide(approval_id, "approve").ok
        executed = await confirm_broker.capability_call(
            "fake.write", {"text": "v"}, "write", approval_request_id=approval_id
        )
        assert executed["status"] == "ok"
        assert confirm_policy.decision is Decision.CONFIRM
        follow = confirm_broker.result_read(executed["truncation"]["handle"], 8, 8)
        assert follow["status"] == "ok"
        assert follow["text"]

    asyncio.run(body())


def test_only_one_inflight_retry_reaches_downstream() -> None:
    async def body() -> None:
        manager = SlowManager()
        store = ApprovalStore()
        broker = _broker(manager, store)
        approval_id = (await _issue(broker))["approval_request_id"]
        assert store.decide(approval_id, "approve").ok
        first, second = await asyncio.gather(
            _issue(broker, approval_request_id=approval_id),
            _issue(broker, approval_request_id=approval_id),
        )
        statuses = sorted(item["status"] for item in (first, second))
        assert statuses == ["denied", "ok"]
        assert len(manager.calls) == 1

    asyncio.run(body())


def test_mcp_claims_carry_room_mode_into_the_session_broker() -> None:
    async def body() -> None:
        manager = RecordingManager()
        store = ApprovalStore()
        registry = _registry()
        deps = BrokerDeps(
            registry=registry,
            manager=manager,  # type: ignore[arg-type]
            policy=TierPolicy(),
            audit=InMemoryAuditSink(),
            approvals=store,
            default_env=EnvScope.PROD_SAFE,
        )
        tokens = {
            ROOM_BEARER: HostIdentity(
                "room", profile="default_assistant", approval_mode=ApprovalMode.ROOM_APPROVAL
            ),
            LEGACY_BEARER: HostIdentity("legacy-host"),
        }
        verifier = JanusTokenVerifier(tokens)
        room_access = await verifier.verify_token(ROOM_BEARER)
        assert room_access is not None
        assert room_access.claims["janus_approval_mode"] == "room_approval"
        room_identity = McpSessionPool.identity_from_access_token(room_access)
        room_broker = McpSessionPool(deps).state_for(room_identity, "sess").broker
        blocked = await room_broker.capability_call(
            "fake.write", _args(), "file the note", confirmed=True
        )
        assert blocked["status"] == "needs_confirmation"
        assert manager.calls == []
        _no_secret(blocked)

        legacy_access = await verifier.verify_token(LEGACY_BEARER)
        assert legacy_access is not None
        assert legacy_access.claims["janus_approval_mode"] == "legacy"
        legacy_identity = McpSessionPool.identity_from_access_token(legacy_access)
        legacy_broker = McpSessionPool(deps).state_for(legacy_identity, "other").broker
        allowed = await legacy_broker.capability_call(
            "fake.write", {"text": "legacy"}, "r", confirmed=True
        )
        assert allowed["status"] == "ok"
        assert len(manager.calls) == 1

        missing = AccessToken(
            token="x",  # noqa: S106
            client_id="host",
            scopes=[],
            claims={"janus_label": "host"},
        )
        resolved = McpSessionPool.identity_from_access_token(missing)
        assert resolved.approval_mode is ApprovalMode.LEGACY
        forged = AccessToken(
            token="x",  # noqa: S106
            client_id="room",
            scopes=[],
            claims={"janus_label": "room", "janus_approval_mode": "confirm"},
        )
        try:
            McpSessionPool.identity_from_access_token(forged)
        except PermissionError:
            pass
        else:
            raise AssertionError("unknown approval mode was accepted")

    asyncio.run(body())


def _http_client(manager: RecordingManager, store: ApprovalStore) -> TestClient:
    registry = _registry()
    deps = BrokerDeps(
        registry=registry,
        manager=manager,  # type: ignore[arg-type]
        policy=ProfilePolicyEngine(),
        audit=InMemoryAuditSink(),
        approvals=store,
        default_env=EnvScope.PROD_SAFE,
    )
    tokens = {
        ROOM_BEARER: HostIdentity(
            "room", profile="default_assistant", approval_mode=ApprovalMode.ROOM_APPROVAL
        ),
        LEGACY_BEARER: HostIdentity("legacy-host", profile="default_assistant"),
    }
    return TestClient(create_rest_app(deps, tokens, approval_token=OPERATOR_BEARER))


def _call(client: TestClient, bearer: str, body: dict[str, Any]) -> Any:
    return client.post(
        "/v1/capability/call",
        headers={"Authorization": f"Bearer {bearer}"},
        json=body,
    )


def test_http_operator_sees_exact_arguments_and_room_bearer_cannot() -> None:
    manager = RecordingManager()
    client = _http_client(manager, ApprovalStore())
    room_headers = {"Authorization": f"Bearer {ROOM_BEARER}"}
    operator_headers = {"Authorization": f"Bearer {OPERATOR_BEARER}"}
    issued = _call(
        client,
        ROOM_BEARER,
        {
            "capability_id": "fake.write",
            "reason": "file the note",
            "confirm": True,
            "arguments": _args(),
        },
    ).json()
    assert issued["status"] == "needs_confirmation"
    assert manager.calls == []
    _no_secret(issued)
    approval_id = issued["approval_request_id"]

    for headers in (room_headers, {"Authorization": f"Bearer {LEGACY_BEARER}"}):
        rejected = client.get(f"/v1/approvals/{approval_id}", headers=headers)
        assert rejected.status_code == 401
        _no_secret(rejected.json())
        decided = client.post(
            f"/v1/approvals/{approval_id}/decision",
            headers=headers,
            json={"decision": "approve", "arguments": {"note": "pwned"}},
        )
        assert decided.status_code == 401
        _no_secret(decided.json())

    capability = _call(
        client,
        OPERATOR_BEARER,
        {"capability_id": "fake.write", "reason": "file the note", "confirm": True},
    )
    assert capability.status_code == 401

    viewed = client.get(f"/v1/approvals/{approval_id}", headers=operator_headers)
    assert viewed.status_code == 200
    assert viewed.headers["cache-control"] == "no-store"
    assert viewed.json()["status"] == "pending"
    assert viewed.json()["arguments"] == {
        "b": 1,
        "nested": {"c": 2, "d": 1},
        "note": NEEDLE,
    }
    assert viewed.json()["reason"] == "file the note"
    assert viewed.json()["tool_name"] == "echo"

    verb = client.post(
        f"/v1/approvals/{approval_id}/decision",
        headers=operator_headers,
        json={"decision": "approved"},
    )
    assert verb.status_code == 422
    still_pending = client.get(f"/v1/approvals/{approval_id}", headers=operator_headers).json()
    assert still_pending["status"] == "pending"

    approved = client.post(
        f"/v1/approvals/{approval_id}/decision",
        headers=operator_headers,
        json={"decision": "approve", "arguments": {"note": "pwned"}},
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"
    assert approved.json()["arguments"]["note"] == NEEDLE

    pwned = _call(
        client,
        ROOM_BEARER,
        {
            "capability_id": "fake.write",
            "reason": "file the note",
            "confirm": True,
            "approval_request_id": approval_id,
            "arguments": {"note": "pwned"},
        },
    ).json()
    assert pwned["status"] == "denied"
    assert manager.calls == []
    _no_secret(pwned)

    exact = _call(
        client,
        ROOM_BEARER,
        {
            "capability_id": "fake.write",
            "reason": "file the note",
            "approval_request_id": approval_id,
            "arguments": _reordered(),
        },
    ).json()
    assert exact["status"] == "ok"
    assert len(manager.calls) == 1
    _no_secret(exact)
    audit = client.get("/v1/audit/recent", headers=room_headers).json()
    _no_secret(audit)

    replay = _call(
        client,
        ROOM_BEARER,
        {
            "capability_id": "fake.write",
            "reason": "file the note",
            "confirm": True,
            "approval_request_id": approval_id,
            "arguments": _reordered(),
        },
    ).json()
    assert replay["status"] == "denied"
    assert len(manager.calls) == 1

    legacy = _call(
        client,
        LEGACY_BEARER,
        {
            "capability_id": "fake.write",
            "reason": "legacy write",
            "confirm": True,
            "arguments": {"text": "legacy"},
        },
    ).json()
    assert legacy["status"] == "ok"
    assert "approval_request_id" not in legacy
    assert len(manager.calls) == 2


def test_http_expiry_and_deny_do_not_execute() -> None:
    clocks = Clocks()
    manager = RecordingManager()
    client = _http_client(
        manager, ApprovalStore(clock=clocks.monotonic, wall_clock=clocks.wall_clock)
    )
    operator = {"Authorization": f"Bearer {OPERATOR_BEARER}"}
    issued = _call(
        client,
        ROOM_BEARER,
        {"capability_id": "fake.write", "reason": "later", "arguments": {"text": "x"}},
    ).json()
    approval_id = issued["approval_request_id"]
    clocks.advance(15 * 60)
    denied_decision = client.post(
        f"/v1/approvals/{approval_id}/decision",
        headers=operator,
        json={"decision": "approve"},
    )
    assert denied_decision.status_code == 409
    _no_secret(denied_decision.json())
    late = _call(
        client,
        ROOM_BEARER,
        {
            "capability_id": "fake.write",
            "reason": "later",
            "arguments": {"text": "x"},
            "approval_request_id": approval_id,
            "confirm": True,
        },
    ).json()
    assert late["status"] == "denied"
    assert manager.calls == []

    fresh = _call(
        client,
        ROOM_BEARER,
        {"capability_id": "fake.write", "reason": "no", "arguments": {"text": "x"}},
    ).json()
    decision = client.post(
        f"/v1/approvals/{fresh['approval_request_id']}/decision",
        headers=operator,
        json={"decision": "deny"},
    )
    assert decision.status_code == 200
    assert decision.json()["status"] == "denied"
    blocked = _call(
        client,
        ROOM_BEARER,
        {
            "capability_id": "fake.write",
            "reason": "no",
            "arguments": {"text": "x"},
            "confirm": True,
            "approval_request_id": fresh["approval_request_id"],
        },
    ).json()
    assert blocked["status"] == "denied"
    assert manager.calls == []
