"""Tests for the REST shim: per-host bearer auth + broker routing.

These exercise auth and the endpoints that don't require a live downstream
(search, explain, server_list, audit, and the deny / needs-confirmation call
paths, which return before touching a downstream). Live execution through REST
uses the identical broker path already covered by test_broker.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

from janus.audit import InMemoryAuditSink
from janus.downstream import DownstreamClientManager
from janus.policy import ProfilePolicyEngine
from janus.registry import (
    Capability,
    EnvScope,
    Registry,
    RiskTier,
    Server,
    Transport,
)
from janus.server_rest import BrokerDeps, HostIdentity, create_rest_app

FAKE = str(Path(__file__).parent / "_fake_downstream.py")

TOKENS = {
    "tok-a": HostIdentity("host-a", profile="default_assistant"),
    "tok-b": HostIdentity("host-b", profile="autonomous_agent"),
}
AUTH_A = {"Authorization": "Bearer tok-a"}
AUTH_B = {"Authorization": "Bearer tok-b"}


def _cap(cid: str, risk: RiskTier) -> Capability:
    return Capability(
        id=cid,
        server_id="fake",
        downstream_tool_name="echo",
        title=cid,
        summary=f"{cid} summary",
        risk=risk,
        env_scope=[EnvScope.DEV, EnvScope.PROD_SAFE],
        approved=True,
    )


def _client() -> TestClient:
    server = Server(
        id="fake",
        display_name="Fake",
        transport=Transport.STDIO,
        command=sys.executable,
        args=[FAKE],
        risk_ceiling=RiskTier.EXTERNAL_WRITE,
        default_env_scope=[EnvScope.DEV, EnvScope.PROD_SAFE],
    )
    registry = Registry(
        servers={"fake": server},
        capabilities={
            "fake.read": _cap("fake.read", RiskTier.READ_ONLY),
            "fake.write": _cap("fake.write", RiskTier.LOCAL_WRITE),
            "fake.ext": _cap("fake.ext", RiskTier.EXTERNAL_WRITE),
        },
    )
    deps = BrokerDeps(
        registry=registry,
        manager=DownstreamClientManager(registry.servers),
        policy=ProfilePolicyEngine(),
        audit=InMemoryAuditSink(),
        default_env=EnvScope.PROD_SAFE,
    )
    return TestClient(create_rest_app(deps, TOKENS))


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
def test_missing_token_401() -> None:
    resp = _client().post("/v1/capability/search", json={"query": "x"})
    assert resp.status_code == 401


def test_bad_token_401() -> None:
    resp = _client().post(
        "/v1/capability/search",
        json={"query": "x"},
        headers={"Authorization": "Bearer nope"},
    )
    assert resp.status_code == 401


def test_health_is_unauthenticated() -> None:
    resp = _client().get("/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
def test_search_authed() -> None:
    resp = _client().post(
        "/v1/capability/search", json={"query": "read"}, headers=AUTH_A
    )
    assert resp.status_code == 200
    ids = {r["capability_id"] for r in resp.json()["results"]}
    assert "fake.read" in ids
    assert "fake.ext" not in ids  # denied for default_assistant


def test_policy_explain_is_per_host_profile() -> None:
    client = _client()
    a = client.post(
        "/v1/policy/explain", json={"capability_id": "fake.write"}, headers=AUTH_A
    ).json()
    b = client.post(
        "/v1/policy/explain", json={"capability_id": "fake.write"}, headers=AUTH_B
    ).json()
    assert a["decision"] == "confirm"  # default_assistant
    assert b["decision"] == "allow"  # autonomous_agent allows local_write


def test_call_confirm_tier_needs_confirmation() -> None:
    resp = _client().post(
        "/v1/capability/call",
        json={"capability_id": "fake.write", "reason": "r", "arguments": {"text": "x"}},
        headers=AUTH_A,
    )
    assert resp.json()["status"] == "needs_confirmation"


def test_call_denied_tier() -> None:
    resp = _client().post(
        "/v1/capability/call",
        json={"capability_id": "fake.ext", "reason": "r"},
        headers=AUTH_A,
    )
    assert resp.json()["status"] == "denied"


def test_call_unknown_capability() -> None:
    resp = _client().post(
        "/v1/capability/call",
        json={"capability_id": "ghost", "reason": "r"},
        headers=AUTH_A,
    )
    assert resp.json()["status"] == "error"


def test_server_list_authed() -> None:
    resp = _client().get("/v1/server/list", headers=AUTH_A)
    assert resp.status_code == 200
    assert resp.json()["servers"][0]["server_id"] == "fake"


def test_audit_is_session_scoped_per_host() -> None:
    client = _client()
    client.post(
        "/v1/capability/call",
        json={"capability_id": "fake.ext", "reason": "r"},
        headers=AUTH_A,
    )
    a = client.get("/v1/audit/recent", headers=AUTH_A).json()
    b = client.get("/v1/audit/recent", headers=AUTH_B).json()
    assert a["count"] == 1
    assert b["count"] == 0  # host-b never called -> nothing in its session
