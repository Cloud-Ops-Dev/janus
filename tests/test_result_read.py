"""Lossless, policy-bound reads of long sanitized results."""

from __future__ import annotations

import asyncio
import hashlib

import pytest

from janus.audit import InMemoryAuditSink
from janus.downstream.client_manager import DownstreamResult
from janus.policy import ProfilePolicyEngine
from janus.registry import Capability, EnvScope, Registry, RiskTier, Server, Transport, TrustLevel
from janus.security import OutputSanitizer, SecretRedactor
from janus.security.output_sanitizer import _UNTRUSTED_FOOTER, _UNTRUSTED_HEADER
from janus.server_mcp import create_mcp_server
from janus.server_rest import BrokerDeps, HostIdentity, create_rest_app
from janus.session_mcp import McpSessionPool


class Manager:
    def __init__(self, text: str) -> None:
        self.text = text

    async def call(self, server_id: str, tool: str, arguments: dict) -> DownstreamResult:
        return DownstreamResult(False, self.text, {"text": self.text})


def setup(text: str, *, trust: TrustLevel = TrustLevel.FIRST_PARTY):
    server = Server(
        id="fake", display_name="Fake", transport=Transport.STDIO,
        command="false", trust_level=trust, risk_ceiling=RiskTier.READ_ONLY,
        default_env_scope=[EnvScope.PROD_SAFE],
    )
    cap = Capability(
        id="fake.read", server_id="fake", downstream_tool_name="read",
        title="Read", summary="Read", risk=RiskTier.READ_ONLY,
        env_scope=[EnvScope.PROD_SAFE], approved=True,
    )
    registry = Registry(servers={"fake": server}, capabilities={cap.id: cap})
    redactor = SecretRedactor()
    sanitizer = OutputSanitizer(redactor)
    deps = BrokerDeps(
        registry, Manager(text), ProfilePolicyEngine(), InMemoryAuditSink(),
        sanitizer=sanitizer,
    )
    return deps, redactor


@pytest.mark.parametrize("length", [0, 20_000, 20_001, 40_000, 40_001])
def test_boundaries_and_repeated_reads(length: int) -> None:
    deps, _ = setup("字" * length)
    broker = deps.broker_for(HostIdentity("one"))
    initial = asyncio.run(broker.capability_call("fake.read", {}, "test"))
    if length <= 20_000:
        assert initial == {
            "status": "ok", "capability_id": "fake.read", "is_error": False,
            "text": "字" * length, "structured": {"text": "字" * length},
        }
        return
    meta = initial["truncation"]
    assert len(initial["text"]) <= 20_000
    assert initial["structured"] == {"text": "字" * length}  # unchanged contract
    assert meta["total_chars"] == length
    assert meta["returned_range"] == [0, 20_000]
    assert meta["next_offset"] == 20_000
    assert meta["sha256"] == hashlib.sha256(("字" * length).encode()).hexdigest()
    repeat = broker.result_read(meta["handle"], 20_000, 173)
    assert repeat == broker.result_read(meta["handle"], 20_000, 173)
    assert broker.result_read(meta["handle"], 20_000, 20_001)["status"] == "error"
    pieces = [initial["text"]]
    offset = meta["next_offset"]
    while offset is not None:
        chunk = broker.result_read(meta["handle"], offset, 20_000)
        assert len(chunk["text"]) <= 20_000
        pieces.append(chunk["text"])
        offset = chunk["truncation"]["next_offset"]
    assert "".join(pieces).encode() == ("字" * length).encode()


def test_unicode_markdown_redaction_and_untrusted_wrapper() -> None:
    secret = "SYNTHETIC_SECRET_12345"  # noqa: S105
    text = "# Report\n" + ("汉字🛰️ café\n" * 5000) + secret + "\n## FINAL SECTION ✅"
    deps, redactor = setup(text, trust=TrustLevel.THIRD_PARTY)
    redactor.register(secret)
    broker = deps.broker_for(HostIdentity("one"))
    first = asyncio.run(broker.capability_call("fake.read", {}, "test"))
    meta = first["truncation"]
    wrapped = [first["text"]]
    offset = meta["next_offset"]
    while offset is not None:
        chunk = broker.result_read(meta["handle"], offset, 19997)
        assert secret not in chunk["text"]
        wrapped.append(chunk["text"])
        offset = chunk["truncation"]["next_offset"]
    assert len(wrapped) >= 3  # a middle slice exists and is checked below
    pieces = []
    for piece in wrapped:
        # Every third-party slice, middle ones included, carries the untrusted wrapper.
        assert piece.startswith(_UNTRUSTED_HEADER + "\n")
        assert piece.endswith("\n" + _UNTRUSTED_FOOTER)
        pieces.append(piece[len(_UNTRUSTED_HEADER) + 1 : -(len(_UNTRUSTED_FOOTER) + 1)])
    full = "".join(pieces)
    assert secret not in full
    assert "«redacted»" in full
    assert full.startswith("# Report")
    assert full.endswith("## FINAL SECTION ✅")
    assert meta["sha256"] == hashlib.sha256(full.encode()).hexdigest()


def test_foreign_denied_expired_and_evicted_handles() -> None:
    deps, _ = setup("x" * 20_001)
    now = [0.0]
    deps.sanitizer = OutputSanitizer(
        SecretRedactor(), ttl_seconds=10, max_entries=1, clock=lambda: now[0]
    )
    owner = deps.broker_for(HostIdentity("one"))
    handle = asyncio.run(owner.capability_call("fake.read", {}, "test"))["truncation"]["handle"]
    assert deps.broker_for(HostIdentity("two")).result_read(handle, 0, 100)["status"] == "error"
    assert owner.result_read("unknown", 0, 100)["status"] == "error"
    cap = deps.registry.capabilities["fake.read"]
    deps.registry.capabilities["fake.read"] = cap.model_copy(update={"quarantined": True})
    assert owner.result_read(handle, 0, 100)["status"] == "denied"
    deps.registry.capabilities["fake.read"] = cap
    now[0] = 10.0
    assert owner.result_read(handle, 0, 100)["status"] == "error"
    now[0] = 11.0
    old = asyncio.run(owner.capability_call("fake.read", {}, "test"))["truncation"]["handle"]
    asyncio.run(owner.capability_call("fake.read", {}, "test"))
    assert owner.result_read(old, 0, 100)["status"] == "error"


def test_read_is_registered_on_mcp_and_rest() -> None:
    deps, _ = setup("x" * 20_001)
    server = create_mcp_server(deps.broker_for(HostIdentity("one")), dynamic_exposure=False)
    assert "result_read" in {tool.name for tool in asyncio.run(server.list_tools())}
    from fastapi.testclient import TestClient

    client = TestClient(create_rest_app(deps, {"synthetic-token": HostIdentity("one")}))
    headers = {"Authorization": "Bearer synthetic-token"}
    first = client.post(
        "/v1/capability/call", headers=headers,
        json={"capability_id": "fake.read", "reason": "test"},
    ).json()
    handle = first["truncation"]["handle"]
    read = client.post(
        "/v1/result/read", headers=headers,
        json={"handle": handle, "offset": 20_000, "limit": 100},
    ).json()
    assert read["text"] == "x"


def test_mcp_session_handle_is_bound_to_session() -> None:
    deps, _ = setup("x" * 20_001)
    pool = McpSessionPool(deps)
    owner = pool.state_for(HostIdentity("one"), "session-one").broker
    other_session = pool.state_for(HostIdentity("one"), "session-two").broker
    handle = asyncio.run(owner.capability_call("fake.read", {}, "test"))["truncation"]["handle"]
    assert owner.result_read(handle, 20_000, 100)["text"] == "x"
    assert other_session.result_read(handle, 20_000, 100)["status"] == "error"


def test_result_larger_than_store_cap_falls_back_to_flagged_truncation() -> None:
    deps, _ = setup("x" * 20_001)
    deps.sanitizer = OutputSanitizer(SecretRedactor(), max_entry_chars=20_000)
    result = asyncio.run(
        deps.broker_for(HostIdentity("one")).capability_call("fake.read", {}, "test")
    )
    # Not an error (the pre-continuation behavior was a truncated result): the head is
    # returned, and the metadata says plainly that no continuation handle exists.
    assert result["status"] == "ok"
    assert result["text"].startswith("x" * 20_000)
    assert "chars truncated" in result["text"]
    meta = result["truncation"]
    assert meta["handle"] is None and meta["next_offset"] is None
    assert meta["total_chars"] == 20_001
    assert meta["reason"] == "exceeds continuation store limit"
