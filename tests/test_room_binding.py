"""Per-session room-binding header on room_approval grants.

The header is read from the authenticated MCP HTTP request. These tests
cover the operator view, member and audit concealment, a missing header,
and malformed or changed headers. Legacy confirm behavior is included so
the pin cannot spill onto identities that do not use room approval.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import socket
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from test_room_approval import (
    LEGACY_BEARER,
    OPERATOR_BEARER,
    ROOM_BEARER,
    ApprovalMode,
    RecordingManager,
    TierPolicy,
    _args,
    _broker,
    _registry,
    _view,
)

from janus.approval import ApprovalStore
from janus.audit import InMemoryAuditSink
from janus.registry import EnvScope
from janus.room_binding import (
    MALFORMED_ROOM_BINDING,
    ROOM_BINDING_LENGTH,
    ROOM_BINDING_MISMATCH,
    RoomBindingRejected,
    interpret_header_values,
    parse_room_binding,
    presented_room_binding,
)
from janus.server_rest import BrokerDeps, HostIdentity, create_rest_app
from janus.session_mcp import McpSessionPool, build_session_mcp_server

WIRE_HEADER = "X-Retinue-Room-Binding"


class _OverlayClient(httpx.AsyncClient):
    """HTTP client whose next requests can replace the room-binding header."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.overlay: dict[str, str] = {}

    async def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        for key, value in self.overlay.items():
            request.headers[key] = value
        return await super().send(request, **kwargs)


def _mint() -> str:
    value = secrets.token_urlsafe(32)
    assert len(value) == ROOM_BINDING_LENGTH
    return value


def _no_secret(payload: object, *secrets_to_hide: str) -> None:
    rendered = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    for secret in secrets_to_hide:
        assert secret not in rendered, "room binding leaked"


def _payload(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        return structured
    data = getattr(result, "data", None)
    if isinstance(data, dict):
        return data
    raise AssertionError("tool result had no dict payload")


def _rendered(result: Any) -> str:
    content = getattr(result, "content", None) or []
    texts = [getattr(block, "text", str(block)) for block in content]
    return json.dumps(
        {
            "data": getattr(result, "data", None),
            "structured": getattr(result, "structured_content", None),
            "content": texts,
        },
        default=str,
    )


def _deps(manager: RecordingManager, store: ApprovalStore, audit: InMemoryAuditSink) -> BrokerDeps:
    return BrokerDeps(
        registry=_registry(),
        manager=manager,  # type: ignore[arg-type]
        policy=TierPolicy(),
        audit=audit,
        approvals=store,
        default_env=EnvScope.PROD_SAFE,
    )


def _tokens() -> dict[str, HostIdentity]:
    return {
        ROOM_BEARER: HostIdentity(
            "room", profile="default_assistant", approval_mode=ApprovalMode.ROOM_APPROVAL
        ),
        LEGACY_BEARER: HostIdentity("legacy-host", profile="default_assistant"),
    }


@asynccontextmanager
async def _mcp_server(
    deps: BrokerDeps, tokens: dict[str, HostIdentity]
) -> AsyncIterator[tuple[str, McpSessionPool]]:
    mcp, pool = build_session_mcp_server(deps, tokens, ttl_seconds=60)
    app = mcp.http_app(path="/", transport="http")
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    sock.setblocking(False)
    port = int(sock.getsockname()[1])
    server = uvicorn.Server(uvicorn.Config(app, log_level="critical", lifespan="on"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        for _ in range(200):
            if server.started:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("test MCP HTTP server did not start")
        yield f"http://127.0.0.1:{port}/", pool
    finally:
        server.should_exit = True
        await task


def _transport(
    url: str,
    headers: dict[str, str],
    factory: Callable[..., _OverlayClient] | None = None,
) -> StreamableHttpTransport:
    return StreamableHttpTransport(url, headers=headers, httpx_client_factory=factory)


def _factory(holder: dict[str, _OverlayClient]) -> Callable[..., _OverlayClient]:
    def factory(**kwargs: Any) -> _OverlayClient:
        client = _OverlayClient(**kwargs)
        holder["client"] = client
        return client

    return factory


def test_parser_accepts_only_the_minted_shape() -> None:
    assert presented_room_binding() is None
    minted = _mint()
    assert parse_room_binding(minted) == minted
    assert interpret_header_values([minted]) == minted
    assert interpret_header_values([]) is None

    samples = [
        "MALFORMED-ROOM-BINDING-SENTINEL",
        minted[:-1],
        minted + "a",
        " " + minted,
        minted + "=",
        "",
        minted + "\n",
    ]
    for sample in samples:
        with pytest.raises(RoomBindingRejected, match="^malformed room binding$") as caught:
            parse_room_binding(sample)
        if sample:
            assert sample not in str(caught.value)
            assert sample not in repr(caught.value)

    other = _mint()
    with pytest.raises(RoomBindingRejected, match="^malformed room binding$") as duplicate:
        interpret_header_values([minted, other])
    assert minted not in str(duplicate.value)
    assert other not in str(duplicate.value)


def test_grant_stores_binding_and_consume_requires_the_same_value() -> None:
    store = ApprovalStore()
    binding = _mint()
    other = _mint()
    approval_id = store.issue(
        identity="room",
        profile="default_assistant",
        session_id="sess",
        capability_id="fake.write",
        server_id="fake",
        tool_name="echo",
        env="prod_safe",
        arguments={"note": "x"},
        reason="because",
        gate_reason="confirm",
        trifecta=False,
        room_binding=binding,
    )
    view = _view(store, approval_id)
    assert view.room_binding == binding
    assert view.as_dict()["room_binding"] == binding

    pending = store.consume(
        approval_id,
        identity="room",
        profile="default_assistant",
        session_id="sess",
        capability_id="fake.write",
        server_id="fake",
        tool_name="echo",
        env="prod_safe",
        arguments={"note": "x"},
        reason="because",
        room_binding=binding,
    )
    assert pending.ok is False
    assert _view(store, approval_id).status == "pending"
    assert store.decide(approval_id, "approve").ok is True

    mismatched = store.consume(
        approval_id,
        identity="room",
        profile="default_assistant",
        session_id="sess",
        capability_id="fake.write",
        server_id="fake",
        tool_name="echo",
        env="prod_safe",
        arguments={"note": "x"},
        reason="because",
        room_binding=other,
    )
    assert mismatched.ok is False
    assert _view(store, approval_id).status == "approved"
    missing = store.consume(
        approval_id,
        identity="room",
        profile="default_assistant",
        session_id="sess",
        capability_id="fake.write",
        server_id="fake",
        tool_name="echo",
        env="prod_safe",
        arguments={"note": "x"},
        reason="because",
        room_binding=None,
    )
    assert missing.ok is False
    assert _view(store, approval_id).status == "approved"

    consumed = store.consume(
        approval_id,
        identity="room",
        profile="default_assistant",
        session_id="sess",
        capability_id="fake.write",
        server_id="fake",
        tool_name="echo",
        env="prod_safe",
        arguments={"note": "x"},
        reason="because",
        room_binding=binding,
    )
    assert consumed.ok is True
    assert _view(store, approval_id).status == "consumed"
    assert _view(store, approval_id).room_binding == binding

    bad = "not-a-binding"
    with pytest.raises(RoomBindingRejected, match="^malformed room binding$") as caught:
        store.issue(
            identity="room",
            profile="default_assistant",
            session_id="sess",
            capability_id="fake.write",
            server_id="fake",
            tool_name="echo",
            env="prod_safe",
            arguments={"note": "x"},
            reason="because",
            gate_reason="confirm",
            trifecta=False,
            room_binding=bad,
        )
    assert bad not in str(caught.value)


def test_tool_arguments_cannot_set_the_binding() -> None:
    async def body() -> None:
        manager = RecordingManager()
        store = ApprovalStore()
        broker = _broker(manager, store)
        decoy = _mint()
        arguments = _args()
        arguments["room_binding"] = decoy
        issued = await broker.capability_call(
            "fake.write", arguments, "file the note", confirmed=True
        )
        assert issued["status"] == "needs_confirmation"
        assert manager.calls == []
        view = _view(store, issued["approval_request_id"])
        assert view.room_binding is None
        assert view.arguments["room_binding"] == decoy
        _no_secret(issued, decoy)

        binding = _mint()
        broker.pin_room_binding(binding)
        pinned = await broker.capability_call(
            "fake.write", arguments, "file the note", confirmed=True
        )
        pinned_view = _view(store, pinned["approval_request_id"])
        assert pinned_view.room_binding == binding
        assert pinned_view.arguments["room_binding"] == decoy
        _no_secret(pinned, binding, decoy)
        _no_secret(
            json.dumps([entry.__dict__ for entry in broker._audit.recent(10)]),
            binding,
            decoy,
        )

        other = _broker(manager, store)
        other.pin_room_binding(_mint())
        store.decide(pinned["approval_request_id"], "approve")
        denied = await other.capability_call(
            "fake.write",
            arguments,
            "file the note",
            confirmed=True,
            approval_request_id=pinned["approval_request_id"],
        )
        assert denied["status"] == "denied"
        assert manager.calls == []
        assert _view(store, pinned["approval_request_id"]).status == "approved"

        executed = await broker.capability_call(
            "fake.write",
            arguments,
            "file the note",
            approval_request_id=pinned["approval_request_id"],
        )
        assert executed["status"] == "ok"
        assert len(manager.calls) == 1
        _no_secret(executed, binding, decoy)

    asyncio.run(body())


def test_rest_call_cannot_set_the_binding() -> None:
    manager = RecordingManager()
    store = ApprovalStore()
    audit = InMemoryAuditSink()
    client = TestClient(
        create_rest_app(
            _deps(manager, store, audit),
            _tokens(),
            approval_token=OPERATOR_BEARER,
        )
    )
    binding = _mint()
    issued = client.post(
        "/v1/capability/call",
        headers={
            "Authorization": f"Bearer {ROOM_BEARER}",
            WIRE_HEADER: binding,
        },
        json={
            "capability_id": "fake.write",
            "reason": "file the note",
            "confirm": True,
            "arguments": _args(),
            "room_binding": binding,
        },
    ).json()
    assert issued["status"] == "needs_confirmation"
    assert manager.calls == []
    _no_secret(issued, binding)
    view = client.get(
        f"/v1/approvals/{issued['approval_request_id']}",
        headers={"Authorization": f"Bearer {OPERATOR_BEARER}"},
    )
    assert view.status_code == 200
    assert view.json()["room_binding"] is None
    _no_secret(view.json(), binding)
    _no_secret([entry.__dict__ for entry in audit.recent(10)], binding)


def test_header_is_returned_only_on_the_operator_view(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="janus")

    async def body() -> None:
        manager = RecordingManager()
        store = ApprovalStore()
        audit = InMemoryAuditSink()
        deps = _deps(manager, store, audit)
        tokens = _tokens()
        binding = _mint()
        decoy = _mint()
        arguments = _args()
        arguments["room_binding"] = decoy
        async with _mcp_server(deps, tokens) as (url, pool):
            rest = TestClient(create_rest_app(deps, tokens, approval_token=OPERATOR_BEARER))
            async with Client(_transport(url, {WIRE_HEADER: binding}), auth=ROOM_BEARER) as client:
                issued_result = await client.call_tool(
                    "capability_call",
                    {
                        "capability_id": "fake.write",
                        "arguments": arguments,
                        "reason": "file the note",
                        "confirm": True,
                    },
                )
                issued = _payload(issued_result)
                assert issued["status"] == "needs_confirmation"
                assert manager.calls == []
                _no_secret(_rendered(issued_result), binding, decoy)
                approval_id = issued["approval_request_id"]

                hidden = rest.get(
                    f"/v1/approvals/{approval_id}",
                    headers={"Authorization": f"Bearer {ROOM_BEARER}"},
                )
                assert hidden.status_code == 401
                _no_secret(hidden.json(), binding, decoy)

                viewed = rest.get(
                    f"/v1/approvals/{approval_id}",
                    headers={"Authorization": f"Bearer {OPERATOR_BEARER}"},
                )
                assert viewed.status_code == 200
                assert viewed.headers["cache-control"] == "no-store"
                assert viewed.json()["room_binding"] == binding
                assert viewed.json()["arguments"]["room_binding"] == decoy
                assert viewed.json()["arguments"]["note"]

                decided = rest.post(
                    f"/v1/approvals/{approval_id}/decision",
                    headers={"Authorization": f"Bearer {OPERATOR_BEARER}"},
                    json={"decision": "approve"},
                )
                assert decided.status_code == 200
                assert decided.json()["status"] == "approved"
                assert decided.json()["room_binding"] == binding

                executed_result = await client.call_tool(
                    "capability_call",
                    {
                        "capability_id": "fake.write",
                        "arguments": arguments,
                        "reason": "file the note",
                        "approval_request_id": approval_id,
                    },
                )
                executed = _payload(executed_result)
                assert executed["status"] == "ok"
                assert len(manager.calls) == 1
                _no_secret(_rendered(executed_result), binding, decoy)
                assert "room_binding" not in executed

                audit_result = await client.call_tool("audit_recent", {"limit": 20})
                _no_secret(_rendered(audit_result), binding, decoy)
                _no_secret([entry.__dict__ for entry in audit.recent(20)], binding, decoy)

                states = list(pool._states.values())
                assert [state.room_binding for state in states] == [binding]
                for state in states:
                    assert binding not in repr(state)

        _no_secret(caplog.text, binding, decoy)

    asyncio.run(body())


def test_absent_header_leaves_the_claim_unset() -> None:
    async def body() -> None:
        manager = RecordingManager()
        store = ApprovalStore()
        audit = InMemoryAuditSink()
        deps = _deps(manager, store, audit)
        tokens = _tokens()
        async with _mcp_server(deps, tokens) as (url, _pool):
            rest = TestClient(create_rest_app(deps, tokens, approval_token=OPERATOR_BEARER))
            async with Client(_transport(url, {}), auth=ROOM_BEARER) as client:
                issued = _payload(
                    await client.call_tool(
                        "capability_call",
                        {
                            "capability_id": "fake.write",
                            "arguments": {"text": "plain"},
                            "reason": "no header",
                            "confirm": True,
                        },
                    )
                )
            assert issued["status"] == "needs_confirmation"
            assert manager.calls == []
            viewed = rest.get(
                f"/v1/approvals/{issued['approval_request_id']}",
                headers={"Authorization": f"Bearer {OPERATOR_BEARER}"},
            )
            assert viewed.status_code == 200
            assert viewed.json()["room_binding"] is None

    asyncio.run(body())


def test_malformed_and_changed_headers_fail_closed() -> None:
    async def body() -> None:
        manager = RecordingManager()
        store = ApprovalStore()
        audit = InMemoryAuditSink()
        deps = _deps(manager, store, audit)
        tokens = _tokens()
        binding = _mint()
        changed = _mint()
        malformed = "MALFORMED-ROOM-BINDING-SENTINEL"
        holder: dict[str, _OverlayClient] = {}
        async with _mcp_server(deps, tokens) as (url, _pool):
            rest = TestClient(create_rest_app(deps, tokens, approval_token=OPERATOR_BEARER))
            async with Client(
                _transport(url, {WIRE_HEADER: binding}, _factory(holder)),
                auth=ROOM_BEARER,
            ) as client:
                issued = _payload(
                    await client.call_tool(
                        "capability_call",
                        {
                            "capability_id": "fake.write",
                            "arguments": {"text": "first"},
                            "reason": "pin",
                            "confirm": True,
                        },
                    )
                )
                assert issued["status"] == "needs_confirmation"
                approval_id = issued["approval_request_id"]
                assert (
                    rest.get(
                        f"/v1/approvals/{approval_id}",
                        headers={"Authorization": f"Bearer {OPERATOR_BEARER}"},
                    ).json()["room_binding"]
                    == binding
                )

                holder["client"].overlay[WIRE_HEADER] = changed
                changed_result = await client.call_tool(
                    "capability_call",
                    {
                        "capability_id": "fake.write",
                        "arguments": {"text": "second"},
                        "reason": "retarget",
                        "confirm": True,
                    },
                )
                changed_payload = _payload(changed_result)
                assert changed_payload["status"] == "error"
                assert changed_payload["error"] == ROOM_BINDING_MISMATCH
                _no_secret(_rendered(changed_result), binding, changed)
                assert len(store._records) == 1
                assert _view(store, approval_id).room_binding == binding
                assert manager.calls == []

                holder["client"].overlay[WIRE_HEADER] = malformed
                malformed_result = await client.call_tool(
                    "capability_call",
                    {
                        "capability_id": "fake.write",
                        "arguments": {"text": "third"},
                        "reason": "corrupt",
                        "confirm": True,
                    },
                )
                malformed_payload = _payload(malformed_result)
                assert malformed_payload["status"] == "error"
                assert malformed_payload["error"] == MALFORMED_ROOM_BINDING
                _no_secret(_rendered(malformed_result), binding, changed, malformed)
                assert len(store._records) == 1
                assert manager.calls == []

                holder["client"].overlay.clear()
                again = _payload(
                    await client.call_tool(
                        "capability_call",
                        {
                            "capability_id": "fake.write",
                            "arguments": {"text": "first"},
                            "reason": "pin",
                            "confirm": True,
                        },
                    )
                )
                assert again["status"] == "needs_confirmation"
                assert (
                    rest.get(
                        f"/v1/approvals/{again['approval_request_id']}",
                        headers={"Authorization": f"Bearer {OPERATOR_BEARER}"},
                    ).json()["room_binding"]
                    == binding
                )

            async with Client(_transport(url, {WIRE_HEADER: binding}), auth=ROOM_BEARER) as copied:
                copied_result = await copied.call_tool(
                    "capability_call",
                    {
                        "capability_id": "fake.write",
                        "arguments": {"text": "copy"},
                        "reason": "stolen",
                        "confirm": True,
                    },
                )
                copied_payload = _payload(copied_result)
                assert copied_payload["status"] == "error"
                assert copied_payload["error"] == ROOM_BINDING_MISMATCH
                _no_secret(_rendered(copied_result), binding)
                assert manager.calls == []

            async with Client(
                _transport(url, {WIRE_HEADER: malformed}), auth=ROOM_BEARER
            ) as bad_client:
                bad_result = await bad_client.call_tool(
                    "capability_call",
                    {
                        "capability_id": "fake.write",
                        "arguments": {"text": "bad"},
                        "reason": "bad",
                        "confirm": True,
                    },
                )
                bad_payload = _payload(bad_result)
                assert bad_payload["status"] == "error"
                assert bad_payload["error"] == MALFORMED_ROOM_BINDING
                _no_secret(_rendered(bad_result), malformed)
                assert manager.calls == []

        _no_secret([entry.__dict__ for entry in audit.recent(20)], binding, changed, malformed)

    asyncio.run(body())


def test_legacy_identity_ignores_the_header() -> None:
    async def body() -> None:
        manager = RecordingManager()
        store = ApprovalStore()
        audit = InMemoryAuditSink()
        deps = _deps(manager, store, audit)
        tokens = _tokens()
        first = "LEGACY-HEADER-SENTINEL-ONE"
        second = _mint()
        holder: dict[str, _OverlayClient] = {}
        async with _mcp_server(deps, tokens) as (url, _pool):
            async with Client(
                _transport(url, {WIRE_HEADER: first}, _factory(holder)),
                auth=LEGACY_BEARER,
            ) as client:
                allowed = _payload(
                    await client.call_tool(
                        "capability_call",
                        {
                            "capability_id": "fake.write",
                            "arguments": {"text": "legacy"},
                            "reason": "legacy write",
                            "confirm": True,
                        },
                    )
                )
                assert allowed["status"] == "ok"
                assert "approval_request_id" not in allowed
                _no_secret(_rendered(allowed), first)
                holder["client"].overlay[WIRE_HEADER] = second
                again = _payload(
                    await client.call_tool(
                        "capability_call",
                        {
                            "capability_id": "fake.write",
                            "arguments": {"text": "legacy-2"},
                            "reason": "legacy write again",
                            "confirm": True,
                        },
                    )
                )
                assert again["status"] == "ok"
                _no_secret(_rendered(again), first, second)
        assert len(manager.calls) == 2
        assert store._records == {}
        _no_secret([entry.__dict__ for entry in audit.recent(10)], first, second)

    asyncio.run(body())
