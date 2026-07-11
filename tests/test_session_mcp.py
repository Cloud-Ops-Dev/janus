"""Authenticated, per-session MCP-over-HTTP acceptance tests (infra-8ja)."""

from __future__ import annotations

import asyncio
import socket
import sys
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastmcp import Client

from janus.audit import InMemoryAuditSink
from janus.downstream import DownstreamClientManager
from janus.policy import ProfilePolicyEngine, TrifectaGuard, TrifectaLeg
from janus.registry import (
    Capability,
    EnvScope,
    Registry,
    RiskTier,
    Server,
    Transport,
    TrustLevel,
)
from janus.server_rest import BrokerDeps, HostIdentity
from janus.session_mcp import JanusTokenVerifier, McpSessionPool, build_session_mcp_server

FAKE = str(Path(__file__).parent / "_fake_downstream.py")
TOKENS = {
    "tok-a": HostIdentity("host-a", profile="default_assistant"),
    "tok-b": HostIdentity("host-b", profile="autonomous_agent"),
}


def _capability() -> Capability:
    return Capability(
        id="fake.add",
        server_id="fake",
        downstream_tool_name="add",
        title="Add integers",
        summary="Add two integers",
        risk=RiskTier.READ_ONLY,
        env_scope=[EnvScope.PROD_SAFE],
        approved=True,
    )


def _deps() -> BrokerDeps:
    server = Server(
        id="fake",
        display_name="Fake",
        transport=Transport.STDIO,
        command=sys.executable,
        args=[FAKE],
        trust_level=TrustLevel.FIRST_PARTY,
        risk_ceiling=RiskTier.READ_ONLY,
        default_env_scope=[EnvScope.PROD_SAFE],
    )
    registry = Registry(servers={"fake": server}, capabilities={"fake.add": _capability()})
    return BrokerDeps(
        registry=registry,
        manager=DownstreamClientManager(registry.servers),
        policy=ProfilePolicyEngine(),
        audit=InMemoryAuditSink(),
        trifecta=TrifectaGuard(),
        default_env=EnvScope.PROD_SAFE,
    )


def test_token_verifier_reuses_rest_identity_map() -> None:
    async def body() -> None:
        verifier = JanusTokenVerifier(TOKENS)
        access = await verifier.verify_token("tok-b")
        assert access is not None
        assert access.client_id == "host-b"
        assert access.claims["janus_profile"] == "autonomous_agent"
        assert await verifier.verify_token("unknown") is None

    asyncio.run(body())


def test_session_pool_binds_identity_and_expires_trifecta_state() -> None:
    now = [0.0]
    deps = _deps()
    pool = McpSessionPool(deps, ttl_seconds=10, clock=lambda: now[0])
    identity = TOKENS["tok-a"]

    first = pool.state_for(identity, "session-one")
    assert pool.state_for(identity, "session-one") is first
    with pytest.raises(PermissionError, match="another identity"):
        pool.state_for(TOKENS["tok-b"], "session-one")

    assert deps.trifecta is not None
    deps.trifecta.record(
        first.key,
        deps.registry.capabilities["fake.add"],
        deps.registry.servers["fake"],
    )
    assert deps.trifecta.session_legs(first.key) == {TrifectaLeg.PRIVATE_DATA}

    now[0] = 10.0
    pool.state_for(identity, "session-two")  # lazily prunes expired sessions
    assert deps.trifecta.session_legs(first.key) == set()
    assert pool.state_for(identity, "session-one") is not first


def test_session_pool_rejects_nonpositive_ttl() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        McpSessionPool(_deps(), ttl_seconds=0)


async def _wait_started(server: uvicorn.Server) -> None:
    for _ in range(200):
        if server.started:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("test MCP HTTP server did not start")


def test_http_clients_have_authenticated_isolated_tool_views() -> None:
    async def body() -> None:
        deps = _deps()
        mcp, _pool = build_session_mcp_server(deps, TOKENS, ttl_seconds=30)
        app = mcp.http_app(path="/", transport="http")
        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        sock.setblocking(False)
        port = int(sock.getsockname()[1])
        http = uvicorn.Server(uvicorn.Config(app, log_level="critical", lifespan="on"))
        task = asyncio.create_task(http.serve(sockets=[sock]))

        async with deps.manager:
            await deps.manager.connect_all()
            await _wait_started(http)
            try:
                url = f"http://127.0.0.1:{port}/"
                initialize = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "auth-test", "version": "1"},
                    },
                }
                headers = {"Accept": "application/json, text/event-stream"}
                async with httpx.AsyncClient() as raw:
                    missing = await raw.post(url, json=initialize, headers=headers)
                    bad = await raw.post(
                        url,
                        json=initialize,
                        headers={**headers, "Authorization": "Bearer wrong"},
                    )
                assert missing.status_code == 401
                assert bad.status_code == 401

                async with Client(url, auth="tok-a") as client_a:
                    base_a = {tool.name for tool in await client_a.list_tools()}
                    assert "cap__fake__add" not in base_a
                    exposed = await client_a.call_tool(
                        "capability_expose", {"capability_ids": ["fake.add"]}
                    )
                    payload = exposed.structured_content or {}
                    assert payload["exposed"] == ["cap__fake__add"]
                    assert "cap__fake__add" in {tool.name for tool in await client_a.list_tools()}

                    async with Client(url, auth="tok-a") as client_b:
                        names_b = {tool.name for tool in await client_b.list_tools()}
                        assert "cap__fake__add" not in names_b
            finally:
                http.should_exit = True
                await task

    asyncio.run(body())
