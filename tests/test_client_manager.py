"""Integration tests for the downstream client manager.

These spin up real ``_fake_downstream.py`` MCP servers over stdio and exercise
the full connect -> list -> call path through ``ClientSessionGroup``. Async
bodies are driven with ``asyncio.run`` so no pytest-asyncio dependency is needed.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from janus.downstream import (
    DownstreamClientManager,
    DownstreamNotConnected,
    EnvConnectionResolver,
)
from janus.registry import AuthType, EnvScope, Server, ServerAuth, Transport

FAKE = str(Path(__file__).parent / "_fake_downstream.py")


def _stdio_server(server_id: str) -> Server:
    return Server(
        id=server_id,
        display_name=f"Fake {server_id}",
        transport=Transport.STDIO,
        command=sys.executable,
        args=[FAKE],
        default_env_scope=[EnvScope.DEV],
    )


# --------------------------------------------------------------------------- #
# Connect / list / call round-trip
# --------------------------------------------------------------------------- #
def test_connect_list_and_call() -> None:
    async def body() -> None:
        mgr = DownstreamClientManager({"fake": _stdio_server("fake")})
        async with mgr:
            connected = await mgr.connect_all()
            assert connected == ["fake"]
            assert mgr.connected_servers == ["fake"]

            tools = {t.name for t in await mgr.list_tools("fake")}
            assert {"echo", "add"} <= tools

            result = await mgr.call("fake", "add", {"a": 2, "b": 3})
            assert result.is_error is False
            structured_ok = bool(result.structured) and 5 in result.structured.values()
            assert "5" in result.text or structured_ok

    asyncio.run(body())


def test_two_servers_no_name_collision() -> None:
    """beads + paperclip both expose ``list_issues``; namespacing must isolate."""

    async def body() -> None:
        mgr = DownstreamClientManager(
            {"a": _stdio_server("a"), "b": _stdio_server("b")}
        )
        async with mgr:
            await mgr.connect_all()
            ra = await mgr.call("a", "echo", {"text": "AAA"})
            rb = await mgr.call("b", "echo", {"text": "BBB"})
            assert "AAA" in ra.text
            assert "BBB" in rb.text

            health = await mgr.health()
            assert health["a"].connected and health["b"].connected
            assert health["a"].tool_count == 2

    asyncio.run(body())


def test_call_unconnected_server_raises() -> None:
    async def body() -> None:
        mgr = DownstreamClientManager({"fake": _stdio_server("fake")})
        async with mgr:
            await mgr.call("fake", "echo", {"text": "x"})

    with pytest.raises(DownstreamNotConnected):
        asyncio.run(body())


# --------------------------------------------------------------------------- #
# Connection resolver (unit)
# --------------------------------------------------------------------------- #
def test_env_resolver_reads_named_vars() -> None:
    server = Server(
        id="ob",
        display_name="OB",
        transport=Transport.STREAMABLE_HTTP,
        endpoint_env="X_URL",
        # secret_env is an env-var NAME, not a secret value.
        auth=ServerAuth(type=AuthType.BEARER, secret_env="X_TOKEN"),  # noqa: S106
        default_env_scope=[EnvScope.DEV],
    )
    resolver = EnvConnectionResolver({"X_URL": "http://h:9/mcp", "X_TOKEN": "s3cr3t"})
    assert resolver.resolve_endpoint(server) == "http://h:9/mcp"
    assert resolver.resolve_secret(server) == "s3cr3t"


def test_env_resolver_defers_op_ref_to_broker() -> None:
    server = Server(
        id="ob",
        display_name="OB",
        transport=Transport.STREAMABLE_HTTP,
        endpoint_env="X_URL",
        # secret_ref is an op:// pointer, not a secret value.
        auth=ServerAuth(type=AuthType.BEARER, secret_ref="op://System/x/credential"),  # noqa: S106
        default_env_scope=[EnvScope.DEV],
    )
    resolver = EnvConnectionResolver({"X_URL": "http://h:9/mcp"})
    # op:// is the credential broker's job (infra-22q.5) — not resolved here.
    assert resolver.resolve_secret(server) is None
