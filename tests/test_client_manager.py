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
    DownstreamError,
    DownstreamNotConnected,
    EnvConnectionResolver,
)
from janus.registry import (
    AuthType,
    EnvScope,
    Lifecycle,
    Server,
    ServerAuth,
    Transport,
)

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
# Tolerant / resilient startup connect (infra-xwx)
# --------------------------------------------------------------------------- #
def test_connect_all_tolerates_failed_downstream() -> None:
    """One dead downstream must not take the gateway down (Logout-Test fix)."""

    async def body() -> None:
        bad = Server(
            id="bad",
            display_name="Bad",
            transport=Transport.STDIO,
            command="/nonexistent/janus-no-such-binary",
            args=[],
            default_env_scope=[EnvScope.DEV],
        )
        mgr = DownstreamClientManager(
            {"good": _stdio_server("good"), "bad": bad},
            connect_retries=1,
            connect_retry_delay=0.0,
        )
        async with mgr:
            connected = await mgr.connect_all()
            assert connected == ["good"]
            assert mgr.connected_servers == ["good"]
            assert "bad" in mgr.connect_failures
            # The healthy server is still fully usable despite 'bad' failing.
            result = await mgr.call("good", "echo", {"text": "hi"})
            assert "hi" in result.text

    asyncio.run(body())


def test_connect_all_retries_transient_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient connect failure (e.g. boot DNS race) is retried, not fatal."""

    async def body() -> None:
        mgr = DownstreamClientManager(
            {"good": _stdio_server("good")},
            connect_retries=3,
            connect_retry_delay=0.0,
        )
        real_connect = mgr.connect_server
        calls = {"n": 0}

        async def flaky(server_id: str) -> None:
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("transient boot race")
            await real_connect(server_id)

        monkeypatch.setattr(mgr, "connect_server", flaky)
        async with mgr:
            connected = await mgr.connect_all()
            assert connected == ["good"]
            assert calls["n"] == 3
            assert mgr.connect_failures == {}

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


def test_env_resolver_resolves_header_secret() -> None:
    resolver = EnvConnectionResolver({"X_BRAIN_KEY": "sb_xyz"})
    assert resolver.resolve_header_secret("X_BRAIN_KEY") == "sb_xyz"
    assert resolver.resolve_header_secret("MISSING") is None


# --------------------------------------------------------------------------- #
# Auth header construction (multi-header, infra-xwx part 1)
# --------------------------------------------------------------------------- #
def _http_server_with_extra_headers() -> Server:
    return Server(
        id="open_brain",
        display_name="Open Brain",
        transport=Transport.STREAMABLE_HTTP,
        endpoint_env="OB_URL",
        auth=ServerAuth(
            type=AuthType.BEARER,
            secret_env="OB_TOKEN",  # noqa: S106 — env-var NAME
            extra_headers={"x-brain-key": "OB_BRAIN_KEY"},
        ),
        default_env_scope=[EnvScope.DEV],
    )


def test_build_auth_headers_bearer_plus_extra() -> None:
    resolver = EnvConnectionResolver({"OB_TOKEN": "tok123", "OB_BRAIN_KEY": "sb_abc"})
    mgr = DownstreamClientManager({}, resolver)
    headers = mgr._build_auth_headers(_http_server_with_extra_headers())
    assert headers == {"Authorization": "Bearer tok123", "x-brain-key": "sb_abc"}


def test_build_auth_headers_extra_only_no_bearer() -> None:
    server = Server(
        id="s",
        display_name="S",
        transport=Transport.STREAMABLE_HTTP,
        endpoint_env="S_URL",
        auth=ServerAuth(extra_headers={"x-api-key": "S_KEY"}),
        default_env_scope=[EnvScope.DEV],
    )
    resolver = EnvConnectionResolver({"S_KEY": "k9"})
    mgr = DownstreamClientManager({}, resolver)
    assert mgr._build_auth_headers(server) == {"x-api-key": "k9"}


def test_build_auth_headers_missing_extra_header_is_fatal() -> None:
    # bearer present but the declared extra-header env var is unset -> §12 loud fail.
    resolver = EnvConnectionResolver({"OB_TOKEN": "tok123"})
    mgr = DownstreamClientManager({}, resolver)
    with pytest.raises(DownstreamError, match="x-brain-key"):
        mgr._build_auth_headers(_http_server_with_extra_headers())


# --------------------------------------------------------------------------- #
# stdio env-injection (infra-b7g)
# --------------------------------------------------------------------------- #
def _stdio_env_server() -> Server:
    return Server(
        id="bd",
        display_name="Beads",
        transport=Transport.STDIO,
        command="bd",
        env={"BEADS_ACTOR": "janus"},
        env_passthrough=["PASSTHRU_VAR"],
        default_env_scope=[EnvScope.DEV],
    )


def test_build_child_env_injects_passthrough_and_static() -> None:
    resolver = EnvConnectionResolver({"PASSTHRU_VAR": "passed-through"})
    mgr = DownstreamClientManager({"bd": _stdio_env_server()}, resolver)
    env = mgr._build_child_env(_stdio_env_server())
    assert env is not None
    assert env["PASSTHRU_VAR"] == "passed-through"  # copied from Janus env
    assert env["BEADS_ACTOR"] == "janus"            # static literal
    assert "PATH" in env                            # SDK default set preserved


def test_build_child_env_none_when_not_declared() -> None:
    server = Server(
        id="x", display_name="X", transport=Transport.STDIO, command="x",
        default_env_scope=[EnvScope.DEV],
    )
    mgr = DownstreamClientManager({"x": server})
    assert mgr._build_child_env(server) is None


def test_build_params_stdio_carries_injected_env() -> None:
    resolver = EnvConnectionResolver({"PASSTHRU_VAR": "v"})
    mgr = DownstreamClientManager({"bd": _stdio_env_server()}, resolver)
    params = mgr._build_params(_stdio_env_server())
    assert params.env is not None
    assert params.env["BEADS_ACTOR"] == "janus"
    assert params.env["PASSTHRU_VAR"] == "v"


# --------------------------------------------------------------------------- #
# Lazy stdio connect-on-demand across tasks (infra-yvs.1.12 regression)
# --------------------------------------------------------------------------- #
def _lazy_stdio_server(server_id: str) -> Server:
    return Server(
        id=server_id,
        display_name=f"Lazy {server_id}",
        transport=Transport.STDIO,
        command=sys.executable,
        args=[FAKE],
        lifecycle=Lifecycle.LAZY,
        default_env_scope=[EnvScope.DEV],
    )


def test_lazy_stdio_connects_on_demand_and_tears_down_across_tasks() -> None:
    """Regression for the lazy-lifecycle hang (infra-yvs.1.12).

    A lazy stdio downstream is connected on demand from a SEPARATE task (the
    per-call task in production) and disconnected from yet another, then the
    manager context exits. Before the owner-task fix, the stdio child's anyio
    cancel scope was entered in the per-call task and exited during teardown in a
    different task -> "Attempted to exit cancel scope in a different task" + hang.
    Now all group ops run in the single owner task, so this completes cleanly.
    """

    async def body() -> None:
        mgr = DownstreamClientManager({"lazy": _lazy_stdio_server("lazy")})
        async with mgr:
            # Lazy server is NOT connected at startup (only_always_on default).
            assert await mgr.connect_all() == []
            assert mgr.connected_servers == []

            # Connect on demand from a DISTINCT task (the cross-task trigger).
            result = await asyncio.create_task(mgr.call("lazy", "echo", {"text": "hi"}))
            assert "hi" in result.text
            assert mgr.connected_servers == ["lazy"]

            # Disconnect from yet another task — must not raise or hang.
            await asyncio.create_task(mgr.disconnect_server("lazy"))
            assert mgr.connected_servers == []
        # Context exit (worker teardown) must complete cleanly.

    asyncio.run(asyncio.wait_for(body(), timeout=30))
