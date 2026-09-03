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
from mcp.client.session_group import ClientSessionGroup

from janus.downstream import (
    DownstreamClientManager,
    DownstreamError,
    DownstreamNotConnected,
    EnvConnectionResolver,
)
from janus.downstream.client_manager import (
    break_exception_context_cycle,
    exception_context_is_cyclic,
    reraise_unlinked,
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


def test_connect_permanent_failure_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing stdio command fails fast — no retry can make the file exist.

    infra-9ahm: two missing wrapper scripts x 5 retries each pushed the
    gateway stdio cold-start past the MCP client 30s connect timeout, so
    the whole gateway looked dead to c-desktop. Config-class failures must
    not consume the retry budget.
    """

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
            {"bad": bad},
            connect_retries=5,
            connect_retry_delay=0.0,
        )
        real_connect = mgr.connect_server
        calls = {"n": 0}

        async def counting(server_id: str) -> None:
            calls["n"] += 1
            await real_connect(server_id)

        monkeypatch.setattr(mgr, "connect_server", counting)
        async with mgr:
            connected = await mgr.connect_all()
            assert connected == []
            assert "bad" in mgr.connect_failures
            # Fail-fast: exactly one attempt despite connect_retries=5.
            assert calls["n"] == 1

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


# --------------------------------------------------------------------------- #
# infra-gn2q — livelock root causes
# --------------------------------------------------------------------------- #
def _cyclic_retry_chain() -> BaseException:
    """Build the cyclic ``__context__`` chain the 08-03 retry/teardown path produced.

    Repeated connect failures + implicit chaining + AsyncExitStack attaching
    the previous exception onto the new one walked into a loop. contextlib's
    ``_fix_exception_context`` is ``while 1:`` over that chain.
    """
    e0 = RuntimeError("connect-0")
    e1 = RuntimeError("connect-1")
    e2 = RuntimeError("connect-2")
    e1.__context__ = e0
    e2.__context__ = e1
    e0.__context__ = e2  # cycle
    return e2


def test_cyclic_context_is_detected_without_hanging() -> None:
    exc = _cyclic_retry_chain()
    assert exception_context_is_cyclic(exc) is True


def test_old_reraise_preserves_cycle_new_reraise_breaks_it() -> None:
    """Red-first: the unfixed ``raise last_exc`` keeps the cycle; the helper clears it."""
    cyclic = _cyclic_retry_chain()
    try:
        try:
            raise cyclic
        except BaseException:
            raise cyclic  # noqa: B904 — old retry reraise (no from None)
    except BaseException as preserved:
        assert exception_context_is_cyclic(preserved) is True

    fresh = _cyclic_retry_chain()
    try:
        reraise_unlinked(fresh)
    except BaseException as broken:
        assert exception_context_is_cyclic(broken) is False
        assert broken.__context__ is None
        assert broken.__cause__ is None


def test_break_helper_unlinks_cycle() -> None:
    exc = _cyclic_retry_chain()
    break_exception_context_cycle(exc)
    assert exception_context_is_cyclic(exc) is False


def test_connect_retry_reraise_breaks_chained_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry loop must not hand AsyncExitStack a cyclic __context__ chain."""

    async def body() -> None:
        mgr = DownstreamClientManager(
            {"bad": _stdio_server("bad")},
            connect_retries=2,
            connect_retry_delay=0.0,
            connect_timeout=2.0,
        )
        errors = [
            RuntimeError("e0"),
            RuntimeError("e1"),
            RuntimeError("e2"),
        ]
        # Chain them the way repeated SDK failures do.
        errors[1].__context__ = errors[0]
        errors[2].__context__ = errors[1]
        errors[0].__context__ = errors[2]
        calls = {"n": 0}

        async def boom(_server_id: str) -> None:
            i = min(calls["n"], len(errors) - 1)
            calls["n"] += 1
            raise errors[i]

        monkeypatch.setattr(mgr, "connect_server", boom)
        async with mgr:
            connected = await mgr.connect_all()
            assert connected == []
            assert "bad" in mgr.connect_failures
            # The recorded failure came out of reraise_unlinked — no cycle.
            assert exception_context_is_cyclic(errors[2]) is False

    asyncio.run(body())


def test_connect_timeout_does_not_cancel_worker_and_isolates_siblings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller-side timeout must not cancel the worker (wait_for-style cancel
    is livelock 2) and a hung downstream must not block a healthy sibling."""

    cancelled = {"n": 0}
    hang = asyncio.Event()

    original = ClientSessionGroup.connect_to_server

    async def maybe_hang(self: ClientSessionGroup, params: object) -> object:
        command = getattr(params, "command", None)
        if command == "/nonexistent/janus-hang-forever":
            try:
                await hang.wait()
            except asyncio.CancelledError:
                cancelled["n"] += 1
                raise
            raise AssertionError("hang returned")
        return await original(self, params)

    monkeypatch.setattr(ClientSessionGroup, "connect_to_server", maybe_hang)

    async def body() -> None:
        hung = Server(
            id="hung",
            display_name="Hung",
            transport=Transport.STDIO,
            command="/nonexistent/janus-hang-forever",
            args=[],
            default_env_scope=[EnvScope.DEV],
        )
        mgr = DownstreamClientManager(
            {"hung": hung, "good": _stdio_server("good")},
            connect_retries=0,
            connect_retry_delay=0.0,
            # Production default. 2.0s is below FastMCP stdio spawn+handshake
            # on this host (~1.9-3.3s), so the healthy sibling timed out too
            # and the test flaked (diagnosed 2026-09-03, pre-existing at HEAD).
            connect_timeout=10.0,
        )
        async with mgr:
            connected = await mgr.connect_all()
            assert connected == ["good"]
            assert "hung" in mgr.connect_failures
            assert "timed out" in mgr.connect_failures["hung"]
            result = await mgr.call("good", "echo", {"text": "ok"})
            assert "ok" in result.text
            # Isolation was checked while hung was still blocked. Release so
            # close() does not wait out connect_timeout (or leak the task).
            hang.set()
        await asyncio.sleep(0.05)
        assert cancelled["n"] == 0

    # hung consumes the full 10s connect_timeout, then sibling spawn (~3s);
    # 25s is mechanical headroom, not a mask — a livelock still trips it.
    asyncio.run(asyncio.wait_for(body(), timeout=25))
