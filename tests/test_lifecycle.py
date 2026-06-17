"""Lazy downstream lifecycle + circuit breaker tests (Phase 4, infra-aa1).

Pure unit tests over the breaker/state machine (clock-injected, deterministic),
then manager integration over the real ``_fake_downstream.py`` stdio server:
a LAZY server connects on first call, is reaped when idle, reconnects on the
next call, and a server that fails to connect trips the breaker to DEGRADED.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from janus.downstream import (
    BreakerState,
    CircuitBreaker,
    DownstreamClientManager,
    DownstreamError,
    LifecycleState,
    ServerLifecycle,
)
from janus.registry import EnvScope, Lifecycle, Server, Transport

FAKE = str(Path(__file__).parent / "_fake_downstream.py")


# --------------------------------------------------------------------------- #
# CircuitBreaker (pure)
# --------------------------------------------------------------------------- #
def test_breaker_opens_after_threshold() -> None:
    b = CircuitBreaker(failure_threshold=3, cooldown_seconds=30)
    assert b.allow(0.0) is True
    b.record_failure(0.0)
    b.record_failure(0.0)
    assert b.state is BreakerState.CLOSED  # still under threshold
    b.record_failure(0.0)
    assert b.state is BreakerState.OPEN
    assert b.allow(1.0) is False  # within cooldown -> fail fast


def test_breaker_half_opens_after_cooldown_and_closes_on_success() -> None:
    b = CircuitBreaker(failure_threshold=1, cooldown_seconds=30)
    b.record_failure(100.0)
    assert b.state is BreakerState.OPEN
    assert b.allow(120.0) is False           # still cooling down
    assert b.allow(130.0) is True            # cooldown elapsed -> half-open trial
    assert b.state is BreakerState.HALF_OPEN
    b.record_success()
    assert b.state is BreakerState.CLOSED
    assert b.allow(131.0) is True


def test_breaker_half_open_failure_reopens() -> None:
    b = CircuitBreaker(failure_threshold=1, cooldown_seconds=10)
    b.record_failure(0.0)
    assert b.allow(20.0) is True             # half-open
    b.record_failure(20.0)                   # trial failed -> re-open
    assert b.state is BreakerState.OPEN
    assert b.allow(21.0) is False


def test_breaker_success_resets_failures() -> None:
    b = CircuitBreaker(failure_threshold=3)
    b.record_failure(0.0)
    b.record_failure(0.0)
    b.record_success()
    b.record_failure(0.0)
    b.record_failure(0.0)
    assert b.state is BreakerState.CLOSED     # the reset means we are not at 3 yet


def test_server_lifecycle_idle() -> None:
    lc = ServerLifecycle()
    assert lc.is_idle(idle_after=10, now=100) is False  # never used
    lc.note_used(100.0)
    assert lc.is_idle(idle_after=10, now=105) is False
    assert lc.is_idle(idle_after=10, now=111) is True


# --------------------------------------------------------------------------- #
# manager integration — lazy connect / idle reap / reconnect
# --------------------------------------------------------------------------- #
def _lazy_server(sid: str, command: str = sys.executable) -> Server:
    return Server(
        id=sid,
        display_name=sid,
        transport=Transport.STDIO,
        command=command,
        args=[FAKE] if command == sys.executable else [],
        lifecycle=Lifecycle.LAZY,
        default_env_scope=[EnvScope.DEV],
    )


def test_lazy_server_connects_on_first_call_not_at_startup() -> None:
    async def body() -> None:
        mgr = DownstreamClientManager({"lz": _lazy_server("lz")})
        async with mgr:
            connected = await mgr.connect_all()  # only_always_on by default
            assert connected == []                       # lazy not eager-connected
            assert mgr.lifecycle_state("lz") is LifecycleState.REGISTERED
            out = await mgr.call("lz", "add", {"a": 1, "b": 2})
            assert out.is_error is False
            assert "lz" in mgr.connected_servers          # connected on demand
            assert mgr.lifecycle_state("lz") is LifecycleState.ACTIVE

    asyncio.run(body())


def test_idle_lazy_server_is_reaped_then_reconnects() -> None:
    async def body() -> None:
        now = [1000.0]
        mgr = DownstreamClientManager(
            {"lz": _lazy_server("lz")},
            idle_after=60.0,
            clock=lambda: now[0],
        )
        async with mgr:
            await mgr.connect_all()
            await mgr.call("lz", "echo", {"text": "hi"})
            assert "lz" in mgr.connected_servers
            # not yet idle -> not reaped.
            now[0] = 1030.0
            assert await mgr.reap_idle() == []
            # past the idle window -> shut down.
            now[0] = 1100.0
            reaped = await mgr.reap_idle()
            assert reaped == ["lz"]
            assert "lz" not in mgr.connected_servers
            assert mgr.lifecycle_state("lz") is LifecycleState.COLD
            # next call brings it back.
            await mgr.call("lz", "echo", {"text": "again"})
            assert "lz" in mgr.connected_servers
            assert mgr.lifecycle_state("lz") is LifecycleState.ACTIVE

    asyncio.run(body())


def test_always_on_server_is_never_reaped() -> None:
    async def body() -> None:
        now = [0.0]
        always = Server(
            id="ao",
            display_name="ao",
            transport=Transport.STDIO,
            command=sys.executable,
            args=[FAKE],
            lifecycle=Lifecycle.ALWAYS_ON,
            default_env_scope=[EnvScope.DEV],
        )
        mgr = DownstreamClientManager(
            {"ao": always}, idle_after=10.0, clock=lambda: now[0]
        )
        async with mgr:
            await mgr.connect_all()
            assert "ao" in mgr.connected_servers
            now[0] = 10_000.0  # very idle
            assert await mgr.reap_idle() == []          # always-on is kept up
            assert "ao" in mgr.connected_servers

    asyncio.run(body())


def test_failing_lazy_server_trips_breaker_to_degraded() -> None:
    async def body() -> None:
        bad = _lazy_server("bad", command="/nonexistent/janus-no-such-binary")
        mgr = DownstreamClientManager(
            {"bad": bad},
            connect_retries=0,         # fail fast in the test
            breaker_threshold=3,
            breaker_cooldown=30.0,
        )
        async with mgr:
            await mgr.connect_all()
            # three connect failures trip the breaker.
            for _ in range(3):
                with pytest.raises(DownstreamError):
                    await mgr.call("bad", "echo", {"text": "x"})
            assert mgr.lifecycle_state("bad") is LifecycleState.DEGRADED
            # the next call fails fast on the open breaker (no connect attempt).
            with pytest.raises(DownstreamError, match="circuit breaker open"):
                await mgr.call("bad", "echo", {"text": "x"})

    asyncio.run(body())
