"""systemd readiness/liveness notification tests (infra-gn2q).

Covers the three things that decide whether the watchdog backstop actually
works: that it is inert outside systemd, that it reads systemd's arming
variables correctly, and — the one that matters — that the heartbeat really
travels over the NOTIFY_SOCKET datagram protocol rather than merely not
raising.
"""

from __future__ import annotations

import asyncio
import os
import socket
import tempfile
from pathlib import Path

import pytest

from janus.watchdog import run_watchdog_heartbeat, sd_notify, watchdog_interval


@pytest.fixture
def notify_socket(monkeypatch):
    """A real AF_UNIX datagram socket standing in for systemd."""
    tmpdir = tempfile.mkdtemp()
    path = str(Path(tmpdir) / "notify.sock")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind(path)
    sock.settimeout(2)
    monkeypatch.setenv("NOTIFY_SOCKET", path)
    yield sock
    sock.close()


# --- inert outside systemd ------------------------------------------------- #

def test_sd_notify_is_a_noop_without_notify_socket(monkeypatch):
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    assert sd_notify("READY=1") is False


def test_sd_notify_never_raises_on_a_dead_socket(monkeypatch):
    """A notification failure must not be able to take the gateway down."""
    monkeypatch.setenv("NOTIFY_SOCKET", "/nonexistent/janus-test.sock")
    assert sd_notify("READY=1") is False


def test_watchdog_interval_is_none_when_not_armed(monkeypatch):
    monkeypatch.delenv("WATCHDOG_USEC", raising=False)
    assert watchdog_interval() is None


# --- arming ---------------------------------------------------------------- #

def test_watchdog_interval_is_half_the_configured_period(monkeypatch):
    monkeypatch.setenv("WATCHDOG_USEC", "45000000")  # WatchdogSec=45
    monkeypatch.delenv("WATCHDOG_PID", raising=False)
    assert watchdog_interval() == pytest.approx(22.5)


def test_watchdog_interval_ignores_another_processes_arming(monkeypatch):
    """WATCHDOG_PID addressed elsewhere means the variable was inherited, not ours."""
    monkeypatch.setenv("WATCHDOG_USEC", "45000000")
    monkeypatch.setenv("WATCHDOG_PID", str(os.getpid() + 1))
    assert watchdog_interval() is None


def test_watchdog_interval_honours_arming_addressed_to_us(monkeypatch):
    monkeypatch.setenv("WATCHDOG_USEC", "30000000")
    monkeypatch.setenv("WATCHDOG_PID", str(os.getpid()))
    assert watchdog_interval() == pytest.approx(15.0)


@pytest.mark.parametrize("raw", ["", "not-a-number", "0", "-1"])
def test_watchdog_interval_rejects_unusable_values(monkeypatch, raw):
    monkeypatch.setenv("WATCHDOG_USEC", raw)
    monkeypatch.delenv("WATCHDOG_PID", raising=False)
    assert watchdog_interval() is None


# --- the message actually travels ------------------------------------------ #

def test_sd_notify_delivers_the_payload(notify_socket):
    assert sd_notify("READY=1") is True
    assert notify_socket.recv(64) == b"READY=1"


def test_heartbeat_returns_immediately_when_not_armed(monkeypatch):
    """No watchdog armed => the task must exit, not spin forever."""
    monkeypatch.delenv("WATCHDOG_USEC", raising=False)

    async def body() -> None:
        await asyncio.wait_for(run_watchdog_heartbeat(), timeout=1)

    asyncio.run(body())


def test_heartbeat_pings_repeatedly(notify_socket):
    """The whole point: WATCHDOG=1 keeps arriving while the loop is turning."""

    async def body() -> None:
        task = asyncio.create_task(run_watchdog_heartbeat(interval=0.05))
        try:
            loop = asyncio.get_running_loop()
            for _ in range(3):
                data = await loop.run_in_executor(None, notify_socket.recv, 64)
                assert data == b"WATCHDOG=1"
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(body())
