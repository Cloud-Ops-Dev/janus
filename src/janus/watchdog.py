"""systemd readiness + liveness notification (infra-gn2q).

Why this exists. On 2026-08-03 the gateway wedged for ~20 minutes while
``systemctl --user is-active janus.service`` reported ``active (running)`` the
entire time. A downstream returned HTTP 503 and the connect/teardown path
livelocked the event loop at 100% CPU, so the process was alive, its port was
either unbound or unresponsive, and *nothing alerted*. Every restart wedged the
same way. See bead ``infra-8r1x`` for the incident and ``infra-gn2q`` for the
hardening.

The lesson is that "the process exists" is not a liveness signal for an asyncio
server. What matters is whether the **event loop is still turning**. So the
gateway now heartbeats from inside the loop: if the loop is starved — by this
livelock or any future one — the heartbeat stops, systemd's ``WatchdogSec``
expires, and the unit is killed and restarted instead of hanging silently.

This is a *backstop*, deliberately. It does not fix any particular livelock; it
bounds how long an unfixed one can go unnoticed.

No third-party dependency: ``sd_notify`` is a one-line datagram protocol, and
taking a dependency for it would be worse than implementing it. When
``NOTIFY_SOCKET`` is unset (foreground runs, tests, ``--stdio``) every function
here is a no-op, so nothing behaves differently outside systemd.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket

logger = logging.getLogger(__name__)

__all__ = ["sd_notify", "watchdog_interval", "run_watchdog_heartbeat"]


def sd_notify(state: str) -> bool:
    """Send a notification to systemd. Returns True if it was actually sent.

    No-op (returns False) when NOTIFY_SOCKET is unset — i.e. whenever the
    process was not started by systemd.
    """
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return False
    # A leading '@' denotes the abstract namespace, encoded as a leading NUL.
    if addr.startswith("@"):
        addr = "\0" + addr[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM | socket.SOCK_CLOEXEC) as sock:
            sock.connect(addr)
            sock.sendall(state.encode("utf-8"))
        return True
    except OSError as exc:  # never let a notification failure take the server down
        logger.warning("sd_notify(%r) failed: %s", state, exc)
        return False


def watchdog_interval() -> float | None:
    """Seconds between heartbeats, or None if systemd did not arm a watchdog.

    systemd sets WATCHDOG_USEC to the configured WatchdogSec. The convention is
    to ping at half that, leaving a full period of margin for a late tick.

    WATCHDOG_PID guards the case where the variable was inherited by a child
    rather than addressed to us.
    """
    raw = os.environ.get("WATCHDOG_USEC")
    if not raw:
        return None
    pid = os.environ.get("WATCHDOG_PID")
    if pid and pid != str(os.getpid()):
        return None
    try:
        usec = int(raw)
    except ValueError:
        logger.warning("WATCHDOG_USEC is not an integer: %r", raw)
        return None
    if usec <= 0:
        return None
    return usec / 2_000_000.0


async def run_watchdog_heartbeat(interval: float | None = None) -> None:
    """Ping systemd's watchdog forever, from inside the event loop.

    Runs as a normal asyncio task ON the loop being monitored — that placement
    is the whole point. A thread would keep pinging happily while the loop was
    wedged and report a dead gateway as healthy, which is precisely the failure
    this exists to catch.

    Returns immediately when no watchdog is armed.
    """
    if interval is None:
        interval = watchdog_interval()
    if not interval:
        return
    logger.info("systemd watchdog armed — heartbeat every %.1fs", interval)
    while True:
        await asyncio.sleep(interval)
        sd_notify("WATCHDOG=1")
