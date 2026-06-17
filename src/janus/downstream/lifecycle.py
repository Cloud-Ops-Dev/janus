"""Downstream lifecycle + circuit breaker (Phase 4, infra-aa1).

Always-on servers connect eagerly at startup and stay up. *Lazy* servers
(``lifecycle: lazy``) connect on first use, are reused while warm, and are shut
down once idle past a timeout — the original process/memory saving. A circuit
breaker guards every connect/call: after repeated failures a server trips to
``DEGRADED`` and calls fail fast (no retry storm) until a cooldown elapses, then
one half-open trial decides whether to recover or re-open.

This module is pure and clock-injected so the state machine and breaker are
deterministic under test; the :class:`DownstreamClientManager` owns the actual
sessions and drives these states.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass, field

Clock = Callable[[], float]


class LifecycleState(enum.StrEnum):
    REGISTERED = "registered"  # declared, never connected
    WARMING = "warming"        # connect in progress
    ACTIVE = "active"          # connected and healthy
    COLD = "cold"              # was up, shut down for idleness (reconnect on use)
    DEGRADED = "degraded"      # breaker open — failing fast
    DISABLED = "disabled"      # manually taken out of service


class BreakerState(enum.StrEnum):
    CLOSED = "closed"        # healthy — calls flow
    OPEN = "open"            # tripped — calls fail fast until cooldown
    HALF_OPEN = "half_open"  # cooldown elapsed — one trial call permitted


@dataclass
class CircuitBreaker:
    """Classic three-state breaker. Deterministic via an injected clock.

    ``failure_threshold`` consecutive failures trip it OPEN. After
    ``cooldown_seconds`` it allows a single HALF_OPEN trial: a success closes it,
    a failure re-opens it (resetting the cooldown).
    """

    failure_threshold: int = 3
    cooldown_seconds: float = 30.0
    state: BreakerState = BreakerState.CLOSED
    _failures: int = 0
    _opened_at: float | None = None

    def allow(self, now: float) -> bool:
        """Whether a call may proceed; transitions OPEN -> HALF_OPEN on cooldown."""
        if self.state is BreakerState.OPEN:
            if self._opened_at is not None and now - self._opened_at >= self.cooldown_seconds:
                self.state = BreakerState.HALF_OPEN
                return True
            return False
        return True  # CLOSED or HALF_OPEN (trial in flight)

    def record_success(self) -> None:
        self.state = BreakerState.CLOSED
        self._failures = 0
        self._opened_at = None

    def record_failure(self, now: float) -> None:
        # A failure during a half-open trial re-opens immediately.
        if self.state is BreakerState.HALF_OPEN:
            self.state = BreakerState.OPEN
            self._opened_at = now
            return
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self.state = BreakerState.OPEN
            self._opened_at = now


@dataclass
class ServerLifecycle:
    """Per-server lifecycle record: state + idle clock + circuit breaker."""

    state: LifecycleState = LifecycleState.REGISTERED
    last_used: float | None = None
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    def note_used(self, now: float) -> None:
        self.last_used = now

    def is_idle(self, idle_after: float, now: float) -> bool:
        return self.last_used is not None and (now - self.last_used) >= idle_after
