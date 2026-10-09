"""Process-local one-time grants for ``room_approval`` identities.

A room identity cannot authorize a confirm-tier or lethal-trifecta call with
the model ``confirm`` bit. The broker records a pending grant and executes
only after the operator API approves that exact call. State lives in this
process only: a restart drops every grant, and expiry fails closed.

Do not log grant records. The canonical arguments, the caller reason, and a
room-binding header can carry secret values. Those belong in the operator
HTTP response, never in audit lines, logs, or model-facing call results.
"""

from __future__ import annotations

import enum
import json
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from janus.room_binding import RoomBindingRejected, parse_room_binding, room_bindings_equal

APPROVAL_TTL_SECONDS = 15 * 60

_APPROVE = "approve"
_DENY = "deny"


class ApprovalMode(enum.StrEnum):
    """How an authenticated identity satisfies a confirmation gate."""

    LEGACY = "legacy"
    ROOM_APPROVAL = "room_approval"


class ApprovalStatus(enum.StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    CONSUMED = "consumed"
    EXPIRED = "expired"


class ApprovalFailure(enum.StrEnum):
    """Why a presented grant did not authorize execution."""

    UNKNOWN = "unknown"
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    CONSUMED = "consumed"
    MISMATCHED = "mismatched"


def canonical_json(arguments: dict[str, Any]) -> str:
    """Stable JSON for an exact-argument comparison.

    Key order and insignificant whitespace do not matter. The encoded string
    is the value that is compared, so ``1`` and ``1.0`` stay distinct.
    """
    try:
        return json.dumps(
            arguments,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError):
        raise ValueError("arguments are not canonical JSON") from None


def _const_eq(left: str, right: str) -> bool:
    left_b = left.encode()
    right_b = right.encode()
    if len(left_b) != len(right_b):
        secrets.compare_digest(left_b, left_b)
        return False
    return secrets.compare_digest(left_b, right_b)


@dataclass(frozen=True, repr=False)
class ApprovalBinding:
    """Everything a grant is allowed to authorize, and nothing else."""

    identity: str
    profile: str
    session_id: str
    capability_id: str
    server_id: str
    tool_name: str
    env: str
    arguments_canonical: str
    reason: str
    # None is the in-process path. A string is the pinned MCP header value.
    room_binding: str | None = None

    def matches(self, other: ApprovalBinding) -> bool:
        same = (
            self.identity == other.identity
            and self.profile == other.profile
            and self.session_id == other.session_id
            and self.capability_id == other.capability_id
            and self.server_id == other.server_id
            and self.tool_name == other.tool_name
            and self.env == other.env
        )
        # Always compare the secret-bearing fields so a mismatch on an earlier
        # field does not skip them.
        args_ok = _const_eq(self.arguments_canonical, other.arguments_canonical)
        reason_ok = _const_eq(self.reason, other.reason)
        origin_ok = room_bindings_equal(self.room_binding or "", other.room_binding or "") and (
            (self.room_binding is None) == (other.room_binding is None)
        )
        return same and args_ok and reason_ok and origin_ok


@dataclass(frozen=True, repr=False)
class ApprovalView:
    """Operator-facing snapshot. ``arguments`` are the exact call payload.

    ``room_binding`` is the pinned MCP header value, or None when the caller
    presented no header. It is operator-only.
    """

    approval_request_id: str
    status: str
    identity: str
    profile: str
    session_id: str
    room_binding: str | None
    capability_id: str
    server_id: str
    tool_name: str
    env: str
    arguments: dict[str, Any]
    reason: str
    gate_reason: str
    trifecta: bool
    created_at: str
    expires_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "approval_request_id": self.approval_request_id,
            "status": self.status,
            "identity": self.identity,
            "profile": self.profile,
            "session_id": self.session_id,
            "room_binding": self.room_binding,
            "capability_id": self.capability_id,
            "server_id": self.server_id,
            "tool_name": self.tool_name,
            "env": self.env,
            "arguments": self.arguments,
            "reason": self.reason,
            "gate_reason": self.gate_reason,
            "trifecta": self.trifecta,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True)
class ConsumeResult:
    ok: bool
    failure: ApprovalFailure | None = None


@dataclass(frozen=True)
class DecideResult:
    ok: bool
    failure: ApprovalFailure | None = None
    view: ApprovalView | None = None


@dataclass(repr=False)
class _Record:
    id: str
    binding: ApprovalBinding
    status: ApprovalStatus
    gate_reason: str
    trifecta: bool
    created_wall: datetime
    expires_wall: datetime
    created_mono: float
    expires_mono: float


class ApprovalStore:
    """In-memory grants. Not durable: process restart fails closed."""

    def __init__(
        self,
        *,
        ttl_seconds: float = APPROVAL_TTL_SECONDS,
        clock: Callable[[], float] | None = None,
        wall_clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("approval TTL must be positive")
        self._ttl = ttl_seconds
        self._clock: Callable[[], float] = clock or time.monotonic
        self._wall: Callable[[], datetime] = wall_clock or (lambda: datetime.now(UTC))
        self._ids: Callable[[], str] = id_factory or (lambda: secrets.token_urlsafe(32))
        self._records: dict[str, _Record] = {}
        self._lock = threading.Lock()

    def issue(
        self,
        *,
        identity: str,
        profile: str,
        session_id: str,
        capability_id: str,
        server_id: str,
        tool_name: str,
        env: str,
        arguments: dict[str, Any],
        reason: str,
        gate_reason: str,
        trifecta: bool,
        room_binding: str | None = None,
    ) -> str:
        binding = _binding(
            identity=identity,
            profile=profile,
            session_id=session_id,
            capability_id=capability_id,
            server_id=server_id,
            tool_name=tool_name,
            env=env,
            arguments=arguments,
            reason=reason,
            room_binding=room_binding,
        )
        with self._lock:
            now_mono, now_wall = self._now()
            self._prune_locked(now_mono, now_wall)
            new_id = self._fresh_id_locked()
            self._records[new_id] = _Record(
                id=new_id,
                binding=binding,
                status=ApprovalStatus.PENDING,
                gate_reason=gate_reason,
                trifecta=trifecta,
                created_wall=now_wall,
                expires_wall=now_wall + timedelta(seconds=self._ttl),
                created_mono=now_mono,
                expires_mono=now_mono + self._ttl,
            )
            return new_id

    def get(self, approval_id: str) -> ApprovalView | None:
        with self._lock:
            record = self._find_locked(approval_id)
            if record is None:
                return None
            return self._view_locked(record)

    def decide(self, approval_id: str, decision: str) -> DecideResult:
        if decision not in {_APPROVE, _DENY}:
            raise ValueError("decision must be approve or deny")
        with self._lock:
            record = self._find_locked(approval_id)
            if record is None:
                return DecideResult(ok=False, failure=ApprovalFailure.UNKNOWN)
            status = record.status
            if status is ApprovalStatus.PENDING:
                record.status = (
                    ApprovalStatus.APPROVED if decision == _APPROVE else ApprovalStatus.DENIED
                )
                return DecideResult(ok=True, view=self._view_locked(record))
            # A deny still wins until the grant is consumed. An approve never
            # reopens a denied, consumed, or expired grant.
            if status is ApprovalStatus.APPROVED and decision == _DENY:
                record.status = ApprovalStatus.DENIED
                return DecideResult(ok=True, view=self._view_locked(record))
            if status is ApprovalStatus.APPROVED and decision == _APPROVE:
                return DecideResult(ok=True, view=self._view_locked(record))
            if status is ApprovalStatus.DENIED and decision == _DENY:
                return DecideResult(ok=True, view=self._view_locked(record))
            return DecideResult(ok=False, failure=_failure_for(status))

    def consume(
        self,
        approval_id: str,
        *,
        identity: str,
        profile: str,
        session_id: str,
        capability_id: str,
        server_id: str,
        tool_name: str,
        env: str,
        arguments: dict[str, Any],
        reason: str,
        room_binding: str | None = None,
    ) -> ConsumeResult:
        """Mark a matching approved grant consumed. Call this before downstream."""
        try:
            expected = _binding(
                identity=identity,
                profile=profile,
                session_id=session_id,
                capability_id=capability_id,
                server_id=server_id,
                tool_name=tool_name,
                env=env,
                arguments=arguments,
                reason=reason,
                room_binding=room_binding,
            )
        except (ValueError, RoomBindingRejected):
            return ConsumeResult(ok=False, failure=ApprovalFailure.MISMATCHED)
        with self._lock:
            record = self._find_locked(approval_id)
            if record is None:
                return ConsumeResult(ok=False, failure=ApprovalFailure.UNKNOWN)
            status = record.status
            if status is ApprovalStatus.EXPIRED:
                return ConsumeResult(ok=False, failure=ApprovalFailure.EXPIRED)
            if status is not ApprovalStatus.APPROVED:
                return ConsumeResult(ok=False, failure=_failure_for(status))
            if not record.binding.matches(expected):
                return ConsumeResult(ok=False, failure=ApprovalFailure.MISMATCHED)
            record.status = ApprovalStatus.CONSUMED
            return ConsumeResult(ok=True)

    def _now(self) -> tuple[float, datetime]:
        return self._clock(), self._wall()

    def _find_locked(self, approval_id: str) -> _Record | None:
        now_mono, now_wall = self._now()
        self._prune_locked(now_mono, now_wall)
        if not isinstance(approval_id, str):
            return None
        record = self._records.get(approval_id.strip())
        if record is None:
            return None
        self._touch_locked(record, now_mono, now_wall)
        return record

    def _touch_locked(self, record: _Record, now_mono: float, now_wall: datetime) -> None:
        if record.status in {
            ApprovalStatus.CONSUMED,
            ApprovalStatus.DENIED,
            ApprovalStatus.EXPIRED,
        }:
            return
        if _expired(record, now_mono, now_wall):
            record.status = ApprovalStatus.EXPIRED

    def _prune_locked(self, now_mono: float, now_wall: datetime) -> None:
        # Keep a record through its TTL, then one more TTL, so a replay still
        # resolves as consumed/expired instead of disappearing immediately.
        horizon = self._ttl
        stale: list[str] = []
        for key, record in self._records.items():
            wall_done = _wall_reached(now_wall, record.expires_wall + timedelta(seconds=horizon))
            if now_mono >= record.expires_mono + horizon or wall_done:
                stale.append(key)
        for key in stale:
            del self._records[key]

    def _fresh_id_locked(self) -> str:
        for _ in range(5):
            new_id = self._ids()
            if isinstance(new_id, str) and new_id and new_id not in self._records:
                return new_id
        raise RuntimeError("approval id space exhausted")

    def _view_locked(self, record: _Record) -> ApprovalView:
        arguments = json.loads(record.binding.arguments_canonical)
        if not isinstance(arguments, dict):
            arguments = {}
        return ApprovalView(
            approval_request_id=record.id,
            status=record.status.value,
            identity=record.binding.identity,
            profile=record.binding.profile,
            session_id=record.binding.session_id,
            room_binding=record.binding.room_binding,
            capability_id=record.binding.capability_id,
            server_id=record.binding.server_id,
            tool_name=record.binding.tool_name,
            env=record.binding.env,
            arguments=arguments,
            reason=record.binding.reason,
            gate_reason=record.gate_reason,
            trifecta=record.trifecta,
            created_at=record.created_wall.isoformat(),
            expires_at=record.expires_wall.isoformat(),
        )


def _binding(
    *,
    identity: str,
    profile: str,
    session_id: str,
    capability_id: str,
    server_id: str,
    tool_name: str,
    env: str,
    arguments: dict[str, Any],
    reason: str,
    room_binding: str | None,
) -> ApprovalBinding:
    if not isinstance(arguments, dict):
        raise ValueError("arguments are not canonical JSON")
    if not isinstance(reason, str):
        raise ValueError("arguments are not canonical JSON")
    checked_binding = parse_room_binding(room_binding) if room_binding is not None else None
    return ApprovalBinding(
        identity=identity,
        profile=profile,
        session_id=session_id,
        capability_id=capability_id,
        server_id=server_id,
        tool_name=tool_name,
        env=env,
        arguments_canonical=canonical_json(arguments),
        reason=reason,
        room_binding=checked_binding,
    )


def _failure_for(status: ApprovalStatus) -> ApprovalFailure:
    if status is ApprovalStatus.PENDING:
        return ApprovalFailure.PENDING
    if status is ApprovalStatus.APPROVED:
        return ApprovalFailure.APPROVED
    if status is ApprovalStatus.DENIED:
        return ApprovalFailure.DENIED
    if status is ApprovalStatus.CONSUMED:
        return ApprovalFailure.CONSUMED
    return ApprovalFailure.EXPIRED


def _expired(record: _Record, now_mono: float, now_wall: datetime) -> bool:
    # Either clock crossing the TTL fails closed. A forward wall jump cannot
    # extend a grant; a backward jump still loses to the monotonic deadline.
    return now_mono >= record.expires_mono or _wall_reached(now_wall, record.expires_wall)


def _wall_reached(now_wall: datetime, deadline: datetime) -> bool:
    if now_wall.tzinfo is None or deadline.tzinfo is None:
        return True
    return now_wall >= deadline
