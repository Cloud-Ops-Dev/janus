"""Audit entry shape + sink protocol (design §5.7).

Every ``capability.call`` — allow, confirm, or deny — produces one
:class:`AuditEntry`. Arguments are recorded as *key names only* (never values),
so secrets and PII never reach the audit store. The durable SQLite+JSONL sink is
infra-22q.6; :class:`InMemoryAuditSink` here keeps the broker testable now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class AuditEntry:
    timestamp: str
    session_id: str
    profile: str
    capability_id: str
    server_id: str
    env: str
    decision: str
    result_status: str
    reason: str
    # Argument *key names* only — never values (secret/PII isolation).
    arg_keys: list[str] = field(default_factory=list)
    latency_ms: float | None = None


@runtime_checkable
class AuditSink(Protocol):
    def record(self, entry: AuditEntry) -> None: ...

    def recent(
        self, limit: int = 20, *, session_id: str | None = None
    ) -> list[AuditEntry]: ...


class InMemoryAuditSink:
    """Non-durable sink for tests and bootstrapping."""

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    def record(self, entry: AuditEntry) -> None:
        self._entries.append(entry)

    def recent(
        self, limit: int = 20, *, session_id: str | None = None
    ) -> list[AuditEntry]:
        items = self._entries
        if session_id is not None:
            items = [e for e in items if e.session_id == session_id]
        return list(reversed(items[-limit:]))

    def __len__(self) -> int:
        return len(self._entries)
