"""Durable audit log — SQLite (query) + JSONL (tail/forward), design §5.7.

Implements the same :class:`AuditSink` protocol as ``InMemoryAuditSink``, so the
broker swaps to it with no change. Every brokered call — allow, confirm, or deny
— is one row. Arguments are stored as key names only (the broker guarantees this
upstream); secrets and PII never reach the audit store.

The database and JSONL file MUST live under ``data/`` (gitignored, WAL, no
file-sync mirror — constitution §15).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from sqlite3 import Connection, Row, connect
from types import TracebackType

from janus.audit.types import AuditEntry

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS invocations (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      TEXT NOT NULL,
    session_id     TEXT NOT NULL,
    profile        TEXT NOT NULL,
    capability_id  TEXT NOT NULL,
    server_id      TEXT NOT NULL,
    env            TEXT NOT NULL,
    decision       TEXT NOT NULL,
    result_status  TEXT NOT NULL,
    reason         TEXT NOT NULL,
    arg_keys       TEXT NOT NULL,
    latency_ms     REAL
);
CREATE INDEX IF NOT EXISTS idx_invocations_session ON invocations(session_id);
"""

_INSERT = """
INSERT INTO invocations (
    timestamp, session_id, profile, capability_id, server_id, env,
    decision, result_status, reason, arg_keys, latency_ms
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


class SqliteAuditLog:
    """SQLite + JSONL audit sink."""

    def __init__(self, db_path: Path | str, jsonl_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = (
            Path(jsonl_path)
            if jsonl_path is not None
            else self.db_path.with_suffix(".jsonl")
        )
        self._conn: Connection = connect(self.db_path)
        self._conn.row_factory = Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    # -- AuditSink ---------------------------------------------------------- #
    def record(self, entry: AuditEntry) -> None:
        self._conn.execute(
            _INSERT,
            (
                entry.timestamp,
                entry.session_id,
                entry.profile,
                entry.capability_id,
                entry.server_id,
                entry.env,
                entry.decision,
                entry.result_status,
                entry.reason,
                json.dumps(entry.arg_keys),
                entry.latency_ms,
            ),
        )
        self._conn.commit()
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

    def recent(
        self, limit: int = 20, *, session_id: str | None = None
    ) -> list[AuditEntry]:
        if session_id is not None:
            rows = self._conn.execute(
                "SELECT * FROM invocations WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM invocations ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_entry(row) for row in rows]

    @staticmethod
    def _row_to_entry(row: Row) -> AuditEntry:
        return AuditEntry(
            timestamp=row["timestamp"],
            session_id=row["session_id"],
            profile=row["profile"],
            capability_id=row["capability_id"],
            server_id=row["server_id"],
            env=row["env"],
            decision=row["decision"],
            result_status=row["result_status"],
            reason=row["reason"],
            arg_keys=list(json.loads(row["arg_keys"])),
            latency_ms=row["latency_ms"],
        )

    # -- lifecycle ---------------------------------------------------------- #
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SqliteAuditLog:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
