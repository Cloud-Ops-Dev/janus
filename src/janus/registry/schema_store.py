"""Descriptor/schema hashing helpers + the SQLite operational cache.

YAML (``registry.py``) is the human-editable source of truth. This module owns
the *operational cache*: a single SQLite file (WAL) under ``data/`` —
deliberately outside any file-sync tree (constitution §15) — that mirrors the
registry and stores per-capability descriptor/schema hashes plus verification
timestamps.

Drift detection (design §5.8): a previously-approved capability whose downstream
raw description or input-schema hash changes is a supply-chain signal. The
discovery crawler (Phase 2) recomputes the live hashes and calls
:meth:`SchemaStore.detect_drift`; a positive result quarantines the capability
until a human re-approves it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from sqlite3 import Connection, Row, connect
from types import TracebackType
from typing import Any

from janus.registry.registry import Capability, Registry, Server

HASH_PREFIX = "sha256"


def hash_text(text: str) -> str:
    """Return ``sha256:<hexdigest>`` for a UTF-8 string."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{HASH_PREFIX}:{digest}"


def hash_schema(schema: Mapping[str, Any]) -> str:
    """Hash a JSON schema canonically (stable regardless of key order)."""
    canonical = json.dumps(
        schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hash_text(canonical)


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS servers (
    id            TEXT PRIMARY KEY,
    display_name  TEXT NOT NULL,
    transport     TEXT NOT NULL,
    trust_level   TEXT NOT NULL,
    lifecycle     TEXT NOT NULL,
    risk_ceiling  TEXT NOT NULL,
    tags          TEXT NOT NULL,
    updated_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS capabilities (
    id                    TEXT PRIMARY KEY,
    server_id             TEXT NOT NULL REFERENCES servers(id),
    downstream_tool_name  TEXT NOT NULL,
    title                 TEXT NOT NULL,
    summary               TEXT NOT NULL,
    risk                  TEXT NOT NULL,
    env_scope             TEXT NOT NULL,
    requires_confirmation INTEGER NOT NULL,
    approved              INTEGER NOT NULL,
    quarantined           INTEGER NOT NULL,
    raw_description_hash  TEXT,
    input_schema_hash     TEXT,
    last_verified         TEXT,
    updated_at            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_capabilities_server ON capabilities(server_id);
"""

_UPSERT_SERVER = """
INSERT INTO servers (id, display_name, transport, trust_level, lifecycle, risk_ceiling, tags)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
    display_name = excluded.display_name,
    transport    = excluded.transport,
    trust_level  = excluded.trust_level,
    lifecycle    = excluded.lifecycle,
    risk_ceiling = excluded.risk_ceiling,
    tags         = excluded.tags,
    updated_at   = strftime('%Y-%m-%dT%H:%M:%fZ','now');
"""

_UPSERT_CAPABILITY = """
INSERT INTO capabilities (
    id, server_id, downstream_tool_name, title, summary, risk, env_scope,
    requires_confirmation, approved, quarantined,
    raw_description_hash, input_schema_hash, last_verified
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
    server_id             = excluded.server_id,
    downstream_tool_name  = excluded.downstream_tool_name,
    title                 = excluded.title,
    summary               = excluded.summary,
    risk                  = excluded.risk,
    env_scope             = excluded.env_scope,
    requires_confirmation = excluded.requires_confirmation,
    approved              = excluded.approved,
    quarantined           = excluded.quarantined,
    raw_description_hash  = excluded.raw_description_hash,
    input_schema_hash     = excluded.input_schema_hash,
    last_verified         = excluded.last_verified,
    updated_at            = strftime('%Y-%m-%dT%H:%M:%fZ','now');
"""


class SchemaStore:
    """SQLite-backed operational cache for the registry.

    Use as a context manager, or call :meth:`close` explicitly. The database
    file and its WAL companions MUST live under ``data/`` (gitignored, no
    file-sync mirror — constitution §15).
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Connection = connect(self.db_path)
        self._conn.row_factory = Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    # -- lifecycle ---------------------------------------------------------- #
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SchemaStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- writes ------------------------------------------------------------- #
    def upsert_server(self, server: Server) -> None:
        self._conn.execute(
            _UPSERT_SERVER,
            (
                server.id,
                server.display_name,
                server.transport.value,
                server.trust_level.value,
                server.lifecycle.value,
                server.risk_ceiling.value,
                json.dumps(server.tags),
            ),
        )
        self._conn.commit()

    def upsert_capability(self, cap: Capability) -> None:
        self._conn.execute(
            _UPSERT_CAPABILITY,
            (
                cap.id,
                cap.server_id,
                cap.downstream_tool_name,
                cap.title,
                cap.summary,
                cap.risk.value,
                json.dumps([scope.value for scope in cap.env_scope]),
                int(cap.requires_confirmation),
                int(cap.approved),
                int(cap.quarantined),
                cap.raw_description_hash,
                cap.input_schema_hash,
                cap.last_verified,
            ),
        )
        self._conn.commit()

    def sync_from_registry(self, registry: Registry) -> int:
        """Upsert every server + capability. Returns the row count written."""
        for server in registry.servers.values():
            self.upsert_server(server)
        for cap in registry.capabilities.values():
            self.upsert_capability(cap)
        return len(registry.servers) + len(registry.capabilities)

    # -- reads -------------------------------------------------------------- #
    def get_capability_hashes(
        self, capability_id: str
    ) -> tuple[str | None, str | None]:
        """Return ``(raw_description_hash, input_schema_hash)`` for a capability."""
        row = self._conn.execute(
            "SELECT raw_description_hash, input_schema_hash "
            "FROM capabilities WHERE id = ?",
            (capability_id,),
        ).fetchone()
        if row is None:
            raise KeyError(capability_id)
        return row["raw_description_hash"], row["input_schema_hash"]

    def count(self, table: str) -> int:
        if table not in {"servers", "capabilities"}:
            raise ValueError(f"unknown table: {table}")
        # table name validated against an allowlist above; values are literals.
        row = self._conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()  # noqa: S608
        return int(row["n"])

    def detect_drift(
        self,
        capability_id: str,
        *,
        raw_description_hash: str | None,
        input_schema_hash: str | None,
    ) -> bool:
        """True if a freshly-observed hash differs from the cached one.

        A ``None`` cached hash means "never observed" — not drift. Only a
        change from a known prior hash counts.
        """
        cached_raw, cached_schema = self.get_capability_hashes(capability_id)
        if cached_raw is not None and raw_description_hash != cached_raw:
            return True
        return cached_schema is not None and input_schema_hash != cached_schema
