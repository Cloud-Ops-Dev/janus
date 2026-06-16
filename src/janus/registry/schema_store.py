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
from dataclasses import dataclass
from pathlib import Path
from sqlite3 import Connection, Row, connect
from types import TracebackType
from typing import Any, Protocol, runtime_checkable

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


# --------------------------------------------------------------------------- #
# Runtime lifecycle state (Phase 2)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CapabilityState:
    """The mutable runtime lifecycle state of one capability (design §5.1/§5.8).

    The YAML registry owns *structure* (which capabilities exist, their risk /
    summary / env scope) and *initial* approval seeding. This store owns the
    mutable lifecycle: the approved/quarantined flags the broker enforces, the
    reviewed **baseline** descriptor/schema hashes (locked at approval), and the
    **observed** hashes from the most recent discovery crawl. Drift = an approved
    capability whose observed hash diverges from its baseline.
    """

    capability_id: str
    approved: bool
    quarantined: bool
    baseline_description_hash: str | None
    baseline_schema_hash: str | None
    observed_description_hash: str | None
    observed_schema_hash: str | None
    last_verified: str | None
    present: bool
    approved_at: str | None = None
    quarantine_reason: str | None = None

    @property
    def callable(self) -> bool:
        """Brokerable only when approved and not quarantined (mirrors Capability)."""
        return self.approved and not self.quarantined


@runtime_checkable
class CapabilityStateProvider(Protocol):
    """Read-only view of runtime lifecycle state, consumed by the broker.

    Lets the broker honor live approval/quarantine decisions (Phase 2) without
    depending on the SQLite store directly; Phase-1 code with no store wired
    simply passes ``None`` and the broker falls back to the frozen registry.
    """

    def get_state(self, capability_id: str) -> CapabilityState | None: ...


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
    observed_description_hash TEXT,
    observed_schema_hash      TEXT,
    present               INTEGER NOT NULL DEFAULT 1,
    approved_at           TEXT,
    quarantine_reason     TEXT,
    updated_at            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_capabilities_server ON capabilities(server_id);
"""

# Columns added after the initial Phase-1 schema. Applied idempotently on open so
# a pre-existing operational cache (gitignored, rebuildable) gains them without a
# manual drop. ``raw_description_hash`` / ``input_schema_hash`` are the reviewed
# *baseline*; ``observed_*`` are the latest crawl values (see CapabilityState).
_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("observed_description_hash", "TEXT"),
    ("observed_schema_hash", "TEXT"),
    ("present", "INTEGER NOT NULL DEFAULT 1"),
    ("approved_at", "TEXT"),
    ("quarantine_reason", "TEXT"),
)

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

# Re-syncing from YAML must NOT clobber runtime lifecycle state: a quarantine or
# approval established at runtime has to survive a service restart (otherwise a
# poisoned, auto-quarantined tool would silently re-enable on reboot). So the
# ON CONFLICT clause updates *structural* columns only; ``approved``,
# ``quarantined``, the baseline/observed hashes, ``last_verified``, ``present``,
# ``approved_at``, and ``quarantine_reason`` are seeded once on INSERT and then
# owned exclusively by record_observation / set_baseline / approve / quarantine.
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
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Add any post-Phase-1 columns missing from a pre-existing cache."""
        existing = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(capabilities)")
        }
        for column, decl in _MIGRATIONS:
            if column not in existing:
                # column/decl are module-level literals, never user input.
                self._conn.execute(
                    f"ALTER TABLE capabilities ADD COLUMN {column} {decl}"  # noqa: S608
                )

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

    # -- lifecycle state (Phase 2) ----------------------------------------- #
    @staticmethod
    def _state_from_row(row: Row) -> CapabilityState:
        return CapabilityState(
            capability_id=row["id"],
            approved=bool(row["approved"]),
            quarantined=bool(row["quarantined"]),
            baseline_description_hash=row["raw_description_hash"],
            baseline_schema_hash=row["input_schema_hash"],
            observed_description_hash=row["observed_description_hash"],
            observed_schema_hash=row["observed_schema_hash"],
            last_verified=row["last_verified"],
            present=bool(row["present"]),
            approved_at=row["approved_at"],
            quarantine_reason=row["quarantine_reason"],
        )

    def get_state(self, capability_id: str) -> CapabilityState | None:
        """Return the runtime lifecycle state, or ``None`` if not cached.

        Satisfies :class:`CapabilityStateProvider`; the broker treats ``None``
        as "fall back to the frozen registry".
        """
        row = self._conn.execute(
            "SELECT * FROM capabilities WHERE id = ?", (capability_id,)
        ).fetchone()
        return None if row is None else self._state_from_row(row)

    def list_states(self) -> list[CapabilityState]:
        """Every cached capability's lifecycle state, ordered by id."""
        rows = self._conn.execute("SELECT * FROM capabilities ORDER BY id").fetchall()
        return [self._state_from_row(row) for row in rows]

    def record_observation(
        self,
        capability_id: str,
        *,
        observed_description_hash: str | None,
        observed_schema_hash: str | None,
        last_verified: str,
        present: bool = True,
    ) -> None:
        """Persist the latest crawl observation. Never touches approval/baseline."""
        cur = self._conn.execute(
            "UPDATE capabilities SET observed_description_hash = ?, "
            "observed_schema_hash = ?, last_verified = ?, present = ?, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id = ?",
            (
                observed_description_hash,
                observed_schema_hash,
                last_verified,
                int(present),
                capability_id,
            ),
        )
        if cur.rowcount == 0:
            raise KeyError(capability_id)
        self._conn.commit()

    def set_baseline(
        self,
        capability_id: str,
        *,
        raw_description_hash: str | None,
        input_schema_hash: str | None,
    ) -> None:
        """Lock the reviewed/trusted baseline hashes (TOFU on first crawl, or
        on human approval). Subsequent drift is measured against these."""
        cur = self._conn.execute(
            "UPDATE capabilities SET raw_description_hash = ?, input_schema_hash = ?, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id = ?",
            (raw_description_hash, input_schema_hash, capability_id),
        )
        if cur.rowcount == 0:
            raise KeyError(capability_id)
        self._conn.commit()

    def approve_capability(
        self,
        capability_id: str,
        *,
        raw_description_hash: str | None,
        input_schema_hash: str | None,
        approved_at: str,
    ) -> None:
        """Approve a capability and lock the reviewed hashes as its baseline.

        Clears any quarantine: re-approving a drifted capability is exactly how a
        human accepts the new descriptor as the trusted baseline.
        """
        cur = self._conn.execute(
            "UPDATE capabilities SET approved = 1, quarantined = 0, "
            "raw_description_hash = ?, input_schema_hash = ?, approved_at = ?, "
            "quarantine_reason = NULL, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id = ?",
            (raw_description_hash, input_schema_hash, approved_at, capability_id),
        )
        if cur.rowcount == 0:
            raise KeyError(capability_id)
        self._conn.commit()

    def quarantine_capability(self, capability_id: str, *, reason: str) -> None:
        """Mark a capability quarantined (uncallable) until re-approved."""
        cur = self._conn.execute(
            "UPDATE capabilities SET quarantined = 1, quarantine_reason = ?, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id = ?",
            (reason, capability_id),
        )
        if cur.rowcount == 0:
            raise KeyError(capability_id)
        self._conn.commit()

    def quarantine_server(self, server_id: str, *, reason: str) -> list[str]:
        """Quarantine every capability of a server. Returns the affected ids."""
        ids = [
            row["id"]
            for row in self._conn.execute(
                "SELECT id FROM capabilities WHERE server_id = ? ORDER BY id",
                (server_id,),
            ).fetchall()
        ]
        self._conn.execute(
            "UPDATE capabilities SET quarantined = 1, quarantine_reason = ?, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE server_id = ?",
            (reason, server_id),
        )
        self._conn.commit()
        return ids
