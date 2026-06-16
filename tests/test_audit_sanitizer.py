"""Tests for the durable SQLite+JSONL audit log and the result sanitizer."""

from __future__ import annotations

import json
from pathlib import Path

from janus.audit import AuditEntry, AuditSink, SqliteAuditLog
from janus.downstream import DownstreamResult
from janus.registry import TrustLevel
from janus.security import NullSanitizer, OutputSanitizer, SecretRedactor
from janus.security.output_sanitizer import ResultSanitizer


def _entry(n: int, *, session: str = "s1", decision: str = "allow") -> AuditEntry:
    return AuditEntry(
        timestamp=f"2026-06-16T00:00:{n:02d}Z",
        session_id=session,
        profile="default_assistant",
        capability_id=f"cap.{n}",
        server_id="open_brain",
        env="prod_safe",
        decision=decision,
        result_status="ok",
        reason="r",
        arg_keys=["a", "b"],
        latency_ms=1.5,
    )


# --------------------------------------------------------------------------- #
# Audit log
# --------------------------------------------------------------------------- #
def test_sqlite_audit_satisfies_protocol(tmp_path: Path) -> None:
    with SqliteAuditLog(tmp_path / "data" / "audit.db") as log:
        assert isinstance(log, AuditSink)


def test_audit_record_and_recent_order(tmp_path: Path) -> None:
    with SqliteAuditLog(tmp_path / "data" / "audit.db") as log:
        for i in range(5):
            log.record(_entry(i))
        recent = log.recent(limit=3)
        assert [e.capability_id for e in recent] == ["cap.4", "cap.3", "cap.2"]
        assert recent[0].arg_keys == ["a", "b"]


def test_audit_session_scoping(tmp_path: Path) -> None:
    with SqliteAuditLog(tmp_path / "data" / "audit.db") as log:
        log.record(_entry(1, session="s1"))
        log.record(_entry(2, session="s2"))
        assert len(log.recent(session_id="s1")) == 1
        assert log.recent(session_id="s1")[0].session_id == "s1"


def test_audit_jsonl_mirror(tmp_path: Path) -> None:
    jsonl = tmp_path / "data" / "audit.jsonl"
    with SqliteAuditLog(tmp_path / "data" / "audit.db", jsonl) as log:
        log.record(_entry(1))
        log.record(_entry(2))
    lines = jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["capability_id"] == "cap.1"
    assert first["arg_keys"] == ["a", "b"]  # keys only, never values


def test_audit_persists_across_reopen(tmp_path: Path) -> None:
    db = tmp_path / "data" / "audit.db"
    with SqliteAuditLog(db) as log:
        log.record(_entry(1))
    with SqliteAuditLog(db) as log:
        assert len(log.recent()) == 1


# --------------------------------------------------------------------------- #
# Sanitizer
# --------------------------------------------------------------------------- #
def test_null_sanitizer_passthrough() -> None:
    s = NullSanitizer()
    r = DownstreamResult(is_error=False, text="hello", structured=None)
    assert s.sanitize(r, trust_level=TrustLevel.FIRST_PARTY) is r


def test_output_sanitizer_redacts_text_and_structured() -> None:
    redactor = SecretRedactor()
    sample = "TOPSECRETVALUE1234567"
    redactor.register(sample)
    s = OutputSanitizer(redactor)
    r = DownstreamResult(
        is_error=False,
        text=f"token={sample}",
        structured={"nested": {"key": sample}, "list": [sample]},
    )
    out = s.sanitize(r, trust_level=TrustLevel.FIRST_PARTY)
    assert sample not in out.text
    assert sample not in json.dumps(out.structured)


def test_output_sanitizer_caps_size() -> None:
    s = OutputSanitizer(SecretRedactor(), max_chars=100)
    r = DownstreamResult(is_error=False, text="x" * 500, structured=None)
    out = s.sanitize(r, trust_level=TrustLevel.FIRST_PARTY)
    assert "truncated" in out.text
    assert len(out.text) < 200


def test_output_sanitizer_labels_third_party() -> None:
    s = OutputSanitizer(SecretRedactor())
    r = DownstreamResult(is_error=False, text="some web content", structured=None)
    first = s.sanitize(r, trust_level=TrustLevel.FIRST_PARTY)
    third = s.sanitize(r, trust_level=TrustLevel.THIRD_PARTY)
    assert "UNTRUSTED" not in first.text
    assert "UNTRUSTED EXTERNAL CONTENT" in third.text


def test_output_sanitizer_satisfies_protocol() -> None:
    assert isinstance(OutputSanitizer(SecretRedactor()), ResultSanitizer)
    assert isinstance(NullSanitizer(), ResultSanitizer)
