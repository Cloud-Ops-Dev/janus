"""Tests for the approval workflow (infra-bpz.2): AdminService, broker honoring
live store state (pending uncallable -> approve -> callable -> quarantine), and
the janus-admin CLI.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from janus.admin import AdminError, AdminService
from janus.admin.cli import main
from janus.audit import InMemoryAuditSink
from janus.broker import Broker
from janus.downstream import DownstreamClientManager
from janus.policy import Decision, PolicyContext, PolicyDecision
from janus.registry import (
    Capability,
    EnvScope,
    Registry,
    RiskTier,
    SchemaStore,
    Server,
    Transport,
    hash_text,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FAKE = str(Path(__file__).parent / "_fake_downstream.py")


class _AllowPolicy:
    """Allow everything, so tests isolate the store-state gate from policy."""

    def evaluate(self, ctx: PolicyContext) -> PolicyDecision:
        return PolicyDecision(
            Decision.ALLOW, "allow-all", ctx.capability.id, ctx.capability.risk
        )


def _registry(*, approved: bool) -> Registry:
    fake = Server(
        id="fake",
        display_name="Fake",
        transport=Transport.STDIO,
        command=sys.executable,
        args=[FAKE],
        risk_ceiling=RiskTier.READ_ONLY,
        default_env_scope=[EnvScope.DEV, EnvScope.PROD_SAFE],
    )
    cap = Capability(
        id="fake.echo",
        server_id="fake",
        downstream_tool_name="echo",
        title="Echo",
        summary="Echo text back.",
        risk=RiskTier.READ_ONLY,
        env_scope=[EnvScope.DEV, EnvScope.PROD_SAFE],
        approved=approved,
    )
    return Registry(servers={"fake": fake}, capabilities={"fake.echo": cap})


def _store(tmp_path: Path, registry: Registry) -> SchemaStore:
    store = SchemaStore(tmp_path / "data" / "reg.db")
    store.sync_from_registry(registry)
    return store


# --------------------------------------------------------------------------- #
# AdminService unit behavior
# --------------------------------------------------------------------------- #
def test_pending_lists_unapproved(tmp_path: Path) -> None:
    registry = _registry(approved=False)
    with _store(tmp_path, registry) as store:
        svc = AdminService(registry, store)
        assert [s.capability_id for s in svc.pending()] == ["fake.echo"]


def test_approve_locks_observed_as_baseline(tmp_path: Path) -> None:
    registry = _registry(approved=False)
    with _store(tmp_path, registry) as store:
        obs = hash_text("observed description")
        store.record_observation(
            "fake.echo",
            observed_description_hash=obs,
            observed_schema_hash=None,
            last_verified="2026-06-16T00:00:00Z",
        )
        result = AdminService(registry, store).approve("fake.echo")
        assert result.approved is True
        assert result.baseline_description_hash == obs
        state = store.get_state("fake.echo")
        assert state is not None and state.callable is True


def test_quarantine_capability_and_server(tmp_path: Path) -> None:
    registry = _registry(approved=True)
    with _store(tmp_path, registry) as store:
        svc = AdminService(registry, store)
        svc.quarantine_capability("fake.echo", "poisoned")
        state = store.get_state("fake.echo")
        assert state is not None
        assert state.quarantined is True and state.quarantine_reason == "poisoned"
        assert state.callable is False
        # server-wide
        affected = svc.quarantine_server("fake", "kill switch")
        assert affected == ["fake.echo"]


def test_diff_reports_descriptor_drift(tmp_path: Path) -> None:
    registry = _registry(approved=True)
    with _store(tmp_path, registry) as store:
        store.set_baseline(
            "fake.echo",
            raw_description_hash=hash_text("original"),
            input_schema_hash=None,
        )
        store.record_observation(
            "fake.echo",
            observed_description_hash=hash_text("poisoned"),
            observed_schema_hash=None,
            last_verified="2026-06-16T00:00:00Z",
        )
        diff = AdminService(registry, store).diff("fake.echo")
        assert diff.description_changed is True
        assert diff.drifted is True
        assert diff.summary == "Echo text back."  # model-safe summary only


def test_unknown_capability_and_server_raise(tmp_path: Path) -> None:
    registry = _registry(approved=True)
    with _store(tmp_path, registry) as store:
        svc = AdminService(registry, store)
        with pytest.raises(AdminError, match="unknown capability"):
            svc.approve("nope")
        with pytest.raises(AdminError, match="unknown server"):
            svc.quarantine_server("nope", "x")


# --------------------------------------------------------------------------- #
# Acceptance: the broker honors live store state
# --------------------------------------------------------------------------- #
def test_broker_pending_then_approve_then_quarantine(tmp_path: Path) -> None:
    registry = _registry(approved=False)  # pending in YAML -> pending in store

    async def body() -> None:
        with _store(tmp_path, registry) as store:
            svc = AdminService(registry, store)
            mgr = DownstreamClientManager(registry.servers)
            async with mgr:
                await mgr.connect_all()
                broker = Broker(
                    registry,
                    mgr,
                    _AllowPolicy(),
                    InMemoryAuditSink(),
                    state=store,
                    default_env=EnvScope.PROD_SAFE,
                )
                # pending -> uncallable even though policy would allow
                out = await broker.capability_call("fake.echo", {"text": "hi"}, reason="r")
                assert out["status"] == "denied" and "unapproved" in out["reason"]
                assert broker.capability_search("echo")["count"] == 0

                # approve -> callable, executes downstream
                svc.approve("fake.echo")
                out = await broker.capability_call("fake.echo", {"text": "hi"}, reason="r")
                assert out["status"] == "ok" and "hi" in out["text"]
                assert broker.capability_search("echo")["count"] == 1

                # quarantine -> uncallable again, no restart needed
                svc.quarantine_capability("fake.echo", "drift")
                out = await broker.capability_call("fake.echo", {"text": "hi"}, reason="r")
                assert out["status"] == "denied" and "quarantined" in out["reason"]

    asyncio.run(body())


# --------------------------------------------------------------------------- #
# janus-admin CLI (store-only commands against the seed config)
# --------------------------------------------------------------------------- #
def _seed_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JANUS_CONFIG_DIR", str(REPO_ROOT / "config"))
    monkeypatch.setenv("JANUS_DATA_DIR", str(tmp_path / "data"))


def _cli_json(argv: list[str], capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    assert main(argv) == 0
    return json.loads(capsys.readouterr().out)


def test_cli_quarantine_then_approve_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed_env(tmp_path, monkeypatch)
    cid = "open_brain.search_thoughts"

    listed = _cli_json(["list"], capsys)
    assert listed["command"] == "list"
    assert cid in {c["capability_id"] for c in listed["capabilities"]}

    _cli_json(["quarantine-capability", cid, "--reason", "test"], capsys)
    after_q = _cli_json(["list"], capsys)
    row = next(c for c in after_q["capabilities"] if c["capability_id"] == cid)
    assert row["quarantined"] is True and row["callable"] is False

    _cli_json(["approve", cid], capsys)
    after_a = _cli_json(["list"], capsys)
    row = next(c for c in after_a["capabilities"] if c["capability_id"] == cid)
    assert row["approved"] is True and row["quarantined"] is False and row["callable"] is True


def test_cli_quarantine_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed_env(tmp_path, monkeypatch)
    out = _cli_json(["quarantine-server", "beads_readonly", "--reason", "kill"], capsys)
    assert out["server_id"] == "beads_readonly"
    assert all(i.startswith("beads_readonly") for i in out["quarantined"])
    assert out["quarantined"]


def test_cli_unknown_capability_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed_env(tmp_path, monkeypatch)
    assert main(["approve", "does.not.exist"]) == 1
    assert "unknown capability" in capsys.readouterr().err
