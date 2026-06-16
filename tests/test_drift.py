"""Tests for drift auto-quarantine + alerting (infra-bpz.3).

The Phase-2 acceptance criterion: changing a downstream tool's description
quarantines it until it is re-approved. Driven against the real
``_fake_downstream.py`` whose ``echo`` description is read from a rewritable file,
so two spawns advertise two different descriptors.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from janus.admin import AdminService
from janus.discovery import DriftMonitor, NullAlerter, WebhookAlerter, build_alerter
from janus.discovery.alerts import WEBHOOK_ENV
from janus.discovery.drift import DriftResult
from janus.downstream import DownstreamClientManager
from janus.registry import (
    Capability,
    CapabilityState,
    EnvScope,
    Registry,
    RiskTier,
    SchemaStore,
    Server,
    Transport,
)

FAKE = str(Path(__file__).parent / "_fake_downstream.py")


class _RecordingAlerter:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def send(self, message: str) -> bool:
        self.messages.append(message)
        return True


def _server(args: list[str]) -> Server:
    return Server(
        id="fake",
        display_name="Fake",
        transport=Transport.STDIO,
        command=sys.executable,
        args=args,
        risk_ceiling=RiskTier.READ_ONLY,
        default_env_scope=[EnvScope.DEV],
    )


def _cap(approved: bool) -> Capability:
    return Capability(
        id="fake.echo",
        server_id="fake",
        downstream_tool_name="echo",
        title="Echo",
        summary="Echo text back.",
        risk=RiskTier.READ_ONLY,
        env_scope=[EnvScope.DEV],
        approved=approved,
    )


def _scan(registry: Registry, db: Path, alerter: _RecordingAlerter) -> DriftResult:
    async def body() -> DriftResult:
        store = SchemaStore(db)
        store.sync_from_registry(registry)  # ON CONFLICT preserves runtime state
        mgr = DownstreamClientManager(registry.servers)
        async with mgr:
            await mgr.connect_all()
            result = await DriftMonitor(registry, mgr, store, alerter=alerter).scan()
        store.close()
        return result

    return asyncio.run(body())


def _state(db: Path, cap_id: str) -> CapabilityState:
    with SchemaStore(db) as store:
        state = store.get_state(cap_id)
        assert state is not None
        return state


def _approve(registry: Registry, db: Path, cap_id: str) -> None:
    with SchemaStore(db) as store:
        store.sync_from_registry(registry)
        AdminService(registry, store).approve(cap_id)


# --------------------------------------------------------------------------- #
# Acceptance: drift -> quarantine -> re-approve clears
# --------------------------------------------------------------------------- #
def test_descriptor_drift_quarantines_until_reapproved(tmp_path: Path) -> None:
    desc = tmp_path / "desc.txt"
    desc.write_text("Original echo description.", encoding="utf-8")
    registry = Registry(
        servers={"fake": _server([FAKE, "--desc-file", str(desc)])},
        capabilities={"fake.echo": _cap(approved=True)},
    )
    db = tmp_path / "data" / "reg.db"
    alerter = _RecordingAlerter()

    # Pass 1: first observation locks the baseline (TOFU); no drift.
    first = _scan(registry, db, alerter)
    assert first.quarantined == []
    assert _state(db, "fake.echo").callable is True

    # Poison the descriptor; next spawn advertises a new description.
    desc.write_text("POISONED echo description.", encoding="utf-8")
    second = _scan(registry, db, alerter)
    assert second.quarantined == ["fake.echo"]
    assert second.alerts_sent == 1
    assert len(alerter.messages) == 1
    assert "fake.echo" in alerter.messages[0]
    poisoned = _state(db, "fake.echo")
    assert poisoned.quarantined is True and poisoned.callable is False

    # Re-approve accepts the new descriptor as the baseline and clears quarantine.
    _approve(registry, db, "fake.echo")
    assert _state(db, "fake.echo").callable is True

    # Pass 3 (still the poisoned text, now the trusted baseline): unchanged.
    third = _scan(registry, db, alerter)
    assert third.quarantined == []
    assert len(alerter.messages) == 1  # no new alert
    assert _state(db, "fake.echo").callable is True


def test_standing_drift_does_not_realert(tmp_path: Path) -> None:
    desc = tmp_path / "desc.txt"
    desc.write_text("Original.", encoding="utf-8")
    registry = Registry(
        servers={"fake": _server([FAKE, "--desc-file", str(desc)])},
        capabilities={"fake.echo": _cap(approved=True)},
    )
    db = tmp_path / "data" / "reg.db"
    alerter = _RecordingAlerter()

    _scan(registry, db, alerter)  # baseline
    desc.write_text("CHANGED.", encoding="utf-8")
    _scan(registry, db, alerter)  # drift -> quarantine + 1 alert
    again = _scan(registry, db, alerter)  # still drifted, already quarantined
    assert again.quarantined == []
    assert len(alerter.messages) == 1  # not re-alerted


def test_pending_capability_never_drift_quarantines(tmp_path: Path) -> None:
    """Only approved (baselined) capabilities can drift; pending ones can't."""
    desc = tmp_path / "desc.txt"
    desc.write_text("Original.", encoding="utf-8")
    registry = Registry(
        servers={"fake": _server([FAKE, "--desc-file", str(desc)])},
        capabilities={"fake.echo": _cap(approved=False)},
    )
    db = tmp_path / "data" / "reg.db"
    alerter = _RecordingAlerter()

    _scan(registry, db, alerter)  # NEW, no baseline (pending)
    desc.write_text("CHANGED.", encoding="utf-8")
    result = _scan(registry, db, alerter)  # still NEW (no baseline) -> not drift
    assert result.quarantined == []
    assert alerter.messages == []


# --------------------------------------------------------------------------- #
# Alerter
# --------------------------------------------------------------------------- #
def test_build_alerter_selects_by_env() -> None:
    assert isinstance(build_alerter({}), NullAlerter)
    assert isinstance(build_alerter({WEBHOOK_ENV: "https://example.invalid/wh"}), WebhookAlerter)


def test_null_alerter_returns_false() -> None:
    assert NullAlerter().send("anything") is False


def test_webhook_alerter_is_best_effort_on_failure() -> None:
    # An unreachable URL must return False, never raise (quarantine must not block).
    assert WebhookAlerter("http://127.0.0.1:1/nope").send("hi") is False
