"""Drift auto-quarantine monitor (design §5.8, infra-bpz.3).

The supply-chain defense. Each discovery pass, any *approved* capability whose
observed descriptor/schema hash has diverged from its reviewed baseline is
auto-quarantined (made uncallable, live, no restart) and an out-of-band alert is
fired. It stays quarantined until a human re-approves it (which re-locks the
baseline to the new descriptor). Pending capabilities never have a baseline, so
they are never "changed" — only genuinely-trusted capabilities can drift.

Re-runs are idempotent: an already-quarantined capability is skipped, so a
standing drift does not re-alert every pass.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from janus.discovery.alerts import Alerter, NullAlerter
from janus.discovery.crawler import (
    CapabilityObservation,
    DiscoveryCrawler,
    DiscoveryReport,
)
from janus.downstream.client_manager import DownstreamClientManager
from janus.registry.registry import Registry
from janus.registry.schema_store import SchemaStore

Clock = Callable[[], datetime]


@dataclass(frozen=True)
class DriftResult:
    report: DiscoveryReport
    quarantined: list[str]
    alerts_sent: int


class DriftMonitor:
    """Run a discovery crawl, then quarantine + alert on approved-capability drift."""

    def __init__(
        self,
        registry: Registry,
        manager: DownstreamClientManager,
        store: SchemaStore,
        *,
        alerter: Alerter | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._store = store
        self._alerter = alerter or NullAlerter()
        self._crawler = DiscoveryCrawler(registry, manager, store, clock=clock)

    async def scan(self) -> DriftResult:
        report = await self._crawler.crawl()
        quarantined: list[str] = []
        alerts = 0
        for obs in report.changed:
            state = self._store.get_state(obs.capability_id)
            # only approved, not-already-quarantined capabilities drift-quarantine.
            if state is None or not state.approved or state.quarantined:
                continue
            self._store.quarantine_capability(
                obs.capability_id, reason=self._reason(obs)
            )
            quarantined.append(obs.capability_id)
            if self._alerter.send(self._alert_message(obs)):
                alerts += 1
        return DriftResult(report=report, quarantined=quarantined, alerts_sent=alerts)

    @staticmethod
    def _changed_parts(obs: CapabilityObservation) -> str:
        parts: list[str] = []
        if obs.observed_description_hash != obs.baseline_description_hash:
            parts.append("description")
        if obs.observed_schema_hash != obs.baseline_schema_hash:
            parts.append("input schema")
        return " + ".join(parts) or "descriptor"

    def _reason(self, obs: CapabilityObservation) -> str:
        return (
            f"descriptor drift ({self._changed_parts(obs)}) on "
            f"{obs.server_id}/{obs.downstream_tool_name}; auto-quarantined "
            "pending human re-approval"
        )

    def _alert_message(self, obs: CapabilityObservation) -> str:
        # hashes/metadata only — never the raw descriptor text (design §11).
        return (
            f"⚠️ Janus drift: `{obs.capability_id}` "
            f"({obs.server_id}/{obs.downstream_tool_name}) "
            f"{self._changed_parts(obs)} changed from the approved baseline. "
            "Auto-quarantined (uncallable) until re-approved. Review with "
            f"`bin/janus-admin diff {obs.capability_id} --fetch`."
        )
