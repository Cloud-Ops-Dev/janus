"""Downstream discovery crawler (design §5.1/§5.8, infra-bpz.1).

For each registered downstream server, connect (via the shared client manager),
run ``tools/list``, and for every brokered capability compute the sha256 hashes
of the live raw description + input schema. Those observations are persisted to
the :class:`SchemaStore` (``observed_*`` columns + ``last_verified``) and diffed
against the reviewed *baseline* to classify each capability as
``new`` / ``changed`` / ``unchanged`` / ``missing``.

Security invariant (design §11): raw downstream descriptions are untrusted and
are **never** returned here or persisted as text — only their hash travels. The
report a model could see carries hashes, statuses, and counts, never the raw
descriptor text.

Trust-on-first-observe: a capability already approved in the git-tracked YAML
registry (the human review) adopts its first observed descriptor as the trusted
baseline. Any later divergence is drift, handled by the Phase-2c monitor.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from janus.downstream.client_manager import (
    DownstreamClientManager,
    DownstreamError,
    ToolInfo,
)
from janus.registry.registry import Capability, Registry
from janus.registry.schema_store import SchemaStore, hash_schema, hash_text

Clock = Callable[[], datetime]


class DiscoveryStatus(enum.StrEnum):
    """How a registered capability compares to its trusted baseline this pass."""

    NEW = "new"  # first observation (no prior baseline)
    CHANGED = "changed"  # observed descriptor/schema diverged from baseline (drift)
    UNCHANGED = "unchanged"  # matches baseline
    MISSING = "missing"  # downstream tool not found (gone or server unreachable)


@dataclass(frozen=True)
class CapabilityObservation:
    """The result of comparing one capability to its baseline on one crawl pass."""

    capability_id: str
    server_id: str
    downstream_tool_name: str
    status: DiscoveryStatus
    present: bool
    observed_description_hash: str | None
    observed_schema_hash: str | None
    baseline_description_hash: str | None
    baseline_schema_hash: str | None


@dataclass(frozen=True)
class DiscoveryReport:
    """The outcome of a full discovery crawl across all registered servers."""

    observations: list[CapabilityObservation]
    server_errors: dict[str, str]
    # downstream tools a server exposes that no registry capability claims.
    unregistered: dict[str, list[str]]

    def _by_status(self, status: DiscoveryStatus) -> list[CapabilityObservation]:
        return [o for o in self.observations if o.status is status]

    @property
    def new(self) -> list[CapabilityObservation]:
        return self._by_status(DiscoveryStatus.NEW)

    @property
    def changed(self) -> list[CapabilityObservation]:
        return self._by_status(DiscoveryStatus.CHANGED)

    @property
    def unchanged(self) -> list[CapabilityObservation]:
        return self._by_status(DiscoveryStatus.UNCHANGED)

    @property
    def missing(self) -> list[CapabilityObservation]:
        return self._by_status(DiscoveryStatus.MISSING)

    def counts(self) -> dict[str, int]:
        return {status.value: len(self._by_status(status)) for status in DiscoveryStatus}

    def summary(self) -> dict[str, object]:
        """A model-safe, JSON-serializable digest (hashes/counts only)."""
        return {
            "counts": self.counts(),
            "server_errors": dict(self.server_errors),
            "unregistered": {k: list(v) for k, v in self.unregistered.items()},
        }


class DiscoveryCrawler:
    """Crawl downstreams, persist descriptor observations, classify drift.

    The ``manager`` must already be started and connected (the crawler calls
    ``list_tools`` only). The ``store`` should already mirror the registry
    (``sync_from_registry``) so observations and baselines have a row to land on;
    capabilities absent from the store are still classified (as ``new``) but not
    persisted.
    """

    def __init__(
        self,
        registry: Registry,
        manager: DownstreamClientManager,
        store: SchemaStore,
        *,
        clock: Clock | None = None,
    ) -> None:
        self._registry = registry
        self._manager = manager
        self._store = store
        self._clock: Clock = clock or (lambda: datetime.now(UTC))

    async def crawl(self) -> DiscoveryReport:
        now = self._clock().isoformat()
        observations: list[CapabilityObservation] = []
        server_errors: dict[str, str] = {}
        unregistered: dict[str, list[str]] = {}

        for server_id in self._registry.servers:
            caps = self._registry.capabilities_for_server(server_id)
            tools_by_name: dict[str, ToolInfo] = {}
            try:
                tools_by_name = {t.name: t for t in await self._manager.list_tools(server_id)}
            except DownstreamError as exc:
                server_errors[server_id] = str(exc)

            for cap in caps:
                observations.append(
                    self._observe(cap, tools_by_name.get(cap.downstream_tool_name), now)
                )

            if server_id not in server_errors:
                claimed = {c.downstream_tool_name for c in caps}
                extra = sorted(set(tools_by_name) - claimed)
                if extra:
                    unregistered[server_id] = extra

        return DiscoveryReport(
            observations=observations,
            server_errors=server_errors,
            unregistered=unregistered,
        )

    def _observe(
        self, cap: Capability, tool: ToolInfo | None, now: str
    ) -> CapabilityObservation:
        state = self._store.get_state(cap.id)
        baseline_desc = state.baseline_description_hash if state else None
        baseline_schema = state.baseline_schema_hash if state else None

        if tool is None:
            self._safe_record(cap.id, None, None, now, present=False)
            return CapabilityObservation(
                capability_id=cap.id,
                server_id=cap.server_id,
                downstream_tool_name=cap.downstream_tool_name,
                status=DiscoveryStatus.MISSING,
                present=False,
                observed_description_hash=None,
                observed_schema_hash=None,
                baseline_description_hash=baseline_desc,
                baseline_schema_hash=baseline_schema,
            )

        obs_desc = hash_text(tool.description or "")
        obs_schema = hash_schema(tool.input_schema)
        self._safe_record(cap.id, obs_desc, obs_schema, now, present=True)

        if baseline_desc is None and baseline_schema is None:
            status = DiscoveryStatus.NEW
            approved = state.approved if state is not None else cap.approved
            # trust-on-first-observe: lock the baseline for already-approved caps.
            if approved and state is not None:
                self._store.set_baseline(
                    cap.id, raw_description_hash=obs_desc, input_schema_hash=obs_schema
                )
                baseline_desc, baseline_schema = obs_desc, obs_schema
        elif self._drifted(cap.id, obs_desc, obs_schema):
            status = DiscoveryStatus.CHANGED
        else:
            status = DiscoveryStatus.UNCHANGED

        return CapabilityObservation(
            capability_id=cap.id,
            server_id=cap.server_id,
            downstream_tool_name=cap.downstream_tool_name,
            status=status,
            present=True,
            observed_description_hash=obs_desc,
            observed_schema_hash=obs_schema,
            baseline_description_hash=baseline_desc,
            baseline_schema_hash=baseline_schema,
        )

    def _drifted(self, cap_id: str, obs_desc: str, obs_schema: str) -> bool:
        try:
            return self._store.detect_drift(
                cap_id, raw_description_hash=obs_desc, input_schema_hash=obs_schema
            )
        except KeyError:
            return False  # not in store -> no baseline to drift from

    def _safe_record(
        self,
        cap_id: str,
        obs_desc: str | None,
        obs_schema: str | None,
        now: str,
        *,
        present: bool,
    ) -> None:
        try:
            self._store.record_observation(
                cap_id,
                observed_description_hash=obs_desc,
                observed_schema_hash=obs_schema,
                last_verified=now,
                present=present,
            )
        except KeyError:
            pass  # capability not synced into the store; classify-only.
