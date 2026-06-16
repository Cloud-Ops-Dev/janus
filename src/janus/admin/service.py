"""Approval workflow service (design §5.1, infra-bpz.2).

The human-in-the-loop control plane for capability lifecycle. Discovered-but-
unreviewed capabilities are ``pending`` (``approved=false``) and uncallable;
approving one records the currently-observed descriptor/schema hashes as the
trusted *baseline* and makes it callable. Quarantine (per-capability or whole-
server) marks capabilities uncallable until re-approved.

All operations are pure store mutations over the :class:`SchemaStore`, so they
are synchronous and trivially testable, and they take effect on the live gateway
immediately (the broker reads the same store). Raw downstream descriptor text is
never handled here — only sha256 hashes — preserving the design §11 invariant.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from janus.registry.registry import Registry
from janus.registry.schema_store import CapabilityState, SchemaStore

Clock = Callable[[], datetime]


class AdminError(ValueError):
    """Raised for unknown capability/server ids or invalid lifecycle requests."""


@dataclass(frozen=True)
class ApprovalResult:
    capability_id: str
    approved: bool
    baseline_description_hash: str | None
    baseline_schema_hash: str | None


@dataclass(frozen=True)
class CapabilityDiff:
    """Descriptor delta between an approved baseline and the latest observation.

    Hash-only by construction: the model-safe ``summary`` plus baseline/observed
    hashes and changed flags. The raw descriptor text is fetched live by the
    operator CLI (``diff --fetch``) for human eyes only — never stored or
    returned here.
    """

    capability_id: str
    title: str
    summary: str
    server_id: str
    approved: bool
    quarantined: bool
    present: bool
    baseline_description_hash: str | None
    baseline_schema_hash: str | None
    observed_description_hash: str | None
    observed_schema_hash: str | None
    description_changed: bool
    schema_changed: bool
    quarantine_reason: str | None
    last_verified: str | None

    @property
    def drifted(self) -> bool:
        return self.description_changed or self.schema_changed


class AdminService:
    """Capability approval / quarantine / diff over the registry + store."""

    def __init__(
        self, registry: Registry, store: SchemaStore, *, clock: Clock | None = None
    ) -> None:
        self._registry = registry
        self._store = store
        self._clock: Clock = clock or (lambda: datetime.now(UTC))

    # -- queries ------------------------------------------------------------ #
    def list_states(self) -> list[CapabilityState]:
        return self._store.list_states()

    def pending(self) -> list[CapabilityState]:
        """Capabilities awaiting first approval (uncallable)."""
        return [s for s in self._store.list_states() if not s.approved]

    def diff(self, capability_id: str) -> CapabilityDiff:
        state = self._require_state(capability_id)
        cap = self._registry.capabilities.get(capability_id)
        desc_changed = (
            state.baseline_description_hash is not None
            and state.observed_description_hash != state.baseline_description_hash
        )
        schema_changed = (
            state.baseline_schema_hash is not None
            and state.observed_schema_hash != state.baseline_schema_hash
        )
        return CapabilityDiff(
            capability_id=capability_id,
            title=cap.title if cap else capability_id,
            summary=cap.summary if cap else "",
            server_id=cap.server_id if cap else "",
            approved=state.approved,
            quarantined=state.quarantined,
            present=state.present,
            baseline_description_hash=state.baseline_description_hash,
            baseline_schema_hash=state.baseline_schema_hash,
            observed_description_hash=state.observed_description_hash,
            observed_schema_hash=state.observed_schema_hash,
            description_changed=desc_changed,
            schema_changed=schema_changed,
            quarantine_reason=state.quarantine_reason,
            last_verified=state.last_verified,
        )

    # -- mutations ---------------------------------------------------------- #
    def approve(self, capability_id: str) -> ApprovalResult:
        """Approve a capability, locking its observed descriptor as the baseline.

        Idempotent re-approval is how a human accepts a drifted descriptor: it
        re-locks the baseline to whatever was last observed and clears quarantine.
        """
        state = self._require_state(capability_id)
        self._store.approve_capability(
            capability_id,
            raw_description_hash=state.observed_description_hash,
            input_schema_hash=state.observed_schema_hash,
            approved_at=self._clock().isoformat(),
        )
        new = self._require_state(capability_id)
        return ApprovalResult(
            capability_id=capability_id,
            approved=new.approved,
            baseline_description_hash=new.baseline_description_hash,
            baseline_schema_hash=new.baseline_schema_hash,
        )

    def quarantine_capability(self, capability_id: str, reason: str) -> None:
        self._require_state(capability_id)
        self._store.quarantine_capability(capability_id, reason=reason)

    def quarantine_server(self, server_id: str, reason: str) -> list[str]:
        if server_id not in self._registry.servers:
            raise AdminError(f"unknown server '{server_id}'")
        return self._store.quarantine_server(server_id, reason=reason)

    # -- helpers ------------------------------------------------------------ #
    def _require_state(self, capability_id: str) -> CapabilityState:
        state = self._store.get_state(capability_id)
        if state is None:
            raise AdminError(
                f"unknown capability '{capability_id}' "
                "(not in the registry cache; run 'discover' first)"
            )
        return state
