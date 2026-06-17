"""Lethal-trifecta session guard (design §5.5 / Phase 3, infra-lxx).

The "lethal trifecta" (Simon Willison; enforced as a control by Open Edison): a
session becomes capable of data exfiltration once it has combined three powers —

  1. ``PRIVATE_DATA``      — read access to sensitive / private data
  2. ``UNTRUSTED_CONTENT`` — exposure to attacker-controllable (third-party) content
  3. ``EXTERNAL_COMM``     — an outbound channel that can send data off-box

Any single leg is fine. All three together let instructions injected into the
untrusted content drive an exfiltration of the private data out through the
outbound channel. Janus tracks, per session, which legs a session has
*successfully* exercised, and gates the **external-communication** call made
while the other two legs are present — that send is the exfiltration step. The
attended path escalates it to confirmation (the operator sees the full picture);
the unattended path hard-denies it. The guard only ever *escalates* — it never
relaxes a policy ``DENY``.

State is per ``session_id`` and lives for the gateway process lifetime (a session
= one host identity / MCP connection). It is intentionally in-memory: trifecta
risk is about what a single *live* session has accumulated, not a durable fact,
and it must reset when the session ends.
"""

from __future__ import annotations

import enum
import threading
from dataclasses import dataclass

from janus.registry.registry import Capability, RiskTier, Server, TrustLevel


class TrifectaLeg(enum.StrEnum):
    PRIVATE_DATA = "private_data"
    UNTRUSTED_CONTENT = "untrusted_content"
    EXTERNAL_COMM = "external_comm"


_ALL_LEGS = frozenset(TrifectaLeg)

# Read tiers — surfacing data into the session.
_READ_RISKS = frozenset(
    {RiskTier.READ_ONLY, RiskTier.PROD_READ, RiskTier.CREDENTIAL_ACCESS}
)
# Leg 3 — risk tiers that can send data off-box (the exfiltration channel).
_EXTERNAL_COMM_RISKS = frozenset(
    {RiskTier.EXTERNAL_WRITE, RiskTier.NETWORK_EGRESS, RiskTier.HUMAN_MESSAGE_SEND}
)


def legs_for(cap: Capability, server: Server) -> frozenset[TrifectaLeg]:
    """The trifecta legs a successful call to ``cap`` on ``server`` exercises.

    Trust level is the key signal: reading from a *first-party* server surfaces
    YOUR private data (leg 1); any output from a *third-party* server is
    attacker-controllable content (leg 2). The write/egress tiers are the
    outbound channel (leg 3) regardless of trust. A single call can light more
    than one leg (e.g. a third-party ``network_egress`` fetch is both untrusted
    content and an external channel).
    """
    legs: set[TrifectaLeg] = set()
    if server.trust_level is TrustLevel.THIRD_PARTY:
        legs.add(TrifectaLeg.UNTRUSTED_CONTENT)
    elif cap.risk in _READ_RISKS:
        # First-party read = access to private/sensitive data.
        legs.add(TrifectaLeg.PRIVATE_DATA)
    if cap.risk is RiskTier.CREDENTIAL_ACCESS:
        # Credential reads are private data regardless of server trust.
        legs.add(TrifectaLeg.PRIVATE_DATA)
    if cap.risk in _EXTERNAL_COMM_RISKS:
        legs.add(TrifectaLeg.EXTERNAL_COMM)
    return frozenset(legs)


@dataclass(frozen=True)
class TrifectaAssessment:
    """Whether a prospective call must be gated as a lethal-trifecta exfil step."""

    gated: bool
    prospective_legs: frozenset[TrifectaLeg]
    accumulated_legs: frozenset[TrifectaLeg]

    @property
    def reason(self) -> str:
        if not self.gated:
            return ""
        present = ", ".join(sorted(self.accumulated_legs | self.prospective_legs))
        return (
            "lethal trifecta: an external-communication call while the session "
            "already holds private-data + untrusted-content exposure "
            f"({present}) — data exfiltration is possible"
        )


class TrifectaGuard:
    """Per-session leg accumulator + exfil-step gate for the lethal trifecta.

    Thread-safe: the REST app may serve concurrent requests across sessions.
    """

    def __init__(self) -> None:
        self._by_session: dict[str, set[TrifectaLeg]] = {}
        self._lock = threading.Lock()

    def assess(
        self, session_id: str, cap: Capability, server: Server
    ) -> TrifectaAssessment:
        prospective = legs_for(cap, server)
        with self._lock:
            accumulated = frozenset(self._by_session.get(session_id, set()))
        combined = accumulated | prospective
        # Gate the SEND: this call performs external communication, and with it
        # the session holds all three legs. Pure reads (no EXTERNAL_COMM leg) are
        # never gated — the exfiltration only happens on the outbound channel, so
        # the next external-comm call is the one we stop.
        gated = combined == _ALL_LEGS and TrifectaLeg.EXTERNAL_COMM in prospective
        return TrifectaAssessment(
            gated=gated,
            prospective_legs=prospective,
            accumulated_legs=accumulated,
        )

    def record(
        self, session_id: str, cap: Capability, server: Server
    ) -> frozenset[TrifectaLeg]:
        """Fold a SUCCESSFUL call's legs into session state. Returns the new total."""
        legs = legs_for(cap, server)
        with self._lock:
            if not legs:
                return frozenset(self._by_session.get(session_id, set()))
            state = self._by_session.setdefault(session_id, set())
            state |= legs
            return frozenset(state)

    def session_legs(self, session_id: str) -> frozenset[TrifectaLeg]:
        with self._lock:
            return frozenset(self._by_session.get(session_id, set()))
