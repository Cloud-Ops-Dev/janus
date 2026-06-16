"""Policy decision contract shared by the broker and the policy engine.

The broker (infra-22q.3) *enforces* policy; the engine (infra-22q.4) *decides*.
Keeping the contract here lets the broker depend on an interface, not the
concrete engine, so the real deny-by-default profile engine drops in without
touching broker code.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from janus.registry.registry import Capability, EnvScope, RiskTier, Server


class Decision(enum.StrEnum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


@dataclass(frozen=True)
class PolicyContext:
    """Everything a policy decision is a function of."""

    capability: Capability
    server: Server
    env: EnvScope
    profile: str = "default_assistant"
    # Is a human present to answer a confirm prompt? Unattended confirm-tier
    # actions are hard-denied (locked operator decision, 2026-06-16).
    attended: bool = True


@dataclass(frozen=True)
class PolicyDecision:
    decision: Decision
    reason: str
    capability_id: str
    risk: RiskTier


@runtime_checkable
class PolicyEngine(Protocol):
    """Resolve a context to allow/confirm/deny with a human-readable reason."""

    def evaluate(self, ctx: PolicyContext) -> PolicyDecision: ...
