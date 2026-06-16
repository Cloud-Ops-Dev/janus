"""Deny-by-default policy engine (design §5.2-5.3).

Resolution order for a :class:`PolicyContext`:

1. quarantined / not-approved capability -> DENY
2. capability not scoped for the requested env -> DENY
3. env not permitted for the agent profile -> DENY
4. risk tier in the profile's ``confirm`` set -> CONFIRM
5. risk tier in the profile's ``allow`` set -> ALLOW
   (escalated to CONFIRM if the capability is flagged ``requires_confirmation``)
6. otherwise -> DENY (deny-by-default)

Every outcome carries a human-readable reason surfaced via ``policy.explain``.
The lethal-trifecta session guard is Phase 3 and not implemented here.
"""

from __future__ import annotations

from janus.policy.profiles import DEFAULT_PROFILES, AgentProfile
from janus.policy.types import Decision, PolicyContext, PolicyDecision
from janus.registry.registry import Capability


class ProfilePolicyEngine:
    def __init__(
        self,
        profiles: dict[str, AgentProfile] | None = None,
    ) -> None:
        self._profiles = profiles if profiles is not None else DEFAULT_PROFILES

    def evaluate(self, ctx: PolicyContext) -> PolicyDecision:
        cap = ctx.capability

        if cap.quarantined:
            return self._decide(Decision.DENY, "capability is quarantined", cap)
        if not cap.approved:
            return self._decide(Decision.DENY, "capability is not approved", cap)
        if ctx.env not in cap.env_scope:
            return self._decide(
                Decision.DENY,
                f"capability not scoped for environment '{ctx.env}'",
                cap,
            )

        profile = self._profiles.get(ctx.profile)
        if profile is None:
            return self._decide(
                Decision.DENY, f"unknown agent profile '{ctx.profile}'", cap
            )
        if ctx.env not in profile.allowed_env:
            return self._decide(
                Decision.DENY,
                f"environment '{ctx.env}' not permitted for profile '{profile.name}'",
                cap,
            )

        risk = cap.risk
        if risk in profile.confirm:
            return self._decide(
                Decision.CONFIRM,
                f"risk '{risk}' requires confirmation for profile '{profile.name}'",
                cap,
            )
        if risk in profile.allow:
            if cap.requires_confirmation:
                return self._decide(
                    Decision.CONFIRM,
                    "capability is flagged requires_confirmation",
                    cap,
                )
            return self._decide(
                Decision.ALLOW,
                f"risk '{risk}' allowed for profile '{profile.name}'",
                cap,
            )
        return self._decide(
            Decision.DENY,
            f"risk '{risk}' denied by default for profile '{profile.name}'",
            cap,
        )

    @staticmethod
    def _decide(decision: Decision, reason: str, cap: Capability) -> PolicyDecision:
        return PolicyDecision(
            decision=decision, reason=reason, capability_id=cap.id, risk=cap.risk
        )
