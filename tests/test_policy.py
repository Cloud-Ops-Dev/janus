"""Decision-matrix tests for the deny-by-default ProfilePolicyEngine."""

from __future__ import annotations

from pathlib import Path

import pytest

from janus.policy import (
    Decision,
    PolicyContext,
    PolicyEngine,
    ProfilePolicyEngine,
    load_profiles,
)
from janus.registry import (
    Capability,
    EnvScope,
    RegistryError,
    RiskTier,
    Server,
    Transport,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

_SERVER = Server(
    id="s",
    display_name="S",
    transport=Transport.STREAMABLE_HTTP,
    endpoint_env="S_URL",
    risk_ceiling=RiskTier.DESTRUCTIVE,
    default_env_scope=list(EnvScope),
)


def _cap(
    risk: RiskTier,
    *,
    env_scope: list[EnvScope] | None = None,
    approved: bool = True,
    quarantined: bool = False,
    requires_confirmation: bool = False,
) -> Capability:
    return Capability(
        id="s.cap",
        server_id="s",
        downstream_tool_name="t",
        title="T",
        summary="s",
        risk=risk,
        env_scope=env_scope or list(EnvScope),
        approved=approved,
        quarantined=quarantined,
        requires_confirmation=requires_confirmation,
    )


def _decide(
    risk: RiskTier,
    *,
    profile: str = "default_assistant",
    env: EnvScope = EnvScope.PROD_SAFE,
    **cap_kw: object,
) -> Decision:
    engine = ProfilePolicyEngine()
    ctx = PolicyContext(
        capability=_cap(risk, **cap_kw),  # type: ignore[arg-type]
        server=_SERVER,
        env=env,
        profile=profile,
    )
    return engine.evaluate(ctx).decision


# --------------------------------------------------------------------------- #
# default_assistant
# --------------------------------------------------------------------------- #
def test_default_assistant_matrix() -> None:
    assert _decide(RiskTier.READ_ONLY) is Decision.ALLOW
    assert _decide(RiskTier.LOCAL_WRITE) is Decision.CONFIRM
    assert _decide(RiskTier.EXTERNAL_WRITE) is Decision.DENY
    assert _decide(RiskTier.PROD_WRITE) is Decision.DENY
    assert _decide(RiskTier.CREDENTIAL_ACCESS) is Decision.DENY
    assert _decide(RiskTier.DESTRUCTIVE) is Decision.DENY


def test_default_assistant_env_gate() -> None:
    # default_assistant is not permitted in prod at all.
    assert _decide(RiskTier.READ_ONLY, env=EnvScope.PROD) is Decision.DENY


# --------------------------------------------------------------------------- #
# autonomous_agent — no confirm tier
# --------------------------------------------------------------------------- #
def test_autonomous_agent_matrix() -> None:
    assert _decide(RiskTier.READ_ONLY, profile="autonomous_agent") is Decision.ALLOW
    # local_write is allowed outright (no confirm tier for autonomous)
    assert _decide(RiskTier.LOCAL_WRITE, profile="autonomous_agent") is Decision.ALLOW
    assert _decide(RiskTier.PROD_WRITE, profile="autonomous_agent") is Decision.DENY
    assert (
        _decide(RiskTier.CREDENTIAL_ACCESS, profile="autonomous_agent")
        is Decision.DENY
    )


# --------------------------------------------------------------------------- #
# infra_operator
# --------------------------------------------------------------------------- #
def test_infra_operator_matrix() -> None:
    assert (
        _decide(RiskTier.PROD_READ, profile="infra_operator", env=EnvScope.PROD)
        is Decision.ALLOW
    )
    assert (
        _decide(RiskTier.PROD_WRITE, profile="infra_operator", env=EnvScope.PROD)
        is Decision.CONFIRM
    )
    assert (
        _decide(RiskTier.DESTRUCTIVE, profile="infra_operator", env=EnvScope.PROD)
        is Decision.CONFIRM
    )
    assert (
        _decide(RiskTier.CREDENTIAL_ACCESS, profile="infra_operator")
        is Decision.DENY
    )


# --------------------------------------------------------------------------- #
# Escalations / guards
# --------------------------------------------------------------------------- #
def test_requires_confirmation_escalates_allow_to_confirm() -> None:
    assert _decide(RiskTier.READ_ONLY, requires_confirmation=True) is Decision.CONFIRM


def test_quarantined_and_unapproved_denied() -> None:
    assert _decide(RiskTier.READ_ONLY, quarantined=True) is Decision.DENY
    assert _decide(RiskTier.READ_ONLY, approved=False) is Decision.DENY


def test_capability_env_scope_enforced() -> None:
    # capability only scoped for dev; asking in prod_safe -> deny
    assert (
        _decide(RiskTier.READ_ONLY, env=EnvScope.PROD_SAFE, env_scope=[EnvScope.DEV])
        is Decision.DENY
    )


def test_unknown_profile_denied() -> None:
    assert _decide(RiskTier.READ_ONLY, profile="ghost") is Decision.DENY


# --------------------------------------------------------------------------- #
# Protocol + config loading
# --------------------------------------------------------------------------- #
def test_engine_satisfies_policy_engine_protocol() -> None:
    assert isinstance(ProfilePolicyEngine(), PolicyEngine)


def test_seed_profiles_yaml_loads_and_matches_defaults() -> None:
    profiles = load_profiles(REPO_ROOT / "config" / "profiles.yaml")
    assert set(profiles) == {
        "default_assistant",
        "infra_operator",
        "autonomous_agent",
        "research_agent",
    }
    engine = ProfilePolicyEngine(profiles)
    ctx = PolicyContext(
        capability=_cap(RiskTier.LOCAL_WRITE),
        server=_SERVER,
        env=EnvScope.PROD_SAFE,
        profile="default_assistant",
    )
    assert engine.evaluate(ctx).decision is Decision.CONFIRM


def test_load_profiles_missing_file() -> None:
    with pytest.raises(RegistryError, match="not found"):
        load_profiles(REPO_ROOT / "config" / "nope.yaml")
