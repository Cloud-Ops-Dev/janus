"""Agent profiles — per-profile allow/confirm risk sets + permitted environments.

Deny-by-default: a risk tier not in a profile's ``allow`` or ``confirm`` set is
denied. The three built-in profiles mirror design §5.2. Profiles may be
overridden from ``config/profiles.yaml`` via :func:`load_profiles`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from janus.registry.registry import EnvScope, RegistryError, RiskTier


@dataclass(frozen=True)
class AgentProfile:
    name: str
    allowed_env: frozenset[EnvScope]
    allow: frozenset[RiskTier]
    confirm: frozenset[RiskTier]


def _profile(
    name: str,
    *,
    allowed_env: list[EnvScope],
    allow: list[RiskTier],
    confirm: list[RiskTier],
) -> AgentProfile:
    return AgentProfile(
        name=name,
        allowed_env=frozenset(allowed_env),
        allow=frozenset(allow),
        confirm=frozenset(confirm),
    )


# Built-in profiles (design §5.2). Anything not listed is denied by default.
DEFAULT_PROFILES: dict[str, AgentProfile] = {
    "default_assistant": _profile(
        "default_assistant",
        allowed_env=[EnvScope.DEV, EnvScope.PROD_SAFE],
        allow=[RiskTier.READ_ONLY],
        confirm=[RiskTier.LOCAL_WRITE],
    ),
    "infra_operator": _profile(
        "infra_operator",
        allowed_env=[EnvScope.DEV, EnvScope.TEST, EnvScope.PROD_SAFE, EnvScope.PROD],
        allow=[RiskTier.READ_ONLY, RiskTier.PROD_READ, RiskTier.LOCAL_WRITE],
        confirm=[
            RiskTier.EXTERNAL_WRITE,
            RiskTier.PROD_WRITE,
            RiskTier.DESTRUCTIVE,
            RiskTier.NETWORK_EGRESS,
        ],
    ),
    # Unattended runs (Ralph, overnight). It cannot answer a confirm prompt, so
    # it gets NO confirm tier — anything past local_write is denied. (The broker
    # additionally hard-denies any confirm-tier in an unattended session.)
    "autonomous_agent": _profile(
        "autonomous_agent",
        allowed_env=[EnvScope.DEV, EnvScope.TEST, EnvScope.PROD_SAFE],
        allow=[RiskTier.READ_ONLY, RiskTier.LOCAL_WRITE],
        confirm=[],
    ),
}


def load_profiles(path: Path | str) -> dict[str, AgentProfile]:
    """Load agent profiles from a YAML file (``agent_profiles:`` mapping)."""
    path = Path(path)
    if not path.exists():
        raise RegistryError(f"profiles file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise RegistryError(f"{path}: top-level YAML must be a mapping")
    section = raw.get("agent_profiles") or {}
    if not isinstance(section, dict):
        raise RegistryError(f"{path}: 'agent_profiles' must be a mapping")
    out: dict[str, AgentProfile] = {}
    for raw_name, body in section.items():
        name = str(raw_name)
        if not isinstance(body, dict):
            raise RegistryError(f"profile '{name}': entry must be a mapping")
        try:
            out[name] = _profile(
                name,
                allowed_env=[EnvScope(e) for e in body.get("allowed_env", [])],
                allow=[RiskTier(r) for r in body.get("allow", [])],
                confirm=[RiskTier(r) for r in body.get("confirm", [])],
            )
        except ValueError as exc:
            raise RegistryError(f"invalid profile '{name}': {exc}") from exc
    return out
