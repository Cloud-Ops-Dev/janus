"""REST front door — the same broker behind FastAPI, for hosts where MCP
tool-surfacing is unreliable (Hermes Desktop) or for SSH/CLI use (design §8).

One policy engine, two front doors: every REST request runs the identical
broker path as the MCP surface. Auth is per-host bearer tokens (locked operator
decision §6) — each token maps to a :class:`HostIdentity` (label + agent profile
+ attended flag), so audit rows are attributable to the calling host. Deny by
default: no/!known token -> 401.

Tokens are supplied at runtime (never committed): the entrypoint builds the map
from the systemd EnvironmentFile / op refs.

NOTE: this module intentionally does NOT use ``from __future__ import
annotations`` — FastAPI resolves endpoint annotations via ``get_type_hints``,
which cannot see the closure-local ``authenticate``/``Identity`` if annotations
are stringized. Runtime ``X | None`` / ``dict[...]`` types are fine on py311+.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from janus.audit.types import AuditSink
from janus.broker import Broker
from janus.discovery.alerts import Alerter
from janus.downstream.client_manager import DownstreamClientManager
from janus.policy.trifecta import TrifectaGuard
from janus.policy.types import PolicyEngine
from janus.registry.registry import EnvScope, Registry
from janus.registry.schema_store import CapabilityStateProvider
from janus.security.output_sanitizer import ResultSanitizer
from janus.server_mcp import _parse_env, _parse_risk


@dataclass(frozen=True)
class HostIdentity:
    label: str
    profile: str = "default_assistant"
    attended: bool = True


@dataclass
class BrokerDeps:
    """Shared, long-lived components both front doors build a Broker from."""

    registry: Registry
    manager: DownstreamClientManager
    policy: PolicyEngine
    audit: AuditSink
    sanitizer: ResultSanitizer | None = None
    state: CapabilityStateProvider | None = None
    # Phase 3: shared across every per-request broker so a session's accumulated
    # trifecta legs persist between its calls; the alerter pings trifecta denials.
    trifecta: TrifectaGuard | None = None
    alerter: Alerter | None = None
    default_env: EnvScope = EnvScope.PROD_SAFE

    def broker_for(self, identity: HostIdentity) -> Broker:
        return Broker(
            self.registry,
            self.manager,
            self.policy,
            self.audit,
            sanitizer=self.sanitizer,
            state=self.state,
            trifecta=self.trifecta,
            alerter=self.alerter,
            session_id=identity.label,
            profile=identity.profile,
            attended=identity.attended,
            default_env=self.default_env,
        )


# -- request bodies --------------------------------------------------------- #
class SearchBody(BaseModel):
    query: str
    env: str | None = None
    max_results: int = 10
    risk_max: str | None = None


class DescribeBody(BaseModel):
    capability_id: str
    env: str | None = None


class CallBody(BaseModel):
    capability_id: str
    reason: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    env: str | None = None
    confirm: bool = False


class ExplainBody(BaseModel):
    capability_id: str
    env: str | None = None


def create_rest_app(
    deps: BrokerDeps,
    tokens: dict[str, HostIdentity],
    *,
    title: str = "Janus",
    lifespan: Callable[[FastAPI], Any] | None = None,
) -> FastAPI:
    app = FastAPI(title=title, version="0.1.0", lifespan=lifespan)

    def authenticate(
        authorization: Annotated[str | None, Header()] = None,
    ) -> HostIdentity:
        prefix = "Bearer "
        if not authorization or not authorization.startswith(prefix):
            raise HTTPException(status_code=401, detail="missing bearer token")
        identity = tokens.get(authorization[len(prefix) :].strip())
        if identity is None:
            raise HTTPException(status_code=401, detail="invalid token")
        return identity

    Identity = Annotated[HostIdentity, Depends(authenticate)]

    @app.get("/v1/health")
    async def health() -> dict[str, Any]:  # unauthenticated liveness
        return {"status": "ok", "servers": len(deps.registry.servers)}

    @app.post("/v1/capability/search")
    async def search(body: SearchBody, identity: Identity) -> dict[str, Any]:
        try:
            return deps.broker_for(identity).capability_search(
                body.query, _parse_env(body.env), body.max_results,
                _parse_risk(body.risk_max),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/capability/describe")
    async def describe(body: DescribeBody, identity: Identity) -> dict[str, Any]:
        try:
            return await deps.broker_for(identity).capability_describe(
                body.capability_id, _parse_env(body.env)
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/capability/call")
    async def call(body: CallBody, identity: Identity) -> dict[str, Any]:
        try:
            return await deps.broker_for(identity).capability_call(
                body.capability_id, body.arguments, body.reason,
                _parse_env(body.env), confirmed=body.confirm,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/policy/explain")
    async def explain(body: ExplainBody, identity: Identity) -> dict[str, Any]:
        try:
            return deps.broker_for(identity).policy_explain(
                body.capability_id, _parse_env(body.env)
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/server/list")
    async def server_list(identity: Identity) -> dict[str, Any]:
        return deps.broker_for(identity).server_list()

    @app.get("/v1/server/health")
    async def server_health(
        identity: Identity, server_id: str | None = None
    ) -> dict[str, Any]:
        return await deps.broker_for(identity).server_health(server_id)

    @app.get("/v1/audit/recent")
    async def audit_recent(identity: Identity, limit: int = 20) -> dict[str, Any]:
        return deps.broker_for(identity).audit_recent(limit)

    return app
