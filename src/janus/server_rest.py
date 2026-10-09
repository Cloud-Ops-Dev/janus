"""REST front door — the same broker behind FastAPI, for hosts where MCP
tool-surfacing is unreliable (Hermes Desktop) or for SSH/CLI use (design §8).

One policy engine, two front doors: every REST request runs the identical
broker path as the MCP surface. Auth is per-host bearer tokens (locked operator
decision §6) — each token maps to a :class:`HostIdentity` (label + agent profile
+ attended flag), so audit rows are attributable to the calling host. Deny by
default: no/!known token -> 401.

Operator approval routes (``/v1/approvals``) are not MCP tools and do not accept
a Janus host or room bearer. They authenticate only with the separate
``JANUS_APPROVAL_TOKEN``.

Tokens are supplied at runtime (never committed): the entrypoint builds the map
from the systemd EnvironmentFile / op refs.

NOTE: this module intentionally does NOT use ``from __future__ import
annotations`` — FastAPI resolves endpoint annotations via ``get_type_hints``,
which cannot see the closure-local ``authenticate``/``Identity`` if annotations
are stringized. Runtime ``X | None`` / ``dict[...]`` types are fine on py311+.
"""

import secrets
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from janus.approval import ApprovalMode, ApprovalStore
from janus.audit.types import AuditSink
from janus.broker import Broker
from janus.discovery.alerts import Alerter
from janus.downstream.client_manager import DownstreamClientManager
from janus.downstream.github_issues import GithubIssueLookup
from janus.policy.trifecta import TrifectaGuard
from janus.policy.types import PolicyEngine
from janus.registry.registry import EnvScope, Registry
from janus.registry.schema_store import CapabilityStateProvider
from janus.search.ranker import BlendedRanker
from janus.security.output_sanitizer import ResultSanitizer
from janus.server_mcp import _parse_env, _parse_risk


@dataclass(frozen=True)
class HostIdentity:
    label: str
    profile: str = "default_assistant"
    attended: bool = True
    # legacy keeps the model confirm bit. room_approval never does.
    approval_mode: ApprovalMode = ApprovalMode.LEGACY


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
    # Phase 5: one prepared blended ranker shared across brokers (cap vectors are
    # precomputed once); None -> keyword search.
    ranker: BlendedRanker | None = None
    default_env: EnvScope = EnvScope.PROD_SAFE
    issue_lookup: GithubIssueLookup | None = None
    # One process-local store shared by every broker this gateway builds.
    approvals: ApprovalStore | None = None

    def broker_for(self, identity: HostIdentity, *, session_id: str | None = None) -> Broker:
        """Broker for one authenticated principal.

        REST omits ``session_id``, so the audit key stays the token label.
        MCP passes a per-session key. Policy always uses ``identity.label``
        and does not parse that key.
        """
        return Broker(
            self.registry,
            self.manager,
            self.policy,
            self.audit,
            sanitizer=self.sanitizer,
            state=self.state,
            trifecta=self.trifecta,
            alerter=self.alerter,
            ranker=self.ranker,
            session_id=identity.label if session_id is None else session_id,
            identity=identity.label,
            profile=identity.profile,
            attended=identity.attended,
            default_env=self.default_env,
            issue_lookup=self.issue_lookup,
            approval_mode=identity.approval_mode,
            approvals=self.approvals,
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
    approval_request_id: str | None = None


class ResultReadBody(BaseModel):
    handle: str
    offset: int
    limit: int = 20_000


class ExplainBody(BaseModel):
    capability_id: str
    env: str | None = None


class ApprovalDecisionBody(BaseModel):
    """Operator decision. Extra fields cannot replace the granted arguments."""

    model_config = ConfigDict(extra="ignore")

    decision: Literal["approve", "deny"]


def _bearer_value(authorization: str | None) -> str | None:
    prefix = "Bearer "
    if not authorization or not authorization.startswith(prefix):
        return None
    return authorization[len(prefix) :].strip()


def _bearer_matches(authorization: str | None, expected: str | None) -> bool:
    presented = _bearer_value(authorization)
    if presented is None or not expected:
        return False
    presented_b = presented.encode()
    expected_b = expected.encode()
    if len(presented_b) != len(expected_b):
        return False
    return secrets.compare_digest(presented_b, expected_b)


def create_rest_app(
    deps: BrokerDeps,
    tokens: dict[str, HostIdentity],
    *,
    title: str = "Janus",
    lifespan: Callable[[FastAPI], Any] | None = None,
    approval_token: str | None = None,
) -> FastAPI:
    app = FastAPI(title=title, version="0.1.0", lifespan=lifespan)
    operator_token = approval_token.strip() if approval_token else None

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
                body.query,
                _parse_env(body.env),
                body.max_results,
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
                body.capability_id,
                body.arguments,
                body.reason,
                _parse_env(body.env),
                confirmed=body.confirm,
                approval_request_id=body.approval_request_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/result/read")
    async def result_read(body: ResultReadBody, identity: Identity) -> dict[str, Any]:
        return deps.broker_for(identity).result_read(body.handle, body.offset, body.limit)

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
    async def server_health(identity: Identity, server_id: str | None = None) -> dict[str, Any]:
        return await deps.broker_for(identity).server_health(server_id)

    @app.get("/v1/audit/recent")
    async def audit_recent(identity: Identity, limit: int = 20) -> dict[str, Any]:
        return deps.broker_for(identity).audit_recent(limit)

    def require_approval_operator(
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        # Host and room bearers are not in this check. A missing operator
        # token fails closed the same way an unknown bearer does.
        if _bearer_value(authorization) is None:
            raise HTTPException(status_code=401, detail="missing bearer token")
        if not _bearer_matches(authorization, operator_token):
            raise HTTPException(status_code=401, detail="invalid token")
        if deps.approvals is None:
            raise HTTPException(status_code=503, detail="approval store unavailable")

    Operator = Annotated[None, Depends(require_approval_operator)]

    def _approval_headers(response: Response) -> None:
        response.headers["Cache-Control"] = "no-store"

    @app.get("/v1/approvals/{approval_id}")
    async def get_approval(approval_id: str, response: Response, _: Operator) -> dict[str, Any]:
        """Operator read of one grant, including exact arguments."""
        _approval_headers(response)
        store = deps.approvals
        if store is None:
            raise HTTPException(status_code=503, detail="approval store unavailable")
        view = store.get(approval_id)
        if view is None:
            raise HTTPException(status_code=404, detail="approval not found")
        return view.as_dict()

    @app.post("/v1/approvals/{approval_id}/decision")
    async def decide_approval(
        approval_id: str, body: ApprovalDecisionBody, response: Response, _: Operator
    ) -> dict[str, Any]:
        """Approve or deny one pending grant. Does not accept new arguments."""
        _approval_headers(response)
        store = deps.approvals
        if store is None:
            raise HTTPException(status_code=503, detail="approval store unavailable")
        try:
            result = store.decide(approval_id, body.decision)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid decision") from exc
        if result.failure is not None and result.failure.value == "unknown":
            raise HTTPException(status_code=404, detail="approval not found")
        if not result.ok or result.view is None:
            status = result.failure.value if result.failure is not None else "unknown"
            raise HTTPException(status_code=409, detail=f"approval {status}")
        return result.view.as_dict()

    return app
