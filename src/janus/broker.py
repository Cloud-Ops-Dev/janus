"""Broker core — the logic behind Janus's 7 agent-facing tools (design §3).

The broker is the single enforcement point: every ``capability_call`` passes
through policy, is audited, and only then reaches a downstream. The broker
*enforces* decisions; it does not make them (policy engine) nor decide how to
reach a server (client manager) nor where audit rows go (audit sink) — all are
injected, so the Phase-1 stubs and the Phase-3+ real engines are interchangeable.

The agent surface intentionally exposes summaries + schemas only. Raw,
untrusted downstream descriptions never appear in any payload returned here.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from janus.audit.types import AuditEntry, AuditSink
from janus.downstream.client_manager import (
    DownstreamClientManager,
    DownstreamError,
)
from janus.policy.types import Decision, PolicyContext, PolicyEngine
from janus.registry.registry import (
    RISK_SEVERITY,
    Capability,
    EnvScope,
    Registry,
    RiskTier,
)
from janus.security.output_sanitizer import NullSanitizer, ResultSanitizer

Clock = Callable[[], datetime]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


class Broker:
    def __init__(
        self,
        registry: Registry,
        manager: DownstreamClientManager,
        policy: PolicyEngine,
        audit: AuditSink,
        *,
        sanitizer: ResultSanitizer | None = None,
        session_id: str = "default",
        profile: str = "default_assistant",
        attended: bool = True,
        default_env: EnvScope = EnvScope.PROD_SAFE,
        clock: Clock | None = None,
    ) -> None:
        self._registry = registry
        self._manager = manager
        self._policy = policy
        self._audit = audit
        self._sanitizer: ResultSanitizer = sanitizer or NullSanitizer()
        self._session_id = session_id
        self._profile = profile
        self._attended = attended
        self._default_env = default_env
        self._clock: Clock = clock or (lambda: datetime.now(UTC))

    # -- helpers ------------------------------------------------------------ #
    def _context(self, cap: Capability, env: EnvScope) -> PolicyContext:
        return PolicyContext(
            capability=cap,
            server=self._registry.servers[cap.server_id],
            env=env,
            profile=self._profile,
            attended=self._attended,
        )

    def _audit_record(
        self,
        cap: Capability,
        env: EnvScope,
        decision: str,
        result_status: str,
        reason: str,
        arg_keys: list[str],
        latency_ms: float | None = None,
    ) -> None:
        self._audit.record(
            AuditEntry(
                timestamp=self._clock().isoformat(),
                session_id=self._session_id,
                profile=self._profile,
                capability_id=cap.id,
                server_id=cap.server_id,
                env=str(env),
                decision=decision,
                result_status=result_status,
                reason=reason,
                arg_keys=arg_keys,
                latency_ms=latency_ms,
            )
        )

    # -- 1. capability_search ---------------------------------------------- #
    def capability_search(
        self,
        query: str,
        env: EnvScope | None = None,
        max_results: int = 10,
        risk_max: RiskTier | None = None,
    ) -> dict[str, Any]:
        """Ranked short list of policy-allowed capabilities (NO full schemas)."""
        env = env or self._default_env
        terms = _tokenize(query)
        scored: list[tuple[float, Capability]] = []
        for cap in self._registry.callable_capabilities():
            if env not in cap.env_scope:
                continue
            if risk_max is not None and RISK_SEVERITY[cap.risk] > RISK_SEVERITY[risk_max]:
                continue
            decision = self._policy.evaluate(self._context(cap, env))
            # Denied tools never surface as normal results (design §7).
            if decision.decision is Decision.DENY:
                continue
            score = self._score(terms, cap)
            if terms and score <= 0:
                continue
            scored.append((score, cap))
        scored.sort(key=lambda pair: (pair[0], pair[1].id), reverse=True)
        results = [self._search_row(cap, env) for _score, cap in scored[:max_results]]
        return {"query": query, "env": str(env), "count": len(results), "results": results}

    @staticmethod
    def _score(terms: set[str], cap: Capability) -> float:
        if not terms:
            return 1.0
        haystack = _tokenize(
            f"{cap.title} {cap.summary} {' '.join(cap.tags)} {cap.downstream_tool_name}"
        )
        overlap = terms & haystack
        return len(overlap) / len(terms)

    def _search_row(self, cap: Capability, env: EnvScope) -> dict[str, Any]:
        decision = self._policy.evaluate(self._context(cap, env))
        return {
            "capability_id": cap.id,
            "title": cap.title,
            "summary": cap.summary,
            "risk": str(cap.risk),
            "server_id": cap.server_id,
            "tags": list(cap.tags),
            "requires_confirmation": cap.requires_confirmation,
            "decision": str(decision.decision),
        }

    # -- 2. capability_describe -------------------------------------------- #
    async def capability_describe(
        self, capability_id: str, env: EnvScope | None = None
    ) -> dict[str, Any]:
        """One capability's schema (fetched just-in-time) + risk + policy."""
        cap = self._registry.capabilities.get(capability_id)
        if cap is None:
            return {"error": f"unknown capability '{capability_id}'"}
        env = env or self._default_env
        server = self._registry.servers[cap.server_id]
        decision = self._policy.evaluate(self._context(cap, env))

        input_schema: dict[str, Any] | None = None
        schema_error: str | None = None
        try:
            tools = await self._manager.list_tools(cap.server_id)
            match = next(
                (t for t in tools if t.name == cap.downstream_tool_name), None
            )
            input_schema = match.input_schema if match is not None else None
            if match is None:
                schema_error = "downstream tool not found"
        except DownstreamError as exc:
            schema_error = str(exc)

        return {
            "capability_id": cap.id,
            "title": cap.title,
            # summary only — the raw downstream description is never exposed.
            "summary": cap.summary,
            "risk": str(cap.risk),
            "env_scope": [str(e) for e in cap.env_scope],
            "requires_confirmation": cap.requires_confirmation,
            "server": {
                "id": server.id,
                "display_name": server.display_name,
                "trust_level": str(server.trust_level),
            },
            "input_schema": input_schema,
            "schema_error": schema_error,
            "policy": {"decision": str(decision.decision), "reason": decision.reason},
        }

    # -- 3. capability_call ------------------------------------------------ #
    async def capability_call(
        self,
        capability_id: str,
        arguments: dict[str, Any] | None,
        reason: str,
        env: EnvScope | None = None,
    ) -> dict[str, Any]:
        """Policy-checked, audited invocation of one capability."""
        arguments = arguments or {}
        arg_keys = sorted(arguments)
        cap = self._registry.capabilities.get(capability_id)
        if cap is None:
            return {"status": "error", "error": f"unknown capability '{capability_id}'"}
        env = env or self._default_env

        if not cap.callable:
            state = "quarantined" if cap.quarantined else "unapproved"
            self._audit_record(cap, env, "deny", "blocked", state, arg_keys)
            return {
                "status": "denied",
                "capability_id": cap.id,
                "reason": f"capability is {state}",
            }

        decision = self._policy.evaluate(self._context(cap, env))

        if decision.decision is Decision.DENY:
            self._audit_record(cap, env, "deny", "blocked", decision.reason, arg_keys)
            return {
                "status": "denied",
                "capability_id": cap.id,
                "reason": decision.reason,
            }

        if decision.decision is Decision.CONFIRM:
            if not self._attended:
                # Locked operator decision (2026-06-16): unattended confirm-tier
                # is hard-denied + audited (no async approval queue in Phase 1).
                hard = f"unattended session: confirm-tier hard-denied ({decision.reason})"
                self._audit_record(cap, env, "deny", "blocked", hard, arg_keys)
                return {"status": "denied", "capability_id": cap.id, "reason": hard}
            # MCP cannot prompt interactively; surface a confirmation requirement
            # for the REST/CLI path (infra-22q.7) to satisfy.
            self._audit_record(
                cap, env, "confirm", "needs_confirmation", decision.reason, arg_keys
            )
            return {
                "status": "needs_confirmation",
                "capability_id": cap.id,
                "reason": decision.reason,
                "preview": {
                    "server_id": cap.server_id,
                    "downstream_tool": cap.downstream_tool_name,
                    "risk": str(cap.risk),
                    "env": str(env),
                    "arg_keys": arg_keys,
                },
            }

        # ALLOW — execute downstream.
        started = self._clock()
        try:
            result = await self._manager.call(
                cap.server_id, cap.downstream_tool_name, arguments
            )
        except DownstreamError as exc:
            latency = self._elapsed_ms(started)
            self._audit_record(
                cap, env, "allow", "error", str(exc), arg_keys, latency
            )
            return {"status": "error", "capability_id": cap.id, "error": str(exc)}

        server = self._registry.servers[cap.server_id]
        result = self._sanitizer.sanitize(result, trust_level=server.trust_level)
        latency = self._elapsed_ms(started)
        status = "error" if result.is_error else "ok"
        self._audit_record(cap, env, "allow", status, reason, arg_keys, latency)
        return {
            "status": status,
            "capability_id": cap.id,
            "is_error": result.is_error,
            "text": result.text,
            "structured": result.structured,
        }

    def _elapsed_ms(self, started: datetime) -> float:
        return (self._clock() - started).total_seconds() * 1000.0

    # -- 4. server_list ---------------------------------------------------- #
    def server_list(self) -> dict[str, Any]:
        connected = set(self._manager.connected_servers)
        servers = []
        for sid, server in self._registry.servers.items():
            caps = self._registry.capabilities_for_server(sid)
            servers.append(
                {
                    "server_id": sid,
                    "display_name": server.display_name,
                    "transport": str(server.transport),
                    "trust_level": str(server.trust_level),
                    "lifecycle": str(server.lifecycle),
                    "risk_ceiling": str(server.risk_ceiling),
                    "connected": sid in connected,
                    "capability_count": len(caps),
                    "approved_count": sum(1 for c in caps if c.callable),
                    "tags": list(server.tags),
                }
            )
        return {"count": len(servers), "servers": servers}

    # -- 5. server_health -------------------------------------------------- #
    async def server_health(self, server_id: str | None = None) -> dict[str, Any]:
        health = await self._manager.health(server_id)
        rows = []
        for sid, status in health.items():
            rows.append(
                {
                    "server_id": sid,
                    "connected": status.connected,
                    "tool_count": status.tool_count,
                    "capability_count": len(
                        self._registry.capabilities_for_server(sid)
                    ),
                    "error": status.error,
                }
            )
        return {"count": len(rows), "servers": rows}

    # -- 6. policy_explain ------------------------------------------------- #
    def policy_explain(
        self, capability_id: str, env: EnvScope | None = None
    ) -> dict[str, Any]:
        cap = self._registry.capabilities.get(capability_id)
        if cap is None:
            return {"error": f"unknown capability '{capability_id}'"}
        env = env or self._default_env
        decision = self._policy.evaluate(self._context(cap, env))
        return {
            "capability_id": cap.id,
            "env": str(env),
            "profile": self._profile,
            "attended": self._attended,
            "risk": str(cap.risk),
            "requires_confirmation": cap.requires_confirmation,
            "decision": str(decision.decision),
            "reason": decision.reason,
        }

    # -- 7. audit_recent --------------------------------------------------- #
    def audit_recent(self, limit: int = 20) -> dict[str, Any]:
        entries = self._audit.recent(limit, session_id=self._session_id)
        return {
            "count": len(entries),
            "entries": [
                {
                    "timestamp": e.timestamp,
                    "capability_id": e.capability_id,
                    "server_id": e.server_id,
                    "env": e.env,
                    "decision": e.decision,
                    "result_status": e.result_status,
                    "reason": e.reason,
                    "arg_keys": e.arg_keys,
                    "latency_ms": e.latency_ms,
                }
                for e in entries
            ],
        }
