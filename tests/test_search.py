"""Semantic capability search tests (Phase 5, infra-idf).

Covers the embedder (determinism, similarity ordering), the blended ranker
(relevance gate, empty query, health/tag blend), and the headline acceptance:
on a small eval set, semantic ranking surfaces capabilities that exact-token
keyword overlap misses — while hard policy filters still apply first.
"""

from __future__ import annotations

import math

from janus.audit import InMemoryAuditSink
from janus.broker import Broker
from janus.downstream import DownstreamClientManager
from janus.policy import Decision, PolicyContext, PolicyDecision
from janus.registry import (
    Capability,
    EnvScope,
    Registry,
    RiskTier,
    Server,
    Transport,
)
from janus.search import BlendedRanker, HashingEmbedder, cosine


# --------------------------------------------------------------------------- #
# embedder
# --------------------------------------------------------------------------- #
def test_embedder_l2_normalised() -> None:
    vec = HashingEmbedder().embed(["search captured thoughts"])[0]
    assert math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, rel_tol=1e-9)


def test_embedder_is_deterministic_across_instances() -> None:
    a = HashingEmbedder().embed(["semantic memory recall"])[0]
    b = HashingEmbedder().embed(["semantic memory recall"])[0]
    assert a == b  # blake2b feature hashing -> stable, not salted


def test_embedder_similar_text_scores_higher() -> None:
    emb = HashingEmbedder()
    memory, memories, health = emb.embed(["memory", "memories", "health check probe"])
    # "memory"/"memories" share sub-word n-grams; "health check" does not.
    assert cosine(memory, memories) > cosine(memory, health)


def test_cosine_empty_is_zero() -> None:
    assert cosine([], [1.0, 2.0]) == 0.0


# --------------------------------------------------------------------------- #
# ranker
# --------------------------------------------------------------------------- #
def _cap(cid: str, server: str, tool: str, summary: str, tags: list[str]) -> Capability:
    return Capability(
        id=cid,
        server_id=server,
        downstream_tool_name=tool,
        title=cid,
        summary=summary,
        risk=RiskTier.READ_ONLY,
        env_scope=[EnvScope.DEV, EnvScope.PROD_SAFE],
        approved=True,
        tags=tags,
    )


_CAPS = {
    "ob.search": _cap(
        "ob.search", "ob", "search_thoughts",
        "Semantic search over captured thoughts by meaning",
        ["memory", "semantic-search", "notes"],
    ),
    "pc.health": _cap(
        "pc.health", "pc", "health_check",
        "Liveness probe for the backend service",
        ["health"],
    ),
    "bd.list": _cap(
        "bd.list", "bd", "list_issues",
        "List issues filtered by status or priority",
        ["issues", "tasks", "tracker"],
    ),
}


def test_ranker_relevance_gate_excludes_irrelevant() -> None:
    ranker = BlendedRanker()
    ranker.prepare(_CAPS)
    ranked = ranker.rank("xyzzy-nonsense", list(_CAPS.values()), connected=set())
    assert ranked == []  # no lexical/semantic signal -> nothing relevant


def test_ranker_empty_query_returns_all() -> None:
    ranker = BlendedRanker()
    ranker.prepare(_CAPS)
    ranked = ranker.rank("", list(_CAPS.values()), connected=set())
    assert {cap.id for _s, cap in ranked} == set(_CAPS)


def test_ranker_health_breaks_ties_toward_connected() -> None:
    ranker = BlendedRanker()
    ranker.prepare(_CAPS)
    # empty query -> equal relevance; the connected server's cap ranks first.
    ranked = ranker.rank("", list(_CAPS.values()), connected={"bd"})
    assert ranked[0][1].server_id == "bd"


# --------------------------------------------------------------------------- #
# acceptance — semantic beats keyword on a small eval set
# --------------------------------------------------------------------------- #
class AllowAllPolicy:
    def evaluate(self, ctx: PolicyContext) -> PolicyDecision:
        return PolicyDecision(
            Decision.ALLOW, "allow", ctx.capability.id, ctx.capability.risk
        )


class DenyWritesPolicy:
    def evaluate(self, ctx: PolicyContext) -> PolicyDecision:
        if ctx.capability.risk is RiskTier.READ_ONLY:
            return PolicyDecision(Decision.ALLOW, "read ok", ctx.capability.id, ctx.capability.risk)
        return PolicyDecision(Decision.DENY, "write denied", ctx.capability.id, ctx.capability.risk)


def _registry(caps: dict[str, Capability]) -> Registry:
    servers = {
        sid: Server(
            id=sid, display_name=sid, transport=Transport.STDIO, command="x",
            risk_ceiling=RiskTier.EXTERNAL_WRITE,
            default_env_scope=[EnvScope.DEV, EnvScope.PROD_SAFE],
        )
        for sid in {c.server_id for c in caps.values()}
    }
    return Registry(servers=servers, capabilities=caps)


def _broker(caps: dict[str, Capability], *, ranker: BlendedRanker | None, policy: object) -> Broker:
    reg = _registry(caps)
    return Broker(
        reg,
        DownstreamClientManager(reg.servers),
        policy,  # type: ignore[arg-type]
        InMemoryAuditSink(),
        ranker=ranker,
        default_env=EnvScope.PROD_SAFE,
    )


def test_semantic_search_beats_keyword_on_morphological_query() -> None:
    # "memories" shares no exact token with any capability, but is sub-word
    # similar to ob.search's "memory" tag + summary.
    ranker = BlendedRanker()
    ranker.prepare(_CAPS)

    semantic = _broker(_CAPS, ranker=ranker, policy=AllowAllPolicy())
    sem_ids = [r["capability_id"] for r in semantic.capability_search("memories")["results"]]
    assert sem_ids and sem_ids[0] == "ob.search"

    keyword = _broker(_CAPS, ranker=None, policy=AllowAllPolicy())
    kw_ids = [r["capability_id"] for r in keyword.capability_search("memories")["results"]]
    # exact-token keyword overlap finds nothing for "memories".
    assert "ob.search" not in kw_ids


def test_search_applies_hard_policy_filter_before_ranking() -> None:
    caps = dict(_CAPS)
    caps["ob.capture"] = Capability(
        id="ob.capture", server_id="ob", downstream_tool_name="capture_thought",
        title="Capture thought", summary="Write a memory to semantic storage",
        risk=RiskTier.EXTERNAL_WRITE, env_scope=[EnvScope.DEV, EnvScope.PROD_SAFE],
        approved=True, tags=["memory", "write"],
    )
    ranker = BlendedRanker()
    ranker.prepare(caps)
    broker = _broker(caps, ranker=ranker, policy=DenyWritesPolicy())
    # "memory" is highly relevant to ob.capture, but it is policy-denied -> absent.
    ids = [r["capability_id"] for r in broker.capability_search("memory")["results"]]
    assert "ob.capture" not in ids
    assert "ob.search" in ids
