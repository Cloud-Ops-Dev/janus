"""Blended capability ranking (Phase 5, infra-idf).

Orders policy-allowed candidates by a blend of signals (design §5.7):

  score = w_semantic · cosine(query, capability)
        + w_tag      · tag overlap
        + w_keyword  · keyword overlap
        + w_health   · downstream connected

Profile and environment are applied by the broker as HARD filters *before*
ranking (a denied or out-of-env capability never reaches the ranker), so they
shape the candidate set rather than the soft score. Historical-success blending
is a declared extension point (weight 0 by default) pending a global success
counter in the audit layer.

Capability vectors are computed over model-visible text only (title + summary +
tags + tool name) and cached by capability id — never over raw descriptions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from janus.registry.registry import Capability
from janus.search.embedder import Embedder, HashingEmbedder, Vector, cosine

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _capability_text(cap: Capability) -> str:
    return f"{cap.title} {cap.summary} {' '.join(cap.tags)} {cap.downstream_tool_name}"


@dataclass(frozen=True)
class RankWeights:
    semantic: float = 1.0
    tag: float = 0.5
    keyword: float = 0.5
    health: float = 0.2
    historical: float = 0.0  # extension point — see module docstring


@dataclass
class BlendedRanker:
    embedder: Embedder = field(default_factory=HashingEmbedder)
    weights: RankWeights = field(default_factory=RankWeights)
    # Below this blended relevance, a non-empty query's match is treated as noise
    # (feature-hash collisions / incidental sub-word overlap) and dropped.
    min_relevance: float = 0.05
    _cap_vectors: dict[str, Vector] = field(default_factory=dict, init=False)
    _cap_tokens: dict[str, set[str]] = field(default_factory=dict, init=False)

    def prepare(self, capabilities: dict[str, Capability]) -> None:
        """Precompute + cache capability vectors/tokens (call once per registry)."""
        ids = list(capabilities)
        vectors = self.embedder.embed([_capability_text(capabilities[i]) for i in ids])
        self._cap_vectors = dict(zip(ids, vectors, strict=True))
        self._cap_tokens = {i: _tokens(_capability_text(capabilities[i])) for i in ids}

    def _vector_for(self, cap: Capability) -> Vector:
        cached = self._cap_vectors.get(cap.id)
        if cached is None:
            cached = self.embedder.embed([_capability_text(cap)])[0]
            self._cap_vectors[cap.id] = cached
        return cached

    def _tokens_for(self, cap: Capability) -> set[str]:
        cached = self._cap_tokens.get(cap.id)
        if cached is None:
            cached = _tokens(_capability_text(cap))
            self._cap_tokens[cap.id] = cached
        return cached

    def rank(
        self,
        query: str,
        candidates: list[Capability],
        *,
        connected: set[str],
    ) -> list[tuple[float, Capability]]:
        """Return ``(score, capability)`` sorted best-first.

        An empty query ranks by health + a neutral keyword score (1.0), so it
        behaves like "list everything allowed" — matching the keyword fallback.
        """
        q_terms = _tokens(query)
        q_vec = self.embedder.embed([query])[0] if q_terms else []
        w = self.weights
        scored: list[tuple[float, Capability]] = []
        for cap in candidates:
            semantic = cosine(q_vec, self._vector_for(cap)) if q_terms else 0.0
            tag_overlap = self._overlap(q_terms, _tokens(" ".join(cap.tags)))
            keyword = self._overlap(q_terms, self._tokens_for(cap)) if q_terms else 1.0
            relevance = w.semantic * semantic + w.tag * tag_overlap + w.keyword * keyword
            # A non-empty query with only noise-level signal for this capability
            # is not a relevant result — the health boost alone must not surface
            # every connected tool (design §7: only relevant results).
            if q_terms and relevance < self.min_relevance:
                continue
            health = 1.0 if cap.server_id in connected else 0.0
            score = relevance + w.health * health
            scored.append((score, cap))
        scored.sort(key=lambda pair: (pair[0], pair[1].id), reverse=True)
        return scored

    @staticmethod
    def _overlap(q_terms: set[str], doc_terms: set[str]) -> float:
        if not q_terms:
            return 0.0
        return len(q_terms & doc_terms) / len(q_terms)
