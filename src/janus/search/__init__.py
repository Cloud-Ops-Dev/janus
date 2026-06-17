"""Semantic capability search (Phase 5, infra-idf).

A pluggable :class:`Embedder` (default: dependency-free :class:`HashingEmbedder`)
plus a :class:`BlendedRanker` that blends semantic similarity with tag/keyword
overlap and downstream health. Hard policy/env/risk filters are applied by the
broker *before* ranking — the ranker only orders the survivors.
"""

from janus.search.embedder import Embedder, HashingEmbedder, cosine
from janus.search.ranker import BlendedRanker, RankWeights

__all__ = [
    "BlendedRanker",
    "Embedder",
    "HashingEmbedder",
    "RankWeights",
    "cosine",
]
