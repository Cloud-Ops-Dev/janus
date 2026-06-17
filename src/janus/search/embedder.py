"""Embeddings for capability search (Phase 5, infra-idf).

The :class:`Embedder` protocol decouples ranking from any specific model, so the
default dependency-free :class:`HashingEmbedder` can be swapped for a local
sentence-transformer or a remote claymore-1 embedding endpoint without touching
the ranker — keeping Janus vendor-neutral and offline-capable (design §5.7).

``HashingEmbedder`` is a deterministic feature-hashing embedding over word
unigrams + character n-grams, L2-normalised. It is NOT a neural model, but its
sub-word features let it match morphological / partial variants that exact-token
keyword overlap misses (e.g. "recall"/"recollection", "memories"/"memory"),
which is enough to beat keyword ranking on realistic queries with zero deps.

SECURITY: only ever embed model-visible text (title/summary/tags) — never raw
downstream descriptions. The registry does not even store raw descriptions
(only their hash), so this boundary holds by construction.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, runtime_checkable

_TOKEN_RE = re.compile(r"[a-z0-9]+")

Vector = list[float]


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[Vector]:
        """Return one L2-normalised vector per input text."""
        ...


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity of two equal-length vectors (0.0 if either is empty)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    # Vectors are L2-normalised by the embedder, so cosine == dot product; guard
    # anyway in case an unnormalised vector is passed in.
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot


class HashingEmbedder:
    """Deterministic feature-hashing embedding (word unigrams + char n-grams).

    Dependency-free and stable across processes (blake2b feature hashing, not
    Python's salted ``hash``), so cached vectors stay valid.
    """

    def __init__(self, dim: int = 4096, ngram: int = 3) -> None:
        # 4096 buckets keep feature-hash collisions low enough that unrelated
        # text scores ~0 (so the ranker's relevance floor can exclude noise).
        if dim <= 0:
            raise ValueError("dim must be positive")
        if ngram < 2:
            raise ValueError("ngram must be >= 2")
        self._dim = dim
        self._n = ngram

    def embed(self, texts: list[str]) -> list[Vector]:
        return [self._vector(text) for text in texts]

    def _features(self, text: str) -> list[str]:
        words = _TOKEN_RE.findall(text.lower())
        feats: list[str] = list(words)  # word unigrams
        for word in words:
            padded = f"#{word}#"
            for i in range(len(padded) - self._n + 1):
                feats.append(padded[i : i + self._n])  # character n-grams
        return feats

    def _bucket(self, feature: str) -> int:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self._dim

    def _vector(self, text: str) -> Vector:
        vec = [0.0] * self._dim
        for feature in self._features(text):
            vec[self._bucket(feature)] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:
            return vec
        return [x / norm for x in vec]
