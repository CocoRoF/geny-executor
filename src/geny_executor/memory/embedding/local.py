"""Local deterministic embedding backend.

Zero external dependencies. Produces a fixed-dimension vector from a
SHA-256 hash of the text plus a token-level hashing-trick bag. Used
for:

- unit / contract tests that need *some* embedding without an API key,
- CI runs that can't reach external providers,
- offline deployments that don't need semantic quality.

It is NOT a semantic embedder. Cosine similarity between unrelated
texts is not meaningful. Provider descriptor is `local/hash-v1`.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List, Sequence

from geny_executor.memory.embedding.client import EmbeddingClient
from geny_executor.memory.provider import EmbeddingDescriptor


_TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣_]+")


class LocalHashEmbeddingClient(EmbeddingClient):
    """Deterministic hash-based embedder.

    Pipeline:
      1. Lowercase + tokenize on `\\w+` (including Hangul).
      2. For each token, SHA-256 → take bytes 0..3 as index mod D,
         bytes 4..5 as signed contribution (+1 or -1).
      3. Seed each vector slot with a salt from the whole-text hash
         so zero-token inputs still produce a non-zero vector.
      4. L2-normalise so cosine similarity reduces to dot product.
    """

    def __init__(self, *, dimension: int = 384, model: str = "hash-v1") -> None:
        if dimension < 32 or dimension > 4096:
            raise ValueError("dimension must be in [32, 4096]")
        self._dimension = dimension
        self._model = model
        self._descriptor = EmbeddingDescriptor(
            provider="local",
            model=model,
            dimension=dimension,
            metric="cosine",
            api_key_present=False,
        )

    @property
    def descriptor(self) -> EmbeddingDescriptor:
        return self._descriptor

    async def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    async def close(self) -> None:
        return None

    # ── internal ────────────────────────────────────────────────────

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self._dimension
        lowered = (text or "").lower()
        # Seed from whole-text hash so empty token bags still differ
        seed = hashlib.sha256(lowered.encode("utf-8")).digest()
        for i in range(self._dimension):
            byte = seed[i % len(seed)]
            vec[i] = ((byte / 255.0) - 0.5) * 0.01

        for token in _TOKEN_RE.findall(lowered):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if (digest[4] & 1) == 0 else -1.0
            magnitude = 1.0 + (digest[5] / 255.0) * 0.5
            vec[idx] += sign * magnitude

        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:
            return vec
        return [x / norm for x in vec]


__all__ = ["LocalHashEmbeddingClient"]
