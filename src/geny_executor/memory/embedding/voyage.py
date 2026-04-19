"""Voyage AI embedding backend.

Voyage doesn't require a heavyweight SDK — the public embeddings
endpoint is a single POST to
`https://api.voyageai.com/v1/embeddings` with a bearer token. We use
`httpx` (already transitive via `anthropic`) and expose the same
`EmbeddingClient` Protocol as the other backends.

Reference models and dimensions (2026-01 cutoff):
    voyage-3         → 1024
    voyage-3-large   → 1024
    voyage-code-3    → 1024
    voyage-finance-2 → 1024
    voyage-law-2     → 1024
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence

from geny_executor.memory.embedding.client import EmbeddingClient, EmbeddingError
from geny_executor.memory.provider import EmbeddingDescriptor


VOYAGE_DEFAULT_URL = "https://api.voyageai.com/v1/embeddings"

_VOYAGE_DIMS = {
    "voyage-3": 1024,
    "voyage-3-large": 1024,
    "voyage-code-3": 1024,
    "voyage-finance-2": 1024,
    "voyage-law-2": 1024,
}


class VoyageEmbeddingClient(EmbeddingClient):
    """Voyage AI embeddings over the REST endpoint.

    `transport` is an optional injection hook: a callable
    `async def transport(url, headers, json_body) -> dict` used in
    tests to stub out HTTP. If `None`, `httpx.AsyncClient` is used.
    """

    def __init__(
        self,
        *,
        model: str = "voyage-3",
        api_key: Optional[str] = None,
        dimension: Optional[int] = None,
        base_url: str = VOYAGE_DEFAULT_URL,
        transport: Optional[Any] = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("VOYAGE_API_KEY", "")
        self._dimension = dimension or _VOYAGE_DIMS.get(model, 0)
        self._base_url = base_url
        self._transport = transport
        self._descriptor = EmbeddingDescriptor(
            provider="voyage",
            model=model,
            dimension=self._dimension,
            metric="cosine",
            api_key_present=bool(self._api_key),
        )

    @property
    def descriptor(self) -> EmbeddingDescriptor:
        return self._descriptor

    async def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        payload = {"input": list(texts), "model": self._model}
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        body = await self._post(self._base_url, headers, payload)
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            raise EmbeddingError(f"voyage embed: malformed response: {body!r}")
        vectors: List[List[float]] = []
        for item in data:
            if not isinstance(item, dict) or "embedding" not in item:
                raise EmbeddingError(f"voyage embed: bad row: {item!r}")
            vec = item["embedding"]
            if not isinstance(vec, list):
                raise EmbeddingError(f"voyage embed: vec not list: {item!r}")
            vectors.append([float(x) for x in vec])
        if self._dimension == 0 and vectors:
            self._dimension = len(vectors[0])
            self._descriptor = EmbeddingDescriptor(
                provider="voyage",
                model=self._model,
                dimension=self._dimension,
                metric="cosine",
                api_key_present=bool(self._api_key),
            )
        return vectors

    async def close(self) -> None:
        return None

    # ── internal ────────────────────────────────────────────────────

    async def _post(self, url: str, headers: dict, body: dict) -> Any:
        if self._transport is not None:
            return await self._transport(url, headers, body)
        try:
            import httpx  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "VoyageEmbeddingClient needs httpx. It ships with anthropic>=0.52 "
                "as a transitive dep; ensure your environment resolves it."
            ) from exc
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, headers=headers, json=body)
            if resp.status_code != 200:
                raise EmbeddingError(f"voyage embed HTTP {resp.status_code}: {resp.text[:200]}")
            return resp.json()


__all__ = ["VoyageEmbeddingClient"]
