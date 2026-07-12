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

from typing import Any, List, Optional, Sequence

from geny_executor.memory.embedding.client import (
    EmbeddingClient,
    EmbeddingError,
    _bound_input,
    _resolve_env_api_key,
    iter_embed_batches,
)
from geny_executor.memory.provider import EmbeddingDescriptor

# Voyage caps requests at 128 inputs / 120k-1M tokens depending on model;
# cap conservatively and let the shared byte budget split large docs.
_VOYAGE_MAX_BATCH = 128


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
        # Explicit api_key wins; env ladder is the DEPRECATED fallback
        # (one-time warning, see embedding/client.py — audit §2.6).
        self._api_key = api_key or _resolve_env_api_key("voyage", "VOYAGE_API_KEY")
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
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        bounded = [_bound_input(t) for t in texts]
        vectors: List[List[float]] = []
        for batch in iter_embed_batches(bounded, max_count=_VOYAGE_MAX_BATCH):
            payload = {"input": list(batch), "model": self._model}
            body = await self._post(self._base_url, headers, payload)
            data = body.get("data") if isinstance(body, dict) else None
            if not isinstance(data, list):
                raise EmbeddingError(f"voyage embed: malformed response: {body!r}")
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
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, headers=headers, json=body)
        except httpx.HTTPError as exc:
            # Connect failures / timeouts: retrying next write is the
            # right policy, so classify as transient (never trips the
            # vector layer's auth breaker).
            raise EmbeddingError(
                f"voyage embed transport failure: {exc}",
                category="transient",
            ) from exc
        if resp.status_code != 200:
            raise EmbeddingError(
                f"voyage embed HTTP {resp.status_code}: {resp.text[:200]}",
                category=_category_for_status(resp.status_code),
            )
        return resp.json()


def _category_for_status(status: int) -> str:
    """HTTP status → `EmbeddingError` category for the Voyage REST API.

    401/403 means the bearer token is wrong — no amount of retrying
    fixes it, so it must count toward the vector layer's trip-once
    breaker. 429 is quota (retry later may work). 408/5xx are
    transient server-side conditions. Anything else (404 model name
    typo, 400 payload issue) stays 'unknown' so it keeps the
    conservative traceback-logging path.
    """
    if status in (401, 403):
        return "auth"
    if status == 429:
        return "quota"
    if status == 408 or status >= 500:
        return "transient"
    # A 400/404/422 will never succeed on retry (bad payload, over-limit,
    # unknown model) — 'invalid' so the caller stops hot-retrying (audit D2).
    if 400 <= status < 500:
        return "invalid"
    return "unknown"


__all__ = ["VoyageEmbeddingClient"]
