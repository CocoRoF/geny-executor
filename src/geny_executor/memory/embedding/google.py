"""Google embedding backend.

Uses the `google-genai` SDK (`pip install geny-executor[google]`).
Models and dimensions:

    text-embedding-004      → 768
    text-multilingual-embedding-002 → 768
    gemini-embedding-001    → 3072 (can be truncated to 768 / 1536)
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from geny_executor.memory.embedding.client import (
    EmbeddingClient,
    EmbeddingError,
    _LoopBoundClientMixin,
    _resolve_env_api_key,
)
from geny_executor.memory.provider import EmbeddingDescriptor


_GOOGLE_DIMS = {
    "text-embedding-004": 768,
    "text-multilingual-embedding-002": 768,
    "gemini-embedding-001": 3072,
}


class GoogleEmbeddingClient(_LoopBoundClientMixin, EmbeddingClient):
    """Google Generative AI embeddings."""

    def __init__(
        self,
        *,
        model: str = "text-embedding-004",
        api_key: Optional[str] = None,
        dimension: Optional[int] = None,
        client: Optional[Any] = None,
    ) -> None:
        self._model = model
        # Explicit api_key wins; env ladder is the DEPRECATED fallback
        # (one-time warning, see embedding/client.py — audit §2.6).
        self._api_key = api_key or _resolve_env_api_key(
            "google", "GOOGLE_API_KEY", "GEMINI_API_KEY"
        )
        self._dimension = dimension or _GOOGLE_DIMS.get(model, 0)
        # Loop-safe client cache (see _LoopBoundClientMixin).
        self._client = client
        self._client_loop = None
        self._injected_client = client is not None
        self._descriptor = EmbeddingDescriptor(
            provider="google",
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
        # Loop-safe: pooled client on the stable loop, ephemeral (closed
        # below) on a sync-bridge loop.
        client, ephemeral = self._acquire_client()
        try:
            try:
                # google-genai v1: `client.aio.models.embed_content(...)`
                resp = await client.aio.models.embed_content(
                    model=self._model,
                    contents=list(texts),
                )
            except Exception as exc:
                raise EmbeddingError(f"google embed failed: {exc}") from exc
            embeds = getattr(resp, "embeddings", None)
            if embeds is None:
                raise EmbeddingError(f"google embed: missing 'embeddings' in {resp!r}")
            vectors: List[List[float]] = []
            for item in embeds:
                values = getattr(item, "values", None)
                if values is None:
                    raise EmbeddingError(f"google embed: bad row: {item!r}")
                vectors.append([float(x) for x in values])
            if self._dimension == 0 and vectors:
                self._dimension = len(vectors[0])
                self._descriptor = EmbeddingDescriptor(
                    provider="google",
                    model=self._model,
                    dimension=self._dimension,
                    metric="cosine",
                    api_key_present=bool(self._api_key),
                )
            return vectors
        finally:
            if ephemeral:
                await self._aclose_client(client)

    async def close(self) -> None:
        await self._close_cached_client()

    # ── internal ────────────────────────────────────────────────────

    def _build_client(self) -> object:
        try:
            from google import genai  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "GoogleEmbeddingClient requires `google-genai`. "
                "Install via `pip install geny-executor[google]`."
            ) from exc
        return genai.Client(api_key=self._api_key or None)


__all__ = ["GoogleEmbeddingClient"]
