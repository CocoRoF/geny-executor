"""OpenAI embedding backend.

Wraps the `openai` SDK's `embeddings.create` endpoint. Package is an
optional dependency — importing this module without `openai>=1.50.0`
installed raises `ImportError` with a helpful message. Construction
takes a `model` (default `text-embedding-3-small`, 1536 dims) and an
`api_key` (falls back to `OPENAI_API_KEY` env var).

Batching: `openai.Embeddings.create` handles arbitrary-sized lists
server-side, but we still cap at `MAX_BATCH_SIZE=2048` per call to
keep request bodies reasonable and allow resume on partial failures.
"""

from __future__ import annotations

import asyncio
import os
from typing import List, Optional, Sequence

from geny_executor.memory.embedding.client import EmbeddingClient, EmbeddingError
from geny_executor.memory.provider import EmbeddingDescriptor


MAX_BATCH_SIZE = 2048


# Reference dimensions for OpenAI's current embedding families.
# Callers can override via `dimension=` kwarg to match a dedicated
# deployment.
_OPENAI_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddingClient(EmbeddingClient):
    """OpenAI embeddings via the official SDK."""

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimension: Optional[int] = None,
        client: Optional[object] = None,  # pre-built AsyncOpenAI, for tests
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._dimension = dimension or _OPENAI_DIMS.get(model, 0)
        self._client = client
        self._descriptor = EmbeddingDescriptor(
            provider="openai",
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
        client = self._ensure_client()
        out: List[List[float]] = []
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch = list(texts[i : i + MAX_BATCH_SIZE])
            try:
                # `openai>=1.x` exposes `await client.embeddings.create(...)`
                resp = await client.embeddings.create(input=batch, model=self._model)
            except Exception as exc:  # narrow is SDK-dependent
                raise EmbeddingError(f"openai embed failed: {exc}") from exc
            # SDK response: `data: List[Embedding(embedding: List[float])]`
            out.extend(item.embedding for item in resp.data)
        # Update descriptor dimension if we learned it at runtime
        if self._dimension == 0 and out:
            self._dimension = len(out[0])
            self._descriptor = EmbeddingDescriptor(
                provider="openai",
                model=self._model,
                dimension=self._dimension,
                metric="cosine",
                api_key_present=bool(self._api_key),
            )
        return out

    async def close(self) -> None:
        client = self._client
        if client is None:
            return
        closer = getattr(client, "close", None)
        if closer is None:
            return
        result = closer()
        if asyncio.iscoroutine(result):
            await result

    # ── internal ────────────────────────────────────────────────────

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import AsyncOpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "OpenAIEmbeddingClient requires the `openai` package. "
                "Install via `pip install geny-executor[openai]`."
            ) from exc
        self._client = AsyncOpenAI(api_key=self._api_key or None)
        return self._client


__all__ = ["OpenAIEmbeddingClient"]
