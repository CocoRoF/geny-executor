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
import logging
from typing import List, Optional, Sequence

from geny_executor.memory.embedding.client import (
    EmbeddingClient,
    EmbeddingError,
    _resolve_env_api_key,
)
from geny_executor.memory.provider import EmbeddingDescriptor


logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 2048

# OpenAI's text-embedding-* models reject inputs over 8192 tokens with a
# 400 that propagates as a hard EmbeddingError — a single over-long note
# (a whole conversation memory, an un-chunked document) could take the
# whole embedding path down. The BPE tokenizer never emits MORE tokens
# than the input's UTF-8 byte count (every token is ≥1 byte), so bounding
# a request's bytes bounds its tokens: ≤8192 bytes ⇒ ≤8192 tokens. We pass
# anything within that budget untouched and defensively truncate only the
# rare over-budget input (on a UTF-8 boundary), so embedding is crash-safe
# for every caller regardless of language. Callers that want full coverage
# of long text should chunk BEFORE embedding (the knowledge repository
# does, via Contextifier); this is the last-resort guard, not a substitute.
_MAX_EMBED_BYTES = 8192
_TRUNCATE_TO_BYTES = 8000  # margin applied when a cut is unavoidable
_truncation_warned = False


def _bound_input(text: str) -> str:
    global _truncation_warned
    encoded = text.encode("utf-8")
    if len(encoded) <= _MAX_EMBED_BYTES:
        return text
    if not _truncation_warned:
        logger.warning(
            "embedding input exceeds the 8192-token budget (%d bytes); "
            "truncating to %d bytes for the vector. Chunk long text before "
            "embedding to avoid losing coverage.",
            len(encoded), _TRUNCATE_TO_BYTES,
        )
        _truncation_warned = True
    return encoded[:_TRUNCATE_TO_BYTES].decode("utf-8", errors="ignore")


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
        # Explicit api_key (CredentialBundle 'embedding' channel via the
        # factory, or direct construction) always wins. The env ladder is
        # a DEPRECATED fallback reached only when nothing was passed —
        # it logs a one-time migration warning (audit §2.6).
        self._api_key = api_key or _resolve_env_api_key("openai", "OPENAI_API_KEY")
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
        # Bound each input to the model's token budget so one over-long
        # text can never 400 the whole batch (crash-safety net).
        texts = [_bound_input(t) for t in texts]
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch = list(texts[i : i + MAX_BATCH_SIZE])
            try:
                # `openai>=1.x` exposes `await client.embeddings.create(...)`
                resp = await client.embeddings.create(input=batch, model=self._model)
            except Exception as exc:  # narrow is SDK-dependent
                raise EmbeddingError(
                    f"openai embed failed: {exc}",
                    category=_classify_openai_error(exc),
                ) from exc
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


def _classify_openai_error(exc: Exception) -> str:
    """Map a typed `openai` SDK exception to an `EmbeddingError` category.

    Uses the SDK's exception hierarchy rather than message text — the
    Google client's ``str(e)`` substring matching is exactly the
    anti-pattern the audit flagged ('400'-containing 500s misroute).
    Falls back to ``'unknown'`` when the SDK isn't importable (the
    caller injected a pre-built client object in tests) or the type
    isn't one we recognise; ``'unknown'`` never trips the vector
    layer's breaker, which is the safe default for a misjudged error.
    """
    try:
        import openai  # type: ignore
    except ImportError:
        return "unknown"
    # Order matters: AuthenticationError / PermissionDeniedError /
    # RateLimitError all subclass APIStatusError; check the specific
    # types before any status-code generalisation.
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return "auth"
    if isinstance(exc, openai.RateLimitError):
        return "quota"
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return "transient"
    if isinstance(exc, openai.InternalServerError):
        return "transient"
    return "unknown"


__all__ = ["OpenAIEmbeddingClient"]
