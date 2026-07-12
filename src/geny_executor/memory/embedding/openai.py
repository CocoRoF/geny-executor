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

import logging
from typing import List, Optional, Sequence

from geny_executor.memory.embedding.client import (
    EmbeddingClient,
    EmbeddingError,
    _LoopBoundClientMixin,
    _bound_input,
    _MAX_EMBED_BYTES,  # noqa: F401 — re-exported for back-compat
    _resolve_env_api_key,
    _TRUNCATE_TO_BYTES,  # noqa: F401 — re-exported for back-compat
    iter_embed_batches,
)
from geny_executor.memory.provider import EmbeddingDescriptor


logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 2048


# Reference dimensions for OpenAI's current embedding families.
# Callers can override via `dimension=` kwarg to match a dedicated
# deployment.
_OPENAI_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddingClient(_LoopBoundClientMixin, EmbeddingClient):
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
        # Loop-safe client cache (see _LoopBoundClientMixin). A caller-
        # supplied client is used verbatim (tests); otherwise lazily built
        # per-loop by _build_client below.
        self._client = client
        self._client_loop = None
        self._injected_client = client is not None
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
        # Loop-safe: a pooled client on the stable loop, or a short-lived
        # client on an ephemeral (sync-bridge) loop that we close below.
        client, ephemeral = self._acquire_client()
        try:
            out: List[List[float]] = []
            # Bound each input (per-input cap), then split into requests
            # under BOTH the count and the cumulative-token budget so a
            # large document's chunks can't 400 the request (audit D2).
            bounded = [_bound_input(t) for t in texts]
            for batch in iter_embed_batches(bounded, max_count=MAX_BATCH_SIZE):
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
        finally:
            if ephemeral:
                await self._aclose_client(client)

    async def close(self) -> None:
        await self._close_cached_client()

    # ── internal ────────────────────────────────────────────────────

    def _build_client(self) -> object:
        try:
            from openai import AsyncOpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "OpenAIEmbeddingClient requires the `openai` package. "
                "Install via `pip install geny-executor[openai]`."
            ) from exc
        return AsyncOpenAI(api_key=self._api_key or None)


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
    # A 400 (over-token-limit request, malformed input, unknown model) is
    # permanent — retrying the identical request 400s forever. 'invalid'
    # so callers stop hot-retrying (audit D2). BadRequestError also covers
    # UnprocessableEntityError (422) in the SDK hierarchy.
    if isinstance(exc, openai.BadRequestError):
        return "invalid"
    return "unknown"


__all__ = ["OpenAIEmbeddingClient"]
