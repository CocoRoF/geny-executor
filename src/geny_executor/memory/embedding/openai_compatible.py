"""OpenAI-compatible embedding backend — any self-hosted `/v1/embeddings`.

First-class support for locally-served embedding models: vLLM, Ollama,
LM Studio, text-embeddings-inference, LiteLLM proxies, or any gateway
that speaks the OpenAI embeddings wire format::

    POST {base_url}/embeddings
    {"input": ["...", ...], "model": "<served-model-name>"}
    → {"data": [{"index": 0, "embedding": [...]}, ...]}

Differences from the ``openai`` backend (which drives the official SDK
against api.openai.com): this client takes an explicit ``base_url``,
treats the API key as OPTIONAL (local servers frequently run with no
auth — the ``Authorization`` header is only sent when a key is set),
and never consults the deprecated env-var credential ladder (a local
endpoint's key has no well-known env name).

Dimension handling mirrors the other REST backends: pass ``dimension``
when known, else it self-heals from the first response vector. No
model→dimension table exists here by design — served model names are
deployment-specific and unknowable to a general-purpose library.

Config example (memory provider ``embedding`` block)::

    {"provider": "openai_compatible",
     "model": "bge-m3",
     "options": {"base_url": "http://vllm-host:8000/v1"}}

``base_url`` may be the API root (``.../v1``) or the full endpoint
(``.../v1/embeddings``) — both normalize to the same request URL.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence

from geny_executor.memory.embedding.client import (
    EmbeddingClient,
    EmbeddingError,
    _bound_input,
    category_for_http_status,
    iter_embed_batches,
)
from geny_executor.memory.provider import EmbeddingDescriptor

# Conservative request cap; the shared byte budget (iter_embed_batches)
# splits large documents independently of this count.
_DEFAULT_MAX_BATCH = 128


def _normalize_endpoint(base_url: str) -> str:
    """Accept an API root or the full endpoint; return the request URL.

    ``http://h:8000/v1`` → ``http://h:8000/v1/embeddings``
    ``http://h:8000/v1/embeddings`` → unchanged.
    """
    url = (base_url or "").strip().rstrip("/")
    if not url:
        raise ValueError("openai_compatible embedding requires a non-empty base_url")
    if url.endswith("/embeddings"):
        return url
    return f"{url}/embeddings"


class OpenAICompatibleEmbeddingClient(EmbeddingClient):
    """Embeddings over any OpenAI-compatible REST endpoint.

    `transport` is an optional injection hook: a callable
    `async def transport(url, headers, json_body) -> dict` used in
    tests to stub out HTTP. If `None`, `httpx.AsyncClient` is used
    per call (created and closed inside the running loop — inherently
    loop-safe for hosts that drive memory from short-lived loops).
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        dimension: Optional[int] = None,
        timeout_s: float = 30.0,
        max_batch: int = _DEFAULT_MAX_BATCH,
        transport: Optional[Callable[..., Any]] = None,
    ) -> None:
        if not model or not str(model).strip():
            raise ValueError(
                "openai_compatible embedding requires an explicit model name "
                "(the served model id — e.g. vLLM's --served-model-name)"
            )
        self._endpoint = _normalize_endpoint(base_url)
        self._model = str(model).strip()
        # Optional by design — local endpoints often run authless. No env
        # ladder fallback: a custom endpoint has no well-known env name.
        self._api_key = (api_key or "").strip()
        self._dimension = int(dimension or 0)
        self._timeout_s = float(timeout_s)
        self._max_batch = max(1, int(max_batch))
        self._transport = transport
        self._descriptor = EmbeddingDescriptor(
            provider="openai_compatible",
            model=self._model,
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
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        bounded = [_bound_input(t) for t in texts]
        vectors: List[List[float]] = []
        for batch in iter_embed_batches(bounded, max_count=self._max_batch):
            payload = {"input": list(batch), "model": self._model}
            body = await self._post(self._endpoint, headers, payload)
            vectors.extend(self._parse_rows(body, expected=len(batch)))
        if self._dimension == 0 and vectors:
            # Self-heal: served models have deployment-specific dimensions.
            self._dimension = len(vectors[0])
            self._descriptor = EmbeddingDescriptor(
                provider="openai_compatible",
                model=self._model,
                dimension=self._dimension,
                metric="cosine",
                api_key_present=bool(self._api_key),
            )
        return vectors

    async def close(self) -> None:
        return None

    # ── internal ────────────────────────────────────────────────────

    @staticmethod
    def _parse_rows(body: Any, *, expected: int) -> List[List[float]]:
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            raise EmbeddingError(
                f"openai_compatible embed: malformed response: {str(body)[:200]!r}",
                category="invalid",
            )
        rows: List[Any] = data
        # The OpenAI wire format carries an `index` per row; servers are
        # ordered in practice, but sort defensively when every row has one.
        if rows and all(isinstance(r, dict) and isinstance(r.get("index"), int) for r in rows):
            rows = sorted(rows, key=lambda r: r["index"])
        vectors: List[List[float]] = []
        for item in rows:
            if not isinstance(item, dict) or "embedding" not in item:
                raise EmbeddingError(
                    f"openai_compatible embed: bad row: {str(item)[:200]!r}",
                    category="invalid",
                )
            vec = item["embedding"]
            if not isinstance(vec, list):
                raise EmbeddingError(
                    f"openai_compatible embed: vec not list: {str(item)[:200]!r}",
                    category="invalid",
                )
            vectors.append([float(x) for x in vec])
        if expected and len(vectors) != expected:
            raise EmbeddingError(
                f"openai_compatible embed: row count mismatch "
                f"(sent {expected}, got {len(vectors)})",
                category="invalid",
            )
        return vectors

    async def _post(self, url: str, headers: dict, body: dict) -> Any:
        if self._transport is not None:
            return await self._transport(url, headers, body)
        try:
            import httpx  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "OpenAICompatibleEmbeddingClient needs httpx. It ships with "
                "anthropic>=0.52 as a transitive dep; ensure your environment "
                "resolves it."
            ) from exc
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                resp = await client.post(url, headers=headers, json=body)
        except httpx.HTTPError as exc:
            # Connect failures / timeouts: retrying next write is the right
            # policy — transient (never trips the vector layer's auth breaker).
            raise EmbeddingError(
                f"openai_compatible embed transport failure: {exc}",
                category="transient",
            ) from exc
        if resp.status_code != 200:
            raise EmbeddingError(
                f"openai_compatible embed HTTP {resp.status_code}: {resp.text[:200]}",
                category=category_for_http_status(resp.status_code),
            )
        return resp.json()


__all__ = ["OpenAICompatibleEmbeddingClient"]
