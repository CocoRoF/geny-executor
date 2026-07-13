"""EmbeddingClient Protocol.

A provider-agnostic surface for turning text into dense vectors.
Mirrors the shape of Geny's embedding strategy without importing any
Geny code. The four concrete backends (`openai`, `voyage`, `google`,
`local`) conform to this Protocol and are dispatched by
`create_embedding_client` in `registry.py`.

The Protocol is deliberately minimal: one async method
(`embed(texts)`) plus a descriptor property. Batch size, retries, and
rate-limit handling are backend concerns; callers hand in a list and
get back a list.

Error contract (2.2.0, audit §2.6)
----------------------------------

`EmbeddingError` carries a ``category`` so callers can react
structurally instead of grepping messages. The categories mirror the
MCP boundary's NEEDS_AUTH FSM — the same package already solved this
problem once, and the live 401-spam incident showed what happens when
an embedding key goes bad without classification: every note write
re-attempted the call and logged a full traceback, forever.

    'auth'      — credentials rejected (401/403). Retrying with the
                  same key cannot succeed; the vector layer trips its
                  breaker after a few of these.
    'quota'     — rate limit / billing exhaustion (429). Retrying
                  *later* may succeed; never trips the breaker.
    'transient' — connection / timeout / 5xx. Retry-next-time is the
                  correct policy; never trips the breaker.
    'unknown'   — anything unclassified. Treated conservatively
                  (no breaker trip, traceback retained in logs).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import List, Optional, Protocol, Sequence, runtime_checkable

from geny_executor.memory.provider import CostEvent, EmbeddingDescriptor

logger = logging.getLogger(__name__)


#: Valid values for ``EmbeddingError.category``.
#: ``invalid`` (2.51.0, audit D2): a permanent 4xx the request will NEVER
#: satisfy by retrying — a malformed payload, an over-token-limit request,
#: an unknown model. Distinct from ``quota`` (retry later may work) and
#: ``transient`` (retry next time). Callers must NOT hot-retry an
#: ``invalid`` error; it does not trip the auth breaker either.
EMBEDDING_ERROR_CATEGORIES = frozenset({"auth", "quota", "transient", "invalid", "unknown"})

# ── shared input bounding + request batching (all embedding backends) ──
#
# Two independent limits every embeddings endpoint enforces:
#   (1) per-INPUT token cap (OpenAI 8192) — one over-long text 400s.
#   (2) per-REQUEST total-token cap (OpenAI 300k) — a batch whose inputs
#       SUM past the ceiling 400s, silently taking a whole document's
#       vectors down (audit D2, the confirmed prod incident).
# The BPE tokenizer never emits MORE tokens than the input's UTF-8 byte
# count (every token is >= 1 byte), so bounding bytes bounds tokens for
# both limits, in every language, without a tokenizer dependency.

_MAX_EMBED_BYTES = 8192  # per-input ceiling
_TRUNCATE_TO_BYTES = 8000  # margin applied when a per-input cut is unavoidable
#: Per-request byte budget. Kept well under OpenAI's 300k-token ceiling
#: (bytes >= tokens, so <=280k bytes ⇒ <280k tokens) so a large document's
#: chunks split across requests instead of 400ing. Conservative for ASCII
#: (over-splits) but never wrong — correctness beats request count here.
_MAX_REQUEST_BYTES = 280_000

_truncation_warned = False


def _bound_input(text: str) -> str:
    """Truncate one input to the per-input byte/token ceiling (last resort).

    Callers that want full coverage of long text should chunk BEFORE
    embedding (the knowledge repository does, via Contextifier); this is
    the crash-safety net, not a substitute.
    """
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


def iter_embed_batches(
    texts: Sequence[str],
    *,
    max_count: int,
    max_bytes: int = _MAX_REQUEST_BYTES,
) -> "List[List[str]]":
    """Split per-input-bounded texts into requests under BOTH a count and a
    cumulative-byte budget. A single (already <=8192-byte) input never
    exceeds ``max_bytes`` alone, so no batch is ever empty."""
    batches: List[List[str]] = []
    batch: List[str] = []
    batch_bytes = 0
    for t in texts:
        n = len(t.encode("utf-8"))
        if batch and (len(batch) >= max_count or batch_bytes + n > max_bytes):
            batches.append(batch)
            batch, batch_bytes = [], 0
        batch.append(t)
        batch_bytes += n
    if batch:
        batches.append(batch)
    return batches


class QueryEmbedLRU:
    """Tiny LRU for single-QUERY embeddings (TTFT program, 2.50.0).

    Stage-2 retrieval embeds the latest user message before every main
    LLM call, and the same query text recurs — the retriever and the
    composite provider both search on it within one turn, and identical
    queries repeat across turns. Embeddings are deterministic for a
    given model, so a small text→vector map removes repeat HTTP
    round-trips from the TTFT-critical path.

    Scope: attach one instance per vector-store (the store owns exactly
    one embedding client, so the model is implicit in the key). Only
    the single-text SEARCH path should use it — document/batch indexing
    embeds novel text and would just churn the cache.

    Not coroutine-locked on purpose: two concurrent misses on the same
    text both embed and both store the identical vector — wasteful once,
    never wrong.
    """

    def __init__(self, maxsize: int = 64):
        from collections import OrderedDict

        self._max = max(1, int(maxsize))
        self._data: "OrderedDict[str, List[float]]" = OrderedDict()

    def get(self, text: str) -> Optional[List[float]]:
        vec = self._data.get(text)
        if vec is not None:
            self._data.move_to_end(text)
        return vec

    def put(self, text: str, vector: Sequence[float]) -> None:
        self._data[text] = list(vector)
        self._data.move_to_end(text)
        while len(self._data) > self._max:
            self._data.popitem(last=False)


@runtime_checkable
class EmbeddingClient(Protocol):
    """Asynchronous embedding backend.

    Implementations must be thread-safe at the method level (the
    VectorHandle may call `embed` from multiple coroutines). They
    should emit a `CostEvent` via the provided emitter (if any) for
    each billable API call so the memory subsystem can surface
    aggregate cost telemetry.
    """

    @property
    def descriptor(self) -> EmbeddingDescriptor:
        """Immutable snapshot of the active model. Used for dimension
        checks, reindex planning, and `MemoryDescriptor.embedding`.
        """
        ...

    async def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a batch of texts. Returns vectors in input order.

        Raises:
            `EmbeddingError` — transport failure, dimension mismatch,
            auth failure. The caller (VectorHandle / provider) is
            responsible for retry policy and should consult
            ``EmbeddingError.category`` to choose it.
        """
        ...

    async def close(self) -> None:
        """Release underlying connections/sessions. Optional."""
        ...


class EmbeddingError(RuntimeError):
    """Base error for embedding transport/validation failures.

    ``category`` classifies the failure for retry/breaker policy
    (see module docstring). Backends that can map typed SDK
    exceptions (openai) or HTTP status codes (voyage) set it;
    everything else defaults to ``'unknown'`` so unclassified
    failures keep their conservative handling.
    """

    def __init__(
        self,
        message: str,
        *,
        cost: CostEvent | None = None,
        category: str = "unknown",
    ) -> None:
        super().__init__(message)
        self.cost = cost
        self.category = category if category in EMBEDDING_ERROR_CATEGORIES else "unknown"


def category_for_http_status(status: int) -> str:
    """HTTP status → `EmbeddingError` category for bearer-token REST backends.

    Shared by every REST-shaped embedding client (voyage,
    openai_compatible): 401/403 means the credential is wrong — no
    amount of retrying fixes it, so it must count toward the vector
    layer's trip-once auth breaker. 429 is quota (retry later may
    work). 408/5xx are transient server-side conditions. Any other
    4xx (404 model-name typo, 400 payload issue, 422) will never
    succeed on retry — 'invalid' so the caller stops hot-retrying
    (audit D2). Everything else stays 'unknown' and keeps the
    conservative traceback-logging path.
    """
    if status in (401, 403):
        return "auth"
    if status == 429:
        return "quota"
    if status == 408 or status >= 500:
        return "transient"
    if 400 <= status < 500:
        return "invalid"
    return "unknown"


# ── deprecated env-var credential ladder ─────────────────────────────
#
# Embedding keys historically resolved from env vars *inside* the
# clients (OPENAI_API_KEY / VOYAGE_API_KEY / GOOGLE_API_KEY /
# GEMINI_API_KEY) — a parallel credential channel that made the
# CredentialBundle docstring's "single channel" claim false (audit
# §2.6). The supported path is now an explicit ``api_key=`` (sourced
# from the bundle's ``'embedding'`` entry by `MemoryProviderFactory`).
# The env ladder stays as a fallback so existing deployments keep
# working through 2.2.x, but it announces itself exactly once so
# operators learn about the migration without their logs getting
# spammed on every client construction.

_env_ladder_warned = False


def _resolve_env_api_key(provider: str, *env_vars: str) -> str:
    """DEPRECATED fallback: read an embedding API key from env vars.

    Returns the first non-empty value among ``env_vars`` (empty string
    when none is set). On the first successful env resolution in this
    process, logs a one-time deprecation warning pointing at the
    CredentialBundle ``'embedding'`` channel — the only supported
    credential path going forward.
    """
    global _env_ladder_warned
    for var in env_vars:
        value = os.environ.get(var, "")
        if value:
            if not _env_ladder_warned:
                _env_ladder_warned = True
                logger.warning(
                    "DEPRECATED: %s embedding API key resolved from env var %s. "
                    "Pass it explicitly via the CredentialBundle 'embedding' "
                    "provider entry (ProviderCredentials(api_key=...)) handed to "
                    "MemoryProviderFactory(credentials=...), or via the embedding "
                    "config's api_key field. The env-var ladder will be removed "
                    "in a future major release.",
                    provider,
                    var,
                )
            return value
    return ""


class _LoopBoundClientMixin:
    """Loop-safe caching for httpx-backed embedding SDK clients.

    An httpx-based SDK client (``AsyncOpenAI``, ``genai.Client``) binds its
    transport/connection-pool to the event loop that first drives it and
    cannot be reused from another loop — a later call on a different loop
    raises ``RuntimeError: Event loop is closed`` / "Future attached to a
    different loop". Hosts that drive memory writes through a sync→async
    bridge (Geny's ``run_coro_sync``) spin a fresh, short-lived event loop
    **per call**, so a single cached client would (a) fail cross-loop on
    every bridged embed and (b), never being closed, leak its socket pool
    once per session.

    Strategy:
      * Cache ONE client on the loop that first uses it — almost always the
        stable server loop, where the vast majority of embeds happen
        (search, the memory stage). That loop's calls reuse it, so
        connection pooling is preserved.
      * If the cached client's loop is found dead, drop the reference (its
        transport is finalized by GC) and rebind to the current loop. An
        all-bridge caller (only ever ephemeral loops) therefore rebinds
        each call instead of accumulating clients.
      * Any OTHER still-live loop gets a short-lived client that the caller
        closes within the same call (``ephemeral=True``) — never drive one
        client's transport from two loops.

    Subclasses set ``self._client`` / ``self._client_loop`` /
    ``self._injected_client`` in ``__init__`` and implement
    ``_build_client()``. A test-injected client (``_injected_client``) is
    used verbatim and never rebuilt or auto-closed — the caller owns it.
    """

    _client: Optional[object] = None
    _client_loop: Optional[asyncio.AbstractEventLoop] = None
    _injected_client: bool = False

    def _build_client(self) -> object:
        raise NotImplementedError

    def _acquire_client(self):
        """Return ``(client, ephemeral)`` bound to the CURRENT running loop.

        ``ephemeral`` clients MUST be closed by the caller (see
        ``_aclose_client``) before the coroutine returns.
        """
        if self._injected_client:
            return self._client, False
        loop = asyncio.get_running_loop()
        cached = self._client
        if cached is not None:
            if self._client_loop is loop and not loop.is_closed():
                return cached, False  # hot path: pooled client on its loop
            if self._client_loop is None or self._client_loop.is_closed():
                # Cached loop is dead — drop so GC finalizes its transport,
                # then rebind to this loop below.
                self._client = None
                self._client_loop = None
                cached = None
        if cached is None:
            self._client = self._build_client()
            self._client_loop = loop
            return self._client, False
        # Cache valid but bound to a different, still-live loop → ephemeral.
        return self._build_client(), True

    async def _aclose_client(self, client: object) -> None:
        closer = getattr(client, "close", None)
        if closer is None:
            return
        try:
            result = closer()
            if asyncio.iscoroutine(result):
                await result
        except Exception:  # noqa: BLE001 — close is best-effort
            logger.debug("embedding client close failed", exc_info=True)

    async def _close_cached_client(self) -> None:
        """Release the cached client (``close()`` surface). No-op for an
        injected client — its lifetime belongs to the caller."""
        client = self._client
        self._client = None
        self._client_loop = None
        if client is None or self._injected_client:
            return
        await self._aclose_client(client)


__all__ = [
    "EmbeddingClient",
    "EmbeddingError",
    "EMBEDDING_ERROR_CATEGORIES",
]
