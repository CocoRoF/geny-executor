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

import logging
import os
from typing import List, Protocol, Sequence, runtime_checkable

from geny_executor.memory.provider import CostEvent, EmbeddingDescriptor

logger = logging.getLogger(__name__)


#: Valid values for ``EmbeddingError.category``.
EMBEDDING_ERROR_CATEGORIES = frozenset({"auth", "quota", "transient", "unknown"})


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


__all__ = [
    "EmbeddingClient",
    "EmbeddingError",
    "EMBEDDING_ERROR_CATEGORIES",
]
