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
"""

from __future__ import annotations

from typing import List, Protocol, Sequence, runtime_checkable

from geny_executor.memory.provider import CostEvent, EmbeddingDescriptor


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
            responsible for retry policy.
        """
        ...

    async def close(self) -> None:
        """Release underlying connections/sessions. Optional."""
        ...


class EmbeddingError(RuntimeError):
    """Base error for embedding transport/validation failures."""

    def __init__(self, message: str, *, cost: CostEvent | None = None) -> None:
        super().__init__(message)
        self.cost = cost


__all__ = ["EmbeddingClient", "EmbeddingError"]
