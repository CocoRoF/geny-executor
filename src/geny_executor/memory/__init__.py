"""geny-executor — memory subsystem.

Two layers live here:

1. **`geny_executor.memory.provider` (Phase 1+, runtime path)** —
   the unified `MemoryProvider` Protocol plus its 7 layer handles
   and supporting domain dataclasses. Every concrete memory
   implementation, including `EphemeralMemoryProvider` in
   `geny_executor.memory.providers.ephemeral`, conforms to this
   contract. Stages 2 (Context) and 15 (Memory) consume the provider;
   they no longer talk to layer-specific strategies directly.

2. **Legacy adapter (`GenyMemoryRetriever` / `GenyMemoryStrategy` /
   `GenyPersistence`)** — duck-typed wrappers around Geny's
   `SessionMemoryManager`. These are kept *only* as validation
   fixtures for Phase 3 (C7 — adapter parity). They are NOT the
   operating path and should not be wired into new code.

Public alias::

    from geny_executor.memory import (
        MemoryProvider,
        EphemeralMemoryProvider,
        Layer, Capability, Scope, Importance,
    )
"""

# ── Phase 1+ unified contract ───────────────────────────────────────
from geny_executor.memory.provider import (
    BackendInfo,
    Capability,
    CostEvent,
    CostModel,
    CuratedHandle,
    EmbeddingDescriptor,
    ExecutionSummary,
    GlobalHandle,
    Importance,
    IndexHandle,
    Insight,
    Layer,
    LTMHandle,
    MemoryDescriptor,
    MemoryEvent,
    MemoryHooks,
    MemoryProvider,
    MemorySnapshot,
    Note,
    NoteDraft,
    NoteGraph,
    NoteMeta,
    NotePatch,
    NoteRef,
    NotesHandle,
    RecordReceipt,
    ReflectionContext,
    ReindexPlan,
    RetrievalQuery,
    RetrievalResult,
    Scope,
    STMHandle,
    Turn,
    VectorHandle,
)
from geny_executor.memory.embedding import (
    EmbeddingClient,
    EmbeddingError,
    LocalHashEmbeddingClient,
    create_embedding_client,
)
from geny_executor.memory.providers import (
    EphemeralMemoryProvider,
    FileMemoryProvider,
    SQLMemoryProvider,
)

# ── Legacy adapter (Phase 3 validation only) ────────────────────────
from geny_executor.memory.retriever import GenyMemoryRetriever
from geny_executor.memory.strategy import GenyMemoryStrategy
from geny_executor.memory.persistence import GenyPersistence
from geny_executor.memory.presets import GenyPresets

__all__ = [
    # contract
    "MemoryProvider",
    "MemoryDescriptor",
    "MemoryHooks",
    "MemoryEvent",
    "Layer",
    "Capability",
    "Scope",
    "Importance",
    "BackendInfo",
    "EmbeddingDescriptor",
    "CostEvent",
    "CostModel",
    "Note",
    "NoteMeta",
    "NoteDraft",
    "NotePatch",
    "NoteRef",
    "NoteGraph",
    "Turn",
    "ExecutionSummary",
    "RecordReceipt",
    "Insight",
    "ReflectionContext",
    "RetrievalQuery",
    "RetrievalResult",
    "MemorySnapshot",
    "ReindexPlan",
    "STMHandle",
    "LTMHandle",
    "NotesHandle",
    "VectorHandle",
    "CuratedHandle",
    "GlobalHandle",
    "IndexHandle",
    # providers
    "EphemeralMemoryProvider",
    "FileMemoryProvider",
    "SQLMemoryProvider",
    # embedding
    "EmbeddingClient",
    "EmbeddingError",
    "LocalHashEmbeddingClient",
    "create_embedding_client",
    # legacy adapter (validation fixture)
    "GenyMemoryRetriever",
    "GenyMemoryStrategy",
    "GenyPersistence",
    "GenyPresets",
]
