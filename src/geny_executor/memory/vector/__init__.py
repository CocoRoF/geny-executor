"""External vector backends implementing the ``VectorHandle`` protocol.

The file/sql providers ship a pure-Python cosine store sized for small
per-session vaults; this package holds real ANN backends a host can
inject (``FileMemoryProvider(vector_store=...)``) when a vault becomes a
knowledge repository (thousands of chunks, multi-document payloads).
"""

from geny_executor.memory.vector.qdrant_store import (
    DocumentChunk,
    QdrantVectorStore,
)

__all__ = ["DocumentChunk", "QdrantVectorStore"]
