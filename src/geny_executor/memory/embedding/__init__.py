"""Embedding backends for the memory subsystem.

Conforms every concrete client to `EmbeddingClient` so the
`VectorHandle` implementation and `MemoryProvider` construction path
can treat all providers uniformly.

Four backends:
    - `local`  — deterministic SHA-256 hashing trick. Zero deps.
    - `openai` — `text-embedding-3-*` via `openai` SDK.
    - `voyage` — `voyage-3*` via REST + httpx.
    - `google` — `text-embedding-004` via `google-genai` SDK.

Factory::

    from geny_executor.memory.embedding import create_embedding_client
    client = create_embedding_client("openai", model="text-embedding-3-small")
"""

from geny_executor.memory.embedding.client import EmbeddingClient, EmbeddingError
from geny_executor.memory.embedding.local import LocalHashEmbeddingClient
from geny_executor.memory.embedding.registry import create_embedding_client

__all__ = [
    "EmbeddingClient",
    "EmbeddingError",
    "LocalHashEmbeddingClient",
    "create_embedding_client",
]
