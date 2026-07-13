"""Embedding backends for the memory subsystem.

Conforms every concrete client to `EmbeddingClient` so the
`VectorHandle` implementation and `MemoryProvider` construction path
can treat all providers uniformly.

Five built-in backends:
    - `local`  — deterministic SHA-256 hashing trick. Zero deps.
    - `openai` — `text-embedding-3-*` via `openai` SDK.
    - `voyage` — `voyage-3*` via REST + httpx.
    - `google` — `text-embedding-004` via `google-genai` SDK.
    - `openai_compatible` — any self-hosted `/v1/embeddings` endpoint
      (vLLM / Ollama / LM Studio / TEI / proxies) via REST + httpx.
      `base_url` first-class, API key optional.

Plus a host-extension seam: `register_embedding_provider(name, builder)`
lets an embedding backend the LIBRARY knows nothing about (a host's own
embedding microservice, a proprietary gateway) resolve through the same
serializable config path.

Factory::

    from geny_executor.memory.embedding import create_embedding_client
    client = create_embedding_client("openai", model="text-embedding-3-small")
    local = create_embedding_client(
        "openai_compatible", model="bge-m3",
        options={"base_url": "http://vllm-host:8000/v1"},
    )
"""

from geny_executor.memory.embedding.client import (
    EMBEDDING_ERROR_CATEGORIES,
    EmbeddingClient,
    EmbeddingError,
    category_for_http_status,
)
from geny_executor.memory.embedding.local import LocalHashEmbeddingClient
from geny_executor.memory.embedding.openai_compatible import (
    OpenAICompatibleEmbeddingClient,
)
from geny_executor.memory.embedding.registry import (
    create_embedding_client,
    register_embedding_provider,
    registered_embedding_providers,
    unregister_embedding_provider,
)

__all__ = [
    "EMBEDDING_ERROR_CATEGORIES",
    "EmbeddingClient",
    "EmbeddingError",
    "LocalHashEmbeddingClient",
    "OpenAICompatibleEmbeddingClient",
    "category_for_http_status",
    "create_embedding_client",
    "register_embedding_provider",
    "registered_embedding_providers",
    "unregister_embedding_provider",
]
