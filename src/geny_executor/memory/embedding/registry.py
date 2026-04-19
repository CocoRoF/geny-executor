"""Factory for `EmbeddingClient` backends.

Single entrypoint `create_embedding_client(provider, model, ...)` maps
a provider string to the right concrete client. Unknown providers
raise `ValueError`. Missing optional deps surface the original
`ImportError` from the backend module.

Provider string matches `EmbeddingDescriptor.provider`
(`openai` | `voyage` | `google` | `local`) so the same identifier
flows from config → client → descriptor unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from geny_executor.memory.embedding.client import EmbeddingClient


_SUPPORTED = frozenset({"openai", "voyage", "google", "local"})


def create_embedding_client(
    provider: str,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    dimension: Optional[int] = None,
    options: Optional[Dict[str, Any]] = None,
) -> EmbeddingClient:
    """Construct an EmbeddingClient for `provider`.

    `options` forwards backend-specific kwargs (e.g. `base_url`,
    `transport`) without bloating this signature. Unknown provider
    names raise `ValueError`; missing optional SDKs raise `ImportError`
    from the backend module with install instructions.
    """
    p = provider.lower().strip()
    if p not in _SUPPORTED:
        raise ValueError(
            f"unknown embedding provider {provider!r} (supported: {sorted(_SUPPORTED)})"
        )
    opts = dict(options or {})
    if p == "local":
        from geny_executor.memory.embedding.local import LocalHashEmbeddingClient

        return LocalHashEmbeddingClient(
            model=model or "hash-v1",
            dimension=dimension or opts.pop("dimension", 384),
        )
    if p == "openai":
        from geny_executor.memory.embedding.openai import OpenAIEmbeddingClient

        return OpenAIEmbeddingClient(
            model=model or "text-embedding-3-small",
            api_key=api_key,
            dimension=dimension,
            **opts,
        )
    if p == "voyage":
        from geny_executor.memory.embedding.voyage import VoyageEmbeddingClient

        return VoyageEmbeddingClient(
            model=model or "voyage-3",
            api_key=api_key,
            dimension=dimension,
            **opts,
        )
    if p == "google":
        from geny_executor.memory.embedding.google import GoogleEmbeddingClient

        return GoogleEmbeddingClient(
            model=model or "text-embedding-004",
            api_key=api_key,
            dimension=dimension,
            **opts,
        )
    # unreachable — guarded by _SUPPORTED above
    raise ValueError(f"unroutable provider {provider!r}")


__all__ = ["create_embedding_client"]
