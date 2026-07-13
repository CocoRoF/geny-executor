"""Factory for `EmbeddingClient` backends.

Single entrypoint `create_embedding_client(provider, model, ...)` maps
a provider string to the right concrete client. Unknown providers
raise `ValueError`. Missing optional deps surface the original
`ImportError` from the backend module.

Provider string matches `EmbeddingDescriptor.provider`
(`openai` | `voyage` | `google` | `local` | `openai_compatible`) so the
same identifier flows from config → client → descriptor unchanged.

Hosts can additionally register their OWN backends at runtime with
:func:`register_embedding_provider` — the same host-extension seam the
tool registry offers. A registered builder is addressed by name through
the ordinary config path::

    register_embedding_provider("my-embedding-service", build_my_client)
    # …then, in any memory provider config:
    {"embedding": {"provider": "my-embedding-service", "model": "…"}}

which keeps provider configs serializable (no client-instance
injection needed) while the actual transport/credential logic stays
entirely in the host.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Optional, Tuple

from geny_executor.memory.embedding.client import EmbeddingClient


_SUPPORTED = frozenset({"openai", "voyage", "google", "local", "openai_compatible"})

# ── host-registered backends (runtime extension seam) ────────────────
#
# Builder signature mirrors the built-ins' construction surface:
#   builder(*, model=None, api_key=None, dimension=None, **options) -> EmbeddingClient
# The config's `options` dict is expanded into kwargs, exactly as for
# built-in backends.
_CUSTOM_BUILDERS: Dict[str, Callable[..., EmbeddingClient]] = {}
_CUSTOM_LOCK = threading.Lock()


def register_embedding_provider(
    name: str,
    builder: Callable[..., EmbeddingClient],
    *,
    replace: bool = False,
) -> None:
    """Register a host-provided embedding backend under ``name``.

    ``builder`` is called as ``builder(model=…, api_key=…, dimension=…,
    **options)`` whenever ``create_embedding_client(name, …)`` runs —
    including through ``MemoryProviderFactory`` config paths, so a
    registered backend is usable from plain serializable config.

    Built-in names cannot be shadowed (ValueError) — a host that wants
    different transport semantics registers its own name. Re-registering
    an existing custom name requires ``replace=True`` (guards against
    accidental double-registration; idempotent boot paths pass it
    deliberately).
    """
    key = (name or "").lower().strip()
    if not key:
        raise ValueError("embedding provider name must be a non-empty string")
    if key in _SUPPORTED:
        raise ValueError(
            f"cannot register embedding provider {name!r}: shadows a built-in "
            f"(built-ins: {sorted(_SUPPORTED)})"
        )
    if not callable(builder):
        raise TypeError("builder must be callable")
    with _CUSTOM_LOCK:
        if key in _CUSTOM_BUILDERS and not replace:
            raise ValueError(
                f"embedding provider {name!r} is already registered "
                "(pass replace=True to overwrite)"
            )
        _CUSTOM_BUILDERS[key] = builder


def unregister_embedding_provider(name: str) -> bool:
    """Remove a host-registered backend. Returns True when it existed."""
    key = (name or "").lower().strip()
    with _CUSTOM_LOCK:
        return _CUSTOM_BUILDERS.pop(key, None) is not None


def registered_embedding_providers() -> Tuple[str, ...]:
    """Names of host-registered backends (built-ins not included)."""
    with _CUSTOM_LOCK:
        return tuple(sorted(_CUSTOM_BUILDERS))


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
    `transport`) without bloating this signature. Host-registered
    backends (see :func:`register_embedding_provider`) resolve first;
    unknown provider names raise `ValueError`; missing optional SDKs
    raise `ImportError` from the backend module with install
    instructions.
    """
    p = provider.lower().strip()
    with _CUSTOM_LOCK:
        custom = _CUSTOM_BUILDERS.get(p)
    if custom is not None:
        return custom(
            model=model,
            api_key=api_key,
            dimension=dimension,
            **dict(options or {}),
        )
    if p not in _SUPPORTED:
        registered = registered_embedding_providers()
        hint = f"; host-registered: {list(registered)}" if registered else ""
        raise ValueError(
            f"unknown embedding provider {provider!r} "
            f"(supported: {sorted(_SUPPORTED)}{hint})"
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
    if p == "openai_compatible":
        from geny_executor.memory.embedding.openai_compatible import (
            OpenAICompatibleEmbeddingClient,
        )

        # `model` has no sensible library default here — served-model names
        # are deployment-specific; the client raises a clear ValueError when
        # it is missing. `base_url` arrives via `options`.
        return OpenAICompatibleEmbeddingClient(
            model=model or "",
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


__all__ = [
    "create_embedding_client",
    "register_embedding_provider",
    "registered_embedding_providers",
    "unregister_embedding_provider",
]
