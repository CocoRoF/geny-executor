"""Factory + host-extension registry for STT providers.

Single entrypoint :func:`create_stt_client` maps a provider string to a
concrete :class:`~geny_executor.audio.stt.provider.STTProvider`.
Built-in: ``openai_compatible`` (also aliased as ``openai`` /
``whisper`` — they all speak ``/v1/audio/transcriptions``).

Hosts register their own engines at runtime with
:func:`register_stt_provider` — same seam as the embedding registry::

    register_stt_provider("my-stt-service", build_my_client)
    # …then, in ToolContext.extras:
    {"stt": {"provider": "my-stt-service", "model": "…", ...}}

which keeps the per-session config serializable while transport and
credential logic stay entirely in the host.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict

from geny_executor.audio.stt.provider import STTProvider

_BUILTIN_ALIASES = {
    "openai_compatible": "openai_compatible",
    "openai": "openai_compatible",
    "whisper": "openai_compatible",
}

_CUSTOM_BUILDERS: Dict[str, Callable[..., STTProvider]] = {}
_CUSTOM_LOCK = threading.Lock()


def register_stt_provider(
    name: str,
    builder: Callable[..., STTProvider],
    *,
    replace: bool = False,
) -> None:
    """Register a host-provided STT backend under ``name``.

    ``builder`` is called as ``builder(**config)`` whenever
    ``create_stt_client(name, **config)`` runs. Built-in names cannot be
    shadowed (ValueError); re-registering a custom name requires
    ``replace=True`` (idempotent boot paths pass it deliberately).
    """
    key = (name or "").lower().strip()
    if not key:
        raise ValueError("stt provider name must be a non-empty string")
    if key in _BUILTIN_ALIASES:
        raise ValueError(
            f"cannot register stt provider {name!r}: shadows a built-in "
            f"(built-ins: {sorted(set(_BUILTIN_ALIASES))})"
        )
    if not callable(builder):
        raise TypeError("builder must be callable")
    with _CUSTOM_LOCK:
        if key in _CUSTOM_BUILDERS and not replace:
            raise ValueError(
                f"stt provider {name!r} is already registered "
                "(pass replace=True to overwrite)"
            )
        _CUSTOM_BUILDERS[key] = builder


def unregister_stt_provider(name: str) -> bool:
    """Remove a host-registered backend. Returns True when it existed."""
    key = (name or "").lower().strip()
    with _CUSTOM_LOCK:
        return _CUSTOM_BUILDERS.pop(key, None) is not None


def create_stt_client(provider: str, **config: Any) -> STTProvider:
    """Build an :class:`STTProvider` for ``provider``.

    Unknown providers raise ``ValueError`` listing what IS available —
    the tool layer surfaces that verbatim so misconfiguration is
    self-explaining.
    """
    key = (provider or "").lower().strip() or "openai_compatible"

    with _CUSTOM_LOCK:
        custom = _CUSTOM_BUILDERS.get(key)
    if custom is not None:
        return custom(**config)

    if key in _BUILTIN_ALIASES:
        from geny_executor.audio.stt.openai_compatible import OpenAICompatibleSTT

        return OpenAICompatibleSTT(**config)

    with _CUSTOM_LOCK:
        available = sorted(set(_BUILTIN_ALIASES) | set(_CUSTOM_BUILDERS))
    raise ValueError(
        f"unknown stt provider {provider!r}; available: {available}"
    )
