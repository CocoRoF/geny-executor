"""Build a :class:`GatewayRunner` from config — the host-facing seam.

A host (Geny, or any consumer) declares gateway platforms as plain dicts —
``{"platform": "telegram", "config": {...}}`` — and supplies a handler. The
executor constructs the adapters and the runner. The host ships no transport
code, only config + "message in → reply text out".

Example::

    runner = build_gateway(
        [{"platform": "telegram", "config": {"token": "123:abc"}}],
        handler=my_handler,
    )
    await runner.start()
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

from geny_executor.gateway.adapter import PlatformAdapter
from geny_executor.gateway.runner import GatewayHandler, GatewayRunner
from geny_executor.gateway.telegram import TelegramGatewayAdapter

logger = logging.getLogger(__name__)


_ADAPTER_BUILDERS: Dict[str, Callable[[Dict[str, Any]], PlatformAdapter]] = {
    "telegram": lambda cfg: TelegramGatewayAdapter(**cfg),
}

#: Platform ids the executor builds out of the box. (Telegram is HTTP
#: long-poll — no SDK, no public endpoint. WebSocket platforms such as Discord
#: slot in here as adapters are added.)
BUILTIN_GATEWAY_PLATFORMS: Tuple[str, ...] = tuple(sorted(_ADAPTER_BUILDERS))


def build_platform_adapter(
    platform: str, config: Optional[Mapping[str, Any]] = None
) -> PlatformAdapter:
    """Construct one adapter for *platform* from its *config*.

    Raises ``ValueError`` for an unknown platform, or whatever the adapter's
    constructor raises for missing config (e.g. a telegram adapter with no
    ``token``).
    """
    builder = _ADAPTER_BUILDERS.get(platform)
    if builder is None:
        raise ValueError(
            f"unknown gateway platform {platform!r}; known: {', '.join(BUILTIN_GATEWAY_PLATFORMS)}"
        )
    return builder(dict(config or {}))


def build_gateway(
    specs: Optional[Iterable[Mapping[str, Any]]],
    handler: GatewayHandler,
    *,
    error_backoff_seconds: float = 5.0,
    max_concurrent_turns: int = 8,
) -> GatewayRunner:
    """Build a runner from platform specs + a handler.

    Each spec is ``{"platform": str, "config": dict}``. Lenient: a malformed or
    unbuildable entry is logged and skipped so one bad platform config can't
    stop the rest. Returns a (not-yet-started) :class:`GatewayRunner`; call
    ``await runner.start()``.
    """
    adapters = []
    for spec in specs or []:
        if not isinstance(spec, Mapping):
            logger.warning("gateway_spec_not_mapping spec=%r", spec)
            continue
        platform = spec.get("platform")
        if not platform:
            logger.warning("gateway_spec_missing_platform spec=%r", spec)
            continue
        config = spec.get("config") or {}
        if not isinstance(config, Mapping):
            config = {}
        try:
            adapters.append(build_platform_adapter(str(platform), config))
        except Exception as exc:  # noqa: BLE001 — one bad platform must not abort
            logger.warning("gateway_adapter_build_failed platform=%s err=%s", platform, exc)
            continue
    return GatewayRunner(
        adapters,
        handler,
        error_backoff_seconds=error_backoff_seconds,
        max_concurrent_turns=max_concurrent_turns,
    )


__all__ = [
    "BUILTIN_GATEWAY_PLATFORMS",
    "build_platform_adapter",
    "build_gateway",
]
