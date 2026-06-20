"""Build :class:`SendMessageChannel` instances + registries from config.

The host-facing seam for the built-in transports: a host (Geny, or any
consumer) declares channels as plain dicts — ``{"name", "kind", "config"}`` —
and the executor constructs + registers the right transport. The host owns
*which* channels exist and their secrets; the executor owns *how* each kind
talks to its service. This is what lets a host stop shipping channel code.

Example::

    registry = build_channel_registry([
        {"name": "ops", "kind": "slack",
         "config": {"webhook_url": "https://hooks.slack.com/…"}},
        {"name": "owner", "kind": "telegram",
         "config": {"token": "123:abc", "chat_id": "456"}},
    ])
    # → pass into ToolContext.extras["send_message_channels"]; the agent's
    #   SendMessage tool dispatches to "ops" / "owner" by name.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

from geny_executor.channels.built_in import (
    DiscordSendMessageChannel,
    NtfySendMessageChannel,
    SlackSendMessageChannel,
    TelegramSendMessageChannel,
    WebhookSendMessageChannel,
)
from geny_executor.channels.send_message_channel import (
    SendMessageChannel,
    SendMessageChannelRegistry,
    StdoutSendMessageChannel,
)

logger = logging.getLogger(__name__)


_BUILDERS: Dict[str, Callable[[Dict[str, Any]], SendMessageChannel]] = {
    "stdout": lambda _cfg: StdoutSendMessageChannel(),
    "webhook": lambda cfg: WebhookSendMessageChannel(**cfg),
    "telegram": lambda cfg: TelegramSendMessageChannel(**cfg),
    "discord": lambda cfg: DiscordSendMessageChannel(**cfg),
    "slack": lambda cfg: SlackSendMessageChannel(**cfg),
    "ntfy": lambda cfg: NtfySendMessageChannel(**cfg),
}

#: Channel ``kind`` strings the executor builds out of the box.
BUILTIN_CHANNEL_KINDS: Tuple[str, ...] = tuple(sorted(_BUILDERS))


def build_send_message_channel(
    kind: str, config: Optional[Mapping[str, Any]] = None
) -> SendMessageChannel:
    """Construct one channel for *kind* from its *config* mapping.

    Raises ``ValueError`` for an unknown kind, or whatever the transport's
    constructor raises for missing/invalid config (e.g. a telegram channel with
    no ``token``).
    """
    builder = _BUILDERS.get(kind)
    if builder is None:
        raise ValueError(
            f"unknown channel kind {kind!r}; known: {', '.join(BUILTIN_CHANNEL_KINDS)}"
        )
    return builder(dict(config or {}))


def build_channel_registry(
    specs: Optional[Iterable[Mapping[str, Any]]],
    *,
    registry: Optional[SendMessageChannelRegistry] = None,
) -> SendMessageChannelRegistry:
    """Build (or extend) a registry from a list of channel specs.

    Each spec is ``{"name": str, "kind": str, "config": dict}``. Lenient by
    design — a malformed or unbuildable entry is logged and skipped so one bad
    channel config can't take down the whole agent. Returns the registry so it
    can be dropped straight into ``ToolContext.extras["send_message_channels"]``.
    """
    reg = registry if registry is not None else SendMessageChannelRegistry()
    for spec in specs or []:
        if not isinstance(spec, Mapping):
            logger.warning("channel_spec_not_mapping spec=%r", spec)
            continue
        name = spec.get("name")
        kind = spec.get("kind")
        if not name or not kind:
            logger.warning("channel_spec_missing_name_or_kind spec=%r", spec)
            continue
        config = spec.get("config") or {}
        if not isinstance(config, Mapping):
            config = {}
        try:
            channel = build_send_message_channel(str(kind), config)
        except Exception as exc:  # noqa: BLE001 — one bad channel must not abort
            logger.warning("channel_build_failed name=%s kind=%s err=%s", name, kind, exc)
            continue
        reg.register(str(name), channel)
    return reg


__all__ = [
    "BUILTIN_CHANNEL_KINDS",
    "build_send_message_channel",
    "build_channel_registry",
]
