"""Delivery channels — files / messages / events.

The framework ships the ABCs *and*, as of 2.10.0, built-in HTTP transports
(webhook / telegram / discord / slack / ntfy) so a host only supplies config.
Build them from config dicts with :func:`build_channel_registry`; the agent's
``SendMessage`` tool dispatches to them by name. Every transport is a plain
HTTP POST over ``httpx`` — no vendor SDKs, no extra deps.
"""

from geny_executor.channels.built_in import (
    DiscordSendMessageChannel,
    NtfySendMessageChannel,
    SlackSendMessageChannel,
    TelegramSendMessageChannel,
    WebhookSendMessageChannel,
)
from geny_executor.channels.factory import (
    BUILTIN_CHANNEL_KINDS,
    build_channel_registry,
    build_send_message_channel,
)
from geny_executor.channels.send_message_channel import (
    SendMessageChannel,
    SendMessageChannelRegistry,
    StdoutSendMessageChannel,
)
from geny_executor.channels.user_file_channel import UserFileChannel

__all__ = [
    "SendMessageChannel",
    "SendMessageChannelRegistry",
    "StdoutSendMessageChannel",
    "UserFileChannel",
    # built-in transports (2.10.0)
    "WebhookSendMessageChannel",
    "TelegramSendMessageChannel",
    "DiscordSendMessageChannel",
    "SlackSendMessageChannel",
    "NtfySendMessageChannel",
    "BUILTIN_CHANNEL_KINDS",
    "build_send_message_channel",
    "build_channel_registry",
]
