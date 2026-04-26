"""SendMessageChannel ABC + registry — multi-channel messaging."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SendMessageChannel(ABC):
    """One messaging backend (Discord / Slack / SMS / internal DM).
    Hosts implement and register; the framework's SendMessageTool
    dispatches by channel name.
    """

    @abstractmethod
    async def send(
        self,
        *,
        to: Optional[str] = None,
        message: str,
        attachments: Optional[List[str]] = None,
    ) -> Dict[str, Any]: ...


class SendMessageChannelRegistry:
    def __init__(self) -> None:
        self._channels: Dict[str, SendMessageChannel] = {}

    def register(self, name: str, channel: SendMessageChannel) -> None:
        if name in self._channels:
            logger.warning("send_message_channel_overwritten", extra={"channel_name": name})
        self._channels[name] = channel

    def get(self, name: str) -> Optional[SendMessageChannel]:
        return self._channels.get(name)

    def list(self) -> List[str]:
        return sorted(self._channels.keys())


class StdoutSendMessageChannel(SendMessageChannel):
    """Reference impl that just logs. Useful for tests + dev."""

    async def send(self, *, to=None, message, attachments=None):
        logger.info(
            "stdout_send_message",
            extra={"to": to, "message": message, "attachments": list(attachments or [])},
        )
        return {"channel": "stdout", "delivered": True}


__all__ = [
    "SendMessageChannel",
    "SendMessageChannelRegistry",
    "StdoutSendMessageChannel",
]
