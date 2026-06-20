"""PlatformAdapter ABC — one inbound/outbound chat platform connection.

An adapter owns the *transport* for a platform: how to fetch new inbound
messages and how to send a reply. It owns no agent logic — the
:class:`~geny_executor.gateway.runner.GatewayRunner` drives the loop and calls
the host's handler. Keeping adapters transport-only is what lets the executor
ship them built-in while the host stays "just a user".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from geny_executor.gateway.types import InboundMessage


class PlatformAdapter(ABC):
    """A chat platform the gateway can receive from and reply to."""

    #: Stable platform id (``"telegram"`` …). Set by each concrete adapter.
    name: str = ""

    @abstractmethod
    async def fetch(self) -> List[InboundMessage]:
        """Return the next batch of inbound messages.

        Implementations may block up to their long-poll/timeout window and
        return ``[]`` when nothing arrived. They MUST advance their own read
        cursor so the same message isn't returned twice. Raising is fine — the
        runner logs it and retries after a backoff.
        """

    @abstractmethod
    async def send(self, *, chat_id: str, text: str) -> dict:
        """Send *text* to *chat_id*. Returns a small status dict."""

    def allow(self, message: InboundMessage) -> bool:
        """Allow-list hook. Default allows everything; adapters with an
        ``allowed_chat_ids`` config override to gate unknown chats."""
        return True

    async def close(self) -> None:
        """Release any resources. Default no-op."""
        return None


__all__ = ["PlatformAdapter"]
