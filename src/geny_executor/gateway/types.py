"""Gateway value types — inbound messages and replies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class InboundMessage:
    """One inbound chat message a :class:`PlatformAdapter` produced.

    ``chat_id`` is the stable conversation key the host maps to an agent
    session (a Telegram chat id, a Discord channel id, …). ``text`` is the
    user's message. ``raw`` keeps the untouched platform payload for hosts
    that need more than the normalized fields.
    """

    platform: str
    chat_id: str
    text: str
    message_id: str = ""
    sender_id: str = ""
    sender_name: str = ""
    attachments: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GatewayReply:
    """A reply to send back. A handler may return a bare ``str`` instead; the
    runner wraps it. ``chat_id`` overrides the inbound chat when set (rare)."""

    text: str
    chat_id: Optional[str] = None
    attachments: List[str] = field(default_factory=list)


__all__ = ["InboundMessage", "GatewayReply"]
