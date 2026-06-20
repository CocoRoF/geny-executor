"""Telegram inbound/outbound adapter — Bot API long-polling.

Pure HTTP over ``httpx`` (a base dep): no telegram SDK. ``fetch`` long-polls
``getUpdates`` (advancing the update offset so each message is seen once);
``send`` posts ``sendMessage``. Create a bot + token with @BotFather, then the
gateway turns "DM the bot" into "run an agent turn and reply".
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from geny_executor.gateway.adapter import PlatformAdapter
from geny_executor.gateway.types import InboundMessage

logger = logging.getLogger(__name__)

#: ``async def transport(method, url, *, params, json, timeout) -> dict`` —
#: test hook returning the parsed Telegram JSON body without real HTTP.
Transport = Callable[..., Awaitable[Dict[str, Any]]]

_API = "https://api.telegram.org"


class TelegramGatewayAdapter(PlatformAdapter):
    """Telegram bot gateway adapter (long-poll)."""

    name = "telegram"

    def __init__(
        self,
        *,
        token: str,
        allowed_chat_ids: Optional[Sequence[Any]] = None,
        poll_timeout: int = 25,
        parse_mode: Optional[str] = None,
        transport: Optional[Transport] = None,
    ) -> None:
        if not token:
            raise ValueError("telegram gateway requires 'token'")
        self._token = token
        # Allow-list: empty/None ⇒ open to every chat. Stored as strings so a
        # config that lists numeric ids still matches the stringified chat_id.
        self._allowed = {str(c) for c in (allowed_chat_ids or []) if str(c)}
        self._poll_timeout = max(0, int(poll_timeout))
        self._parse_mode = parse_mode
        self._transport = transport
        self._offset = 0

    # ── transport ──────────────────────────────────────────────────────
    async def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: float,
    ) -> Dict[str, Any]:
        url = f"{_API}/bot{self._token}/{endpoint}"
        if self._transport is not None:
            return await self._transport(method, url, params=params, json=json, timeout=timeout)
        import httpx

        async with httpx.AsyncClient(timeout=timeout) as client:
            if method == "GET":
                resp = await client.get(url, params=params)
            else:
                resp = await client.post(url, json=json)
        try:
            return resp.json()
        except ValueError:
            return {"ok": False, "error": f"non-json HTTP {resp.status_code}"}

    # ── inbound ────────────────────────────────────────────────────────
    async def fetch(self) -> List[InboundMessage]:
        # Read timeout must exceed the long-poll window or httpx aborts mid-poll.
        body = await self._request(
            "GET",
            "getUpdates",
            params={"offset": self._offset, "timeout": self._poll_timeout},
            timeout=self._poll_timeout + 15,
        )
        if not body.get("ok"):
            logger.debug("telegram_getupdates_not_ok body=%s", body)
            return []
        messages: List[InboundMessage] = []
        for update in body.get("result", []):
            uid = update.get("update_id")
            if isinstance(uid, int):
                self._offset = max(self._offset, uid + 1)
            msg = update.get("message") or update.get("edited_message")
            if not isinstance(msg, dict):
                continue
            text = msg.get("text")
            if not text:
                continue  # skip non-text (stickers, photos, service messages)
            chat = msg.get("chat") or {}
            sender = msg.get("from") or {}
            messages.append(
                InboundMessage(
                    platform="telegram",
                    chat_id=str(chat.get("id", "")),
                    text=text,
                    message_id=str(msg.get("message_id", "")),
                    sender_id=str(sender.get("id", "")),
                    sender_name=sender.get("first_name") or sender.get("username") or "",
                    raw=update,
                )
            )
        return messages

    def allow(self, message: InboundMessage) -> bool:
        return not self._allowed or message.chat_id in self._allowed

    # ── outbound ───────────────────────────────────────────────────────
    async def send(self, *, chat_id: str, text: str) -> dict:
        body: Dict[str, Any] = {"chat_id": chat_id, "text": text}
        if self._parse_mode:
            body["parse_mode"] = self._parse_mode
        result = await self._request("POST", "sendMessage", json=body, timeout=20.0)
        return {"platform": "telegram", "ok": bool(result.get("ok")), "chat_id": chat_id}


__all__ = ["TelegramGatewayAdapter"]
