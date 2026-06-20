"""Discord inbound/outbound adapter — Gateway WebSocket + REST.

Connects to the Discord Gateway (v10), IDENTIFYs with message intents,
heartbeats, and turns ``MESSAGE_CREATE`` events into inbound messages; replies
go back over the REST API. Uses the ``websockets`` lib for the gateway and
``httpx`` for REST — no discord SDK.

Bot setup: create an application + bot at https://discord.com/developers,
enable the **Message Content Intent** (privileged), invite the bot to your
server, and use the bot token here.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Sequence

from geny_executor.gateway.types import InboundMessage
from geny_executor.gateway.ws_base import HttpTransport, _QueuedWSAdapter

logger = logging.getLogger(__name__)

_GATEWAY_URL = "wss://gateway.discord.gg/?v=10&encoding=json"
_API = "https://discord.com/api/v10"

# Gateway intents. GUILD_MESSAGES (1<<9) + DIRECT_MESSAGES (1<<12) +
# MESSAGE_CONTENT (1<<15, privileged — enable it in the Dev Portal). Without
# MESSAGE_CONTENT the gateway delivers empty ``content``.
_DEFAULT_INTENTS = (1 << 9) | (1 << 12) | (1 << 15)


def parse_discord_message(data: Dict[str, Any]) -> Optional[InboundMessage]:
    """``MESSAGE_CREATE`` payload → InboundMessage, or ``None`` to ignore.

    Skips messages from bots (including this bot itself) and empty text.
    """
    if not isinstance(data, dict):
        return None
    author = data.get("author") or {}
    if author.get("bot"):
        return None
    content = data.get("content") or ""
    if not content:
        return None
    return InboundMessage(
        platform="discord",
        chat_id=str(data.get("channel_id", "")),
        text=content,
        message_id=str(data.get("id", "")),
        sender_id=str(author.get("id", "")),
        sender_name=author.get("username") or author.get("global_name") or "",
        raw=data,
    )


class DiscordGatewayAdapter(_QueuedWSAdapter):
    """Discord bot gateway adapter (WebSocket inbound, REST outbound)."""

    name = "discord"

    def __init__(
        self,
        *,
        token: str,
        allowed_channel_ids: Optional[Sequence[Any]] = None,
        intents: Optional[int] = None,
        idle_timeout: float = 25.0,
        reconnect_backoff: float = 5.0,
        http_transport: Optional[HttpTransport] = None,
    ) -> None:
        if not token:
            raise ValueError("discord gateway requires 'token'")
        super().__init__(
            idle_timeout=idle_timeout,
            reconnect_backoff=reconnect_backoff,
            http_transport=http_transport,
        )
        self._token = token
        self._allowed = {str(c) for c in (allowed_channel_ids or []) if str(c)}
        self._intents = int(intents) if intents is not None else _DEFAULT_INTENTS

    def allow(self, message: InboundMessage) -> bool:
        return not self._allowed or message.chat_id in self._allowed

    # ── gateway connection ─────────────────────────────────────────────
    async def _run_connection(self) -> None:
        import websockets

        seq_box: List[Optional[int]] = [None]
        async with websockets.connect(_GATEWAY_URL, max_size=2**21) as ws:
            hello = json.loads(await ws.recv())
            interval = float(hello.get("d", {}).get("heartbeat_interval", 41250)) / 1000.0
            await ws.send(
                json.dumps(
                    {
                        "op": 2,
                        "d": {
                            "token": self._token,
                            "intents": self._intents,
                            "properties": {
                                "os": "linux",
                                "browser": "geny-executor",
                                "device": "geny-executor",
                            },
                        },
                    }
                )
            )
            hb = asyncio.create_task(self._heartbeat(ws, interval, seq_box))
            try:
                async for raw in ws:
                    payload = json.loads(raw)
                    if payload.get("s") is not None:
                        seq_box[0] = payload["s"]
                    op = payload.get("op")
                    if op == 1:  # gateway asked for an immediate heartbeat
                        await ws.send(json.dumps({"op": 1, "d": seq_box[0]}))
                    elif op == 0 and payload.get("t") == "MESSAGE_CREATE":
                        await self._put(parse_discord_message(payload.get("d") or {}))
                    elif op in (7, 9):  # reconnect / invalid-session → drop + retry
                        logger.info("discord_gateway_reconnect op=%s", op)
                        break
            finally:
                hb.cancel()

    async def _heartbeat(self, ws: Any, interval: float, seq_box: List[Optional[int]]) -> None:
        while True:
            await asyncio.sleep(interval)
            await ws.send(json.dumps({"op": 1, "d": seq_box[0]}))

    # ── outbound (REST) ────────────────────────────────────────────────
    async def send(self, *, chat_id: str, text: str) -> dict:
        result = await self._http_post(
            f"{_API}/channels/{chat_id}/messages",
            json={"content": text[:2000]},
            headers={"Authorization": f"Bot {self._token}"},
        )
        status = result.get("_status")
        return {"platform": "discord", "ok": status in (200, 201), "channel_id": chat_id}


__all__ = ["DiscordGatewayAdapter", "parse_discord_message"]
