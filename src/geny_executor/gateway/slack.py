"""Slack inbound/outbound adapter — Socket Mode WebSocket + Web API.

Socket Mode needs no public endpoint: open a WS via ``apps.connections.open``
(app-level token ``xapp-…``), receive Events API envelopes, ACK each, and turn
``message`` events into inbound messages; replies go via ``chat.postMessage``
(bot token ``xoxb-…``). Uses ``websockets`` + ``httpx`` — no slack SDK.

Setup: create a Slack app, enable **Socket Mode** + the ``message.channels`` /
``message.im`` event subscriptions and the ``chat:write`` scope. Use the
app-level token (``app_token``) for the socket and the bot token (``bot_token``)
for replies.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Sequence

from geny_executor.gateway.types import InboundMessage
from geny_executor.gateway.ws_base import HttpTransport, _QueuedWSAdapter

logger = logging.getLogger(__name__)

_CONNECTIONS_OPEN = "https://slack.com/api/apps.connections.open"
_POST_MESSAGE = "https://slack.com/api/chat.postMessage"


def parse_slack_event(event: Dict[str, Any]) -> Optional[InboundMessage]:
    """A Slack ``event`` object → InboundMessage, or ``None`` to ignore.

    Keeps only plain user ``message`` events — drops bot messages
    (``bot_id``) and edits/joins/etc (any ``subtype``).
    """
    if not isinstance(event, dict):
        return None
    if event.get("type") != "message":
        return None
    if event.get("bot_id") or event.get("subtype"):
        return None
    text = event.get("text") or ""
    if not text:
        return None
    return InboundMessage(
        platform="slack",
        chat_id=str(event.get("channel", "")),
        text=text,
        message_id=str(event.get("ts", "")),
        sender_id=str(event.get("user", "")),
        raw=event,
    )


class SlackGatewayAdapter(_QueuedWSAdapter):
    """Slack Socket Mode adapter (WebSocket inbound, Web API outbound)."""

    name = "slack"

    def __init__(
        self,
        *,
        app_token: str,
        bot_token: str,
        allowed_channel_ids: Optional[Sequence[Any]] = None,
        idle_timeout: float = 25.0,
        reconnect_backoff: float = 5.0,
        http_transport: Optional[HttpTransport] = None,
    ) -> None:
        if not app_token:
            raise ValueError("slack gateway requires 'app_token' (xapp-…)")
        if not bot_token:
            raise ValueError("slack gateway requires 'bot_token' (xoxb-…)")
        super().__init__(
            idle_timeout=idle_timeout,
            reconnect_backoff=reconnect_backoff,
            http_transport=http_transport,
        )
        self._app_token = app_token
        self._bot_token = bot_token
        self._allowed = {str(c) for c in (allowed_channel_ids or []) if str(c)}

    def allow(self, message: InboundMessage) -> bool:
        return not self._allowed or message.chat_id in self._allowed

    # ── socket-mode connection ─────────────────────────────────────────
    async def _run_connection(self) -> None:
        opened = await self._http_post(
            _CONNECTIONS_OPEN, headers={"Authorization": f"Bearer {self._app_token}"}
        )
        url = opened.get("url")
        if not opened.get("ok") or not url:
            raise RuntimeError(f"slack apps.connections.open failed: {opened}")

        import websockets

        async with websockets.connect(url, max_size=2**21) as ws:
            async for raw in ws:
                envelope = json.loads(raw)
                etype = envelope.get("type")
                if etype == "hello":
                    continue
                if etype == "disconnect":
                    logger.info("slack_socket_disconnect reason=%s", envelope.get("reason"))
                    break
                # ACK every envelope that carries an id (required within 3s).
                env_id = envelope.get("envelope_id")
                if env_id:
                    await ws.send(json.dumps({"envelope_id": env_id}))
                if etype == "events_api":
                    event = (envelope.get("payload") or {}).get("event") or {}
                    await self._put(parse_slack_event(event))

    # ── outbound (Web API) ─────────────────────────────────────────────
    async def send(self, *, chat_id: str, text: str) -> dict:
        result = await self._http_post(
            _POST_MESSAGE,
            json={"channel": chat_id, "text": text},
            headers={"Authorization": f"Bearer {self._bot_token}"},
        )
        return {"platform": "slack", "ok": bool(result.get("ok")), "channel": chat_id}


__all__ = ["SlackGatewayAdapter", "parse_slack_event"]
