"""Built-in :class:`SendMessageChannel` transports.

Historically the framework shipped only the ABC + ``StdoutSendMessageChannel``
and left every real transport (Telegram / Discord / Slack / …) to the host.
That pushed the same boilerplate into every host. These built-ins flip that:
the executor now *owns* the common output channels so a host only supplies
config (a token, a webhook URL) and the agent's ``SendMessage`` tool just works.

Every transport here is a plain HTTP POST, so they need no vendor SDKs — only
``httpx`` (already a base dependency). Each ``send`` is best-effort and returns
a small status dict; an injectable ``transport`` hook keeps them testable
without a live network (mirrors ``llm_client.local_probe``).
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional

from geny_executor.channels.send_message_channel import SendMessageChannel

logger = logging.getLogger(__name__)

#: ``async def transport(url, *, json, data, headers, params) -> dict`` — test
#: hook returning ``{"status": int, "ok": bool, "body": str}`` without HTTP.
Transport = Callable[..., Awaitable[Dict[str, Any]]]

_DEFAULT_TIMEOUT_S = 15.0


def _append_attachments(message: str, attachments: Optional[List[str]]) -> str:
    """Most chat webhooks take text only; surface attachment URLs inline so
    they aren't silently dropped."""
    if not attachments:
        return message
    links = "\n".join(str(a) for a in attachments if a)
    return f"{message}\n{links}" if links else message


class _HttpSendMessageChannel(SendMessageChannel):
    """Shared HTTP POST plumbing for the built-in transports."""

    kind: str = "http"

    def __init__(
        self,
        *,
        timeout: float = _DEFAULT_TIMEOUT_S,
        transport: Optional[Transport] = None,
    ) -> None:
        self._timeout = timeout
        self._transport = transport

    async def _post(
        self,
        url: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        headers: Optional[Mapping[str, str]] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self._transport is not None:
            return await self._transport(url, json=json, data=data, headers=headers, params=params)
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - httpx is a base dep
            raise RuntimeError("httpx is required for built-in channels") from exc
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                url,
                json=json,
                content=data,
                headers=dict(headers) if headers else None,
                params=dict(params) if params else None,
            )
        ok = 200 <= resp.status_code < 300
        if not ok:
            logger.warning(
                "send_message_channel_http_error kind=%s status=%s",
                self.kind,
                resp.status_code,
            )
        return {"status": resp.status_code, "ok": ok, "body": resp.text[:500]}

    def _result(self, posted: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
        return {
            "channel": self.kind,
            "delivered": bool(posted.get("ok")),
            "status": posted.get("status"),
            **extra,
        }


class WebhookSendMessageChannel(_HttpSendMessageChannel):
    """Generic webhook — POSTs ``{to, message, attachments}`` as JSON to a URL.

    The escape hatch for any HTTP endpoint not covered by a typed transport.
    """

    kind = "webhook"

    def __init__(
        self,
        *,
        url: str,
        headers: Optional[Mapping[str, str]] = None,
        timeout: float = _DEFAULT_TIMEOUT_S,
        transport: Optional[Transport] = None,
    ) -> None:
        if not url:
            raise ValueError("webhook channel requires 'url'")
        super().__init__(timeout=timeout, transport=transport)
        self._url = url
        self._headers = dict(headers) if headers else None

    async def send(self, *, to=None, message, attachments=None):
        posted = await self._post(
            self._url,
            json={"to": to, "message": message, "attachments": list(attachments or [])},
            headers=self._headers,
        )
        return self._result(posted)


class TelegramSendMessageChannel(_HttpSendMessageChannel):
    """Telegram Bot API ``sendMessage``. ``to`` (or the default chat id) is the
    target chat. Requires a bot token from @BotFather."""

    kind = "telegram"

    def __init__(
        self,
        *,
        token: str,
        chat_id: Optional[str] = None,
        parse_mode: Optional[str] = None,
        timeout: float = _DEFAULT_TIMEOUT_S,
        transport: Optional[Transport] = None,
    ) -> None:
        if not token:
            raise ValueError("telegram channel requires 'token'")
        super().__init__(timeout=timeout, transport=transport)
        self._token = token
        self._chat_id = chat_id
        self._parse_mode = parse_mode

    async def send(self, *, to=None, message, attachments=None):
        chat_id = to or self._chat_id
        if not chat_id:
            raise ValueError("telegram send needs a chat id (via 'to' or channel config)")
        body: Dict[str, Any] = {
            "chat_id": chat_id,
            "text": _append_attachments(message, attachments),
        }
        if self._parse_mode:
            body["parse_mode"] = self._parse_mode
        posted = await self._post(
            f"https://api.telegram.org/bot{self._token}/sendMessage", json=body
        )
        return self._result(posted, chat_id=str(chat_id))


class DiscordSendMessageChannel(_HttpSendMessageChannel):
    """Discord incoming webhook — POSTs ``{content}`` to a channel webhook URL.
    ``to`` optionally overrides the webhook's display name."""

    kind = "discord"

    def __init__(
        self,
        *,
        webhook_url: str,
        username: Optional[str] = None,
        timeout: float = _DEFAULT_TIMEOUT_S,
        transport: Optional[Transport] = None,
    ) -> None:
        if not webhook_url:
            raise ValueError("discord channel requires 'webhook_url'")
        super().__init__(timeout=timeout, transport=transport)
        self._webhook_url = webhook_url
        self._username = username

    async def send(self, *, to=None, message, attachments=None):
        # Discord hard-caps content at 2000 chars.
        content = _append_attachments(message, attachments)[:2000]
        body: Dict[str, Any] = {"content": content}
        name = to or self._username
        if name:
            body["username"] = name
        posted = await self._post(self._webhook_url, json=body)
        return self._result(posted)


class SlackSendMessageChannel(_HttpSendMessageChannel):
    """Slack incoming webhook — POSTs ``{text}`` to a workspace webhook URL."""

    kind = "slack"

    def __init__(
        self,
        *,
        webhook_url: str,
        timeout: float = _DEFAULT_TIMEOUT_S,
        transport: Optional[Transport] = None,
    ) -> None:
        if not webhook_url:
            raise ValueError("slack channel requires 'webhook_url'")
        super().__init__(timeout=timeout, transport=transport)
        self._webhook_url = webhook_url

    async def send(self, *, to=None, message, attachments=None):
        text = _append_attachments(message, attachments)
        if to:
            text = f"<{to}> {text}" if to.startswith("@") or to.startswith("#") else text
        posted = await self._post(self._webhook_url, json={"text": text})
        return self._result(posted)


class NtfySendMessageChannel(_HttpSendMessageChannel):
    """ntfy.sh push — POSTs the message body to ``{server}/{topic}``.
    ``to`` overrides the topic; optional bearer ``token`` for protected topics."""

    kind = "ntfy"

    def __init__(
        self,
        *,
        topic: str,
        server: str = "https://ntfy.sh",
        token: Optional[str] = None,
        title: Optional[str] = None,
        timeout: float = _DEFAULT_TIMEOUT_S,
        transport: Optional[Transport] = None,
    ) -> None:
        if not topic:
            raise ValueError("ntfy channel requires 'topic'")
        super().__init__(timeout=timeout, transport=transport)
        self._topic = topic
        self._server = server.rstrip("/")
        self._token = token
        self._title = title

    async def send(self, *, to=None, message, attachments=None):
        topic = to or self._topic
        headers: Dict[str, str] = {}
        if self._title:
            headers["Title"] = self._title
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        atts = [a for a in (attachments or []) if a]
        if atts:
            headers["Attach"] = atts[0]
        posted = await self._post(
            f"{self._server}/{topic}",
            data=message.encode("utf-8"),
            headers=headers or None,
        )
        return self._result(posted, topic=str(topic))


__all__ = [
    "Transport",
    "WebhookSendMessageChannel",
    "TelegramSendMessageChannel",
    "DiscordSendMessageChannel",
    "SlackSendMessageChannel",
    "NtfySendMessageChannel",
]
