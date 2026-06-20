"""WebSocket gateway adapters — Discord + Slack (2.12.0)."""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor.gateway import (
    BUILTIN_GATEWAY_PLATFORMS,
    DiscordGatewayAdapter,
    InboundMessage,
    SlackGatewayAdapter,
    build_platform_adapter,
)
from geny_executor.gateway.discord import parse_discord_message
from geny_executor.gateway.slack import parse_slack_event


def _http_recorder(reply):
    calls = {}

    async def transport(url, *, json=None, headers=None):
        calls.update(url=url, json=json, headers=headers)
        return reply

    return transport, calls


def run(coro):
    return asyncio.run(coro)


# ── Discord ────────────────────────────────────────────────────────────


def test_parse_discord_message():
    m = parse_discord_message(
        {"id": "9", "channel_id": "555", "content": "hi",
         "author": {"id": "7", "username": "alice", "bot": False}}
    )
    assert m and m.platform == "discord" and m.chat_id == "555" and m.text == "hi"
    assert m.sender_name == "alice"


def test_parse_discord_ignores_bots_and_empty():
    assert parse_discord_message({"content": "x", "author": {"bot": True}}) is None
    assert parse_discord_message({"content": "", "author": {"id": "7"}}) is None
    assert parse_discord_message({}) is None


def test_discord_send_rest():
    t, calls = _http_recorder({"_status": 200, "id": "1"})
    a = DiscordGatewayAdapter(token="botT", http_transport=t)
    r = run(a.send(chat_id="555", text="y" * 2500))  # truncated to 2000
    assert calls["url"] == "https://discord.com/api/v10/channels/555/messages"
    assert len(calls["json"]["content"]) == 2000
    assert calls["headers"]["Authorization"] == "Bot botT"
    assert r["ok"] is True


def test_discord_allow_and_token_guard():
    a = DiscordGatewayAdapter(token="x", allowed_channel_ids=[555], http_transport=_http_recorder({})[0])
    assert a.allow(InboundMessage(platform="discord", chat_id="555", text="x")) is True
    assert a.allow(InboundMessage(platform="discord", chat_id="999", text="x")) is False
    with pytest.raises(ValueError):
        DiscordGatewayAdapter(token="")


# ── Slack ──────────────────────────────────────────────────────────────


def test_parse_slack_event():
    m = parse_slack_event({"type": "message", "channel": "C1", "user": "U1", "text": "yo", "ts": "1.2"})
    assert m and m.platform == "slack" and m.chat_id == "C1" and m.text == "yo"


def test_parse_slack_ignores_bot_and_subtype():
    assert parse_slack_event({"type": "message", "text": "x", "bot_id": "B1"}) is None
    assert parse_slack_event({"type": "message", "text": "x", "subtype": "message_changed"}) is None
    assert parse_slack_event({"type": "reaction_added"}) is None
    assert parse_slack_event({"type": "message", "text": ""}) is None


def test_slack_send_rest():
    t, calls = _http_recorder({"ok": True})
    a = SlackGatewayAdapter(app_token="xapp", bot_token="xoxb", http_transport=t)
    r = run(a.send(chat_id="C1", text="hello"))
    assert calls["url"] == "https://slack.com/api/chat.postMessage"
    assert calls["json"] == {"channel": "C1", "text": "hello"}
    assert calls["headers"]["Authorization"] == "Bearer xoxb"
    assert r["ok"] is True


def test_slack_token_guards():
    with pytest.raises(ValueError):
        SlackGatewayAdapter(app_token="", bot_token="xoxb")
    with pytest.raises(ValueError):
        SlackGatewayAdapter(app_token="xapp", bot_token="")


# ── queue draining (no live WS) ────────────────────────────────────────


def test_fetch_drains_queue_without_starting_ws():
    a = DiscordGatewayAdapter(token="x", http_transport=_http_recorder({})[0])

    async def scenario():
        a._closed = True  # so _ensure_connected() no-ops (no real WS)
        await a._put(InboundMessage(platform="discord", chat_id="c", text="m1"))
        await a._put(InboundMessage(platform="discord", chat_id="c", text="m2"))
        return await a.fetch()

    msgs = run(scenario())
    assert [m.text for m in msgs] == ["m1", "m2"]


def test_fetch_idle_timeout_returns_empty():
    a = DiscordGatewayAdapter(token="x", idle_timeout=0.05, http_transport=_http_recorder({})[0])

    async def scenario():
        a._closed = True
        return await a.fetch()  # nothing queued → times out → []

    assert run(scenario()) == []


# ── factory ────────────────────────────────────────────────────────────


def test_builtin_platforms_now_three():
    assert set(BUILTIN_GATEWAY_PLATFORMS) == {"telegram", "discord", "slack"}


def test_factory_builds_discord_and_slack():
    d = build_platform_adapter("discord", {"token": "x"})
    assert isinstance(d, DiscordGatewayAdapter)
    s = build_platform_adapter("slack", {"app_token": "xapp", "bot_token": "xoxb"})
    assert isinstance(s, SlackGatewayAdapter)
