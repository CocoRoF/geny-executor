"""Built-in SendMessageChannel transports + config factory (2.10.0)."""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor.channels import (
    BUILTIN_CHANNEL_KINDS,
    build_channel_registry,
    build_send_message_channel,
)
from geny_executor.channels.built_in import (
    DiscordSendMessageChannel,
    NtfySendMessageChannel,
    SlackSendMessageChannel,
    TelegramSendMessageChannel,
    WebhookSendMessageChannel,
)


def _recorder():
    """An injectable transport that records the last call and returns 200."""
    calls = {}

    async def transport(url, *, json=None, data=None, headers=None, params=None):
        calls.update(url=url, json=json, data=data, headers=headers, params=params)
        return {"status": 200, "ok": True, "body": "ok"}

    return transport, calls


def run(coro):
    return asyncio.run(coro)


# ── per-transport wire format ──────────────────────────────────────────


def test_webhook_posts_envelope():
    t, calls = _recorder()
    ch = WebhookSendMessageChannel(url="https://h/wh", headers={"X-Auth": "k"}, transport=t)
    r = run(ch.send(to="u1", message="hello", attachments=["a.png"]))
    assert calls["url"] == "https://h/wh"
    assert calls["json"] == {"to": "u1", "message": "hello", "attachments": ["a.png"]}
    assert calls["headers"] == {"X-Auth": "k"}
    assert r == {"channel": "webhook", "delivered": True, "status": 200}


def test_telegram_url_and_body():
    t, calls = _recorder()
    ch = TelegramSendMessageChannel(token="123:abc", chat_id="555", parse_mode="Markdown", transport=t)
    run(ch.send(message="hi"))
    assert calls["url"] == "https://api.telegram.org/bot123:abc/sendMessage"
    assert calls["json"] == {"chat_id": "555", "text": "hi", "parse_mode": "Markdown"}


def test_telegram_to_overrides_chat_id_and_appends_attachments():
    t, calls = _recorder()
    ch = TelegramSendMessageChannel(token="x", transport=t)
    run(ch.send(to="999", message="m", attachments=["http://a", "http://b"]))
    assert calls["json"]["chat_id"] == "999"
    assert calls["json"]["text"] == "m\nhttp://a\nhttp://b"


def test_telegram_requires_chat_id():
    ch = TelegramSendMessageChannel(token="x", transport=_recorder()[0])
    with pytest.raises(ValueError):
        run(ch.send(message="m"))


def test_discord_content_and_truncation():
    t, calls = _recorder()
    ch = DiscordSendMessageChannel(webhook_url="https://d/wh", transport=t)
    run(ch.send(message="z" * 2500))
    assert len(calls["json"]["content"]) == 2000


def test_slack_text():
    t, calls = _recorder()
    ch = SlackSendMessageChannel(webhook_url="https://s/wh", transport=t)
    run(ch.send(message="ping"))
    assert calls["json"] == {"text": "ping"}


def test_ntfy_topic_body_and_auth():
    t, calls = _recorder()
    ch = NtfySendMessageChannel(topic="mytopic", token="tok", title="T", transport=t)
    run(ch.send(message="body", attachments=["http://file"]))
    assert calls["url"] == "https://ntfy.sh/mytopic"
    assert calls["data"] == b"body"
    assert calls["headers"]["Authorization"] == "Bearer tok"
    assert calls["headers"]["Title"] == "T"
    assert calls["headers"]["Attach"] == "http://file"


# ── construction guards ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (WebhookSendMessageChannel, {"url": ""}),
        (TelegramSendMessageChannel, {"token": ""}),
        (DiscordSendMessageChannel, {"webhook_url": ""}),
        (SlackSendMessageChannel, {"webhook_url": ""}),
        (NtfySendMessageChannel, {"topic": ""}),
    ],
)
def test_missing_config_raises(cls, kwargs):
    with pytest.raises(ValueError):
        cls(**kwargs)


# ── factory + registry ─────────────────────────────────────────────────


def test_builtin_kinds():
    assert set(BUILTIN_CHANNEL_KINDS) == {
        "stdout", "webhook", "telegram", "discord", "slack", "ntfy"
    }


def test_build_send_message_channel():
    ch = build_send_message_channel("slack", {"webhook_url": "https://s/wh"})
    assert isinstance(ch, SlackSendMessageChannel)


def test_build_unknown_kind_raises():
    with pytest.raises(ValueError):
        build_send_message_channel("carrier-pigeon", {})


def test_build_registry_skips_bad_entries():
    reg = build_channel_registry([
        {"name": "ops", "kind": "slack", "config": {"webhook_url": "https://s"}},
        {"name": "missing-token", "kind": "telegram", "config": {}},  # ValueError → skip
        {"name": "no-kind"},                                          # skip
        {"kind": "webhook", "config": {"url": "https://w"}},          # no name → skip
        {"name": "huh", "kind": "nope", "config": {}},               # unknown → skip
        {"name": "gen", "kind": "webhook", "config": {"url": "https://w"}},
    ])
    assert reg.list() == ["gen", "ops"]


def test_build_registry_extends_existing():
    reg = build_channel_registry([{"name": "a", "kind": "stdout"}])
    reg2 = build_channel_registry([{"name": "b", "kind": "stdout"}], registry=reg)
    assert reg2 is reg
    assert reg.list() == ["a", "b"]
