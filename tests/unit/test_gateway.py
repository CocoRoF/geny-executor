"""Inbound gateway — Telegram adapter, runner loop, config factory (2.11.0)."""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor.gateway import (
    BUILTIN_GATEWAY_PLATFORMS,
    GatewayReply,
    GatewayRunner,
    InboundMessage,
    PlatformAdapter,
    TelegramGatewayAdapter,
    build_gateway,
    build_platform_adapter,
)


# ── Telegram adapter (injected transport) ──────────────────────────────


def _tg_transport(updates):
    """A fake telegram transport: GET getUpdates → the given updates once, then
    empty; POST sendMessage → record + ok."""
    state = {"updates": list(updates), "sent": []}

    async def transport(method, url, *, params=None, json=None, timeout=None):
        if method == "GET":
            out = {"ok": True, "result": state["updates"]}
            state["updates"] = []  # consume once
            return out
        state["sent"].append(json)
        return {"ok": True, "result": {"message_id": 1}}

    return transport, state


def test_telegram_fetch_parses_and_advances_offset():
    t, _ = _tg_transport([
        {"update_id": 10, "message": {"message_id": 1, "text": "hi",
                                      "chat": {"id": 555}, "from": {"id": 7, "first_name": "A"}}},
        {"update_id": 11, "message": {"message_id": 2, "text": "yo",
                                      "chat": {"id": 555}, "from": {"id": 7}}},
        {"update_id": 12, "message": {"message_id": 3, "sticker": {},  # non-text → skipped
                                      "chat": {"id": 555}}},
    ])
    a = TelegramGatewayAdapter(token="x", transport=t)
    msgs = asyncio.run(a.fetch())
    assert [m.text for m in msgs] == ["hi", "yo"]
    assert msgs[0].chat_id == "555" and msgs[0].platform == "telegram"
    assert msgs[0].sender_name == "A"
    assert a._offset == 13  # max update_id + 1


def test_telegram_send_body():
    t, state = _tg_transport([])
    a = TelegramGatewayAdapter(token="x", parse_mode="Markdown", transport=t)
    r = asyncio.run(a.send(chat_id="555", text="hello"))
    assert state["sent"] == [{"chat_id": "555", "text": "hello", "parse_mode": "Markdown"}]
    assert r["ok"] is True


def test_telegram_allowlist():
    a = TelegramGatewayAdapter(token="x", allowed_chat_ids=[555], transport=_tg_transport([])[0])
    assert a.allow(InboundMessage(platform="telegram", chat_id="555", text="x")) is True
    assert a.allow(InboundMessage(platform="telegram", chat_id="999", text="x")) is False


def test_telegram_open_when_no_allowlist():
    a = TelegramGatewayAdapter(token="x", transport=_tg_transport([])[0])
    assert a.allow(InboundMessage(platform="telegram", chat_id="anything", text="x")) is True


def test_telegram_requires_token():
    with pytest.raises(ValueError):
        TelegramGatewayAdapter(token="")


# ── factory ────────────────────────────────────────────────────────────


def test_builtin_platforms():
    assert "telegram" in BUILTIN_GATEWAY_PLATFORMS


def test_build_platform_adapter_unknown_raises():
    with pytest.raises(ValueError):
        build_platform_adapter("carrier-pigeon", {})


def test_build_gateway_skips_bad_specs():
    async def handler(_m):
        return None

    runner = build_gateway(
        [
            {"platform": "telegram", "config": {"token": "a"}},
            {"platform": "telegram", "config": {}},     # missing token → skip
            {"config": {"token": "b"}},                  # no platform → skip
            {"platform": "nope", "config": {}},          # unknown → skip
        ],
        handler,
    )
    assert [a.name for a in runner.adapters] == ["telegram"]


# ── runner loop (fake adapter) ─────────────────────────────────────────


class FakeAdapter(PlatformAdapter):
    name = "fake"

    def __init__(self, batches, *, allow_all=True):
        self._batches = list(batches)
        self.sent = []
        self.closed = False
        self._allow_all = allow_all

    async def fetch(self):
        if self._batches:
            return self._batches.pop(0)
        await asyncio.sleep(3600)  # idle until cancelled
        return []

    async def send(self, *, chat_id, text):
        self.sent.append((chat_id, text))
        return {"ok": True}

    def allow(self, message):
        return self._allow_all

    async def close(self):
        self.closed = True


def _msg(text, chat="c1"):
    return InboundMessage(platform="fake", chat_id=chat, text=text)


async def _run_until(adapter, runner, predicate, tries=100):
    await runner.start()
    for _ in range(tries):
        if predicate():
            break
        await asyncio.sleep(0.01)
    await runner.shutdown()


async def test_runner_dispatches_and_replies():
    adapter = FakeAdapter([[_msg("hi"), _msg("yo", chat="c2")]])

    async def handler(m):
        return f"echo:{m.text}"

    runner = GatewayRunner([adapter], handler)
    await _run_until(adapter, runner, lambda: len(adapter.sent) >= 2)
    assert set(adapter.sent) == {("c1", "echo:hi"), ("c2", "echo:yo")}
    assert adapter.closed is True


async def test_runner_none_reply_sends_nothing():
    adapter = FakeAdapter([[_msg("hi")]])

    async def handler(_m):
        return None

    runner = GatewayRunner([adapter], handler)
    await _run_until(adapter, runner, lambda: False, tries=20)  # give it time
    assert adapter.sent == []


async def test_runner_respects_allow():
    adapter = FakeAdapter([[_msg("hi")]], allow_all=False)

    async def handler(_m):
        return "should-not-send"

    runner = GatewayRunner([adapter], handler)
    await _run_until(adapter, runner, lambda: False, tries=20)
    assert adapter.sent == []


async def test_runner_handler_exception_is_isolated():
    adapter = FakeAdapter([[_msg("boom"), _msg("ok")]])

    async def handler(m):
        if m.text == "boom":
            raise RuntimeError("intentional")
        return "fine"

    runner = GatewayRunner([adapter], handler)
    await _run_until(adapter, runner, lambda: ("c1", "fine") in adapter.sent)
    assert ("c1", "fine") in adapter.sent  # the good message still replied


async def test_runner_gateway_reply_override_chat():
    adapter = FakeAdapter([[_msg("hi", chat="c1")]])

    async def handler(_m):
        return GatewayReply(text="hey", chat_id="other")

    runner = GatewayRunner([adapter], handler)
    await _run_until(adapter, runner, lambda: bool(adapter.sent))
    assert adapter.sent == [("other", "hey")]


async def test_runner_start_idempotent_and_no_adapters():
    runner = GatewayRunner([], lambda m: None)  # no adapters
    await runner.start()
    assert runner.running is False  # nothing to run
    await runner.shutdown()
