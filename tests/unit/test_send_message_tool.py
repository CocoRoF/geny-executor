"""SendMessageTool + channel registry tests (PR-A.3.7)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from geny_executor.channels import (
    SendMessageChannel,
    SendMessageChannelRegistry,
    StdoutSendMessageChannel,
)
from geny_executor.tools.base import ToolContext
from geny_executor.tools.built_in import BUILT_IN_TOOL_CLASSES, SendMessageTool


def test_registered():
    assert "SendMessage" in BUILT_IN_TOOL_CLASSES


# ── Registry ─────────────────────────────────────────────────────────


class _RecordingChannel(SendMessageChannel):
    def __init__(self, response=None, raise_exc=None):
        self.response = response or {"delivered": True}
        self.raise_exc = raise_exc
        self.calls: List[Dict[str, Any]] = []

    async def send(self, *, to=None, message, attachments=None):
        self.calls.append({"to": to, "message": message, "attachments": list(attachments or [])})
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.response


class TestRegistry:
    def test_register_then_get(self):
        reg = SendMessageChannelRegistry()
        c = _RecordingChannel()
        reg.register("slack", c)
        assert reg.get("slack") is c

    def test_get_unknown_returns_none(self):
        assert SendMessageChannelRegistry().get("ghost") is None

    def test_list_channel_names(self):
        reg = SendMessageChannelRegistry()
        reg.register("a", _RecordingChannel())
        reg.register("b", _RecordingChannel())
        assert reg.list() == ["a", "b"]

    @pytest.mark.asyncio
    async def test_stdout_channel_logs(self, caplog):
        c = StdoutSendMessageChannel()
        result = await c.send(to="user", message="hi")
        assert result["delivered"] is True


# ── Tool ─────────────────────────────────────────────────────────────


def _ctx_with(channel_name="slack", **channel_kwargs):
    reg = SendMessageChannelRegistry()
    ch = _RecordingChannel(**channel_kwargs)
    reg.register(channel_name, ch)
    return ToolContext(extras={"send_message_channels": reg}), ch


class TestExecute:
    @pytest.mark.asyncio
    async def test_dispatches_to_registered(self):
        ctx, ch = _ctx_with()
        result = await SendMessageTool().execute(
            {"channel": "slack", "to": "user-1", "message": "hello"}, ctx,
        )
        assert result.is_error is False
        assert ch.calls[0]["message"] == "hello"
        assert ch.calls[0]["to"] == "user-1"

    @pytest.mark.asyncio
    async def test_no_registry(self):
        ctx = ToolContext(extras={})
        result = await SendMessageTool().execute(
            {"channel": "slack", "message": "hi"}, ctx,
        )
        assert result.is_error is True
        assert result.content["error"]["code"] == "NO_REGISTRY"

    @pytest.mark.asyncio
    async def test_unknown_channel(self):
        ctx, _ = _ctx_with("known")
        result = await SendMessageTool().execute(
            {"channel": "ghost", "message": "hi"}, ctx,
        )
        assert result.is_error is True
        assert result.content["error"]["code"] == "UNKNOWN_CHANNEL"

    @pytest.mark.asyncio
    async def test_send_failure(self):
        ctx, _ = _ctx_with(raise_exc=RuntimeError("rate limited"))
        result = await SendMessageTool().execute(
            {"channel": "slack", "message": "hi"}, ctx,
        )
        assert result.is_error is True
        assert result.content["error"]["code"] == "SEND_FAILED"

    @pytest.mark.asyncio
    async def test_attachments_passed(self):
        ctx, ch = _ctx_with()
        await SendMessageTool().execute(
            {"channel": "slack", "message": "see file", "attachments": ["a.png", "b.txt"]},
            ctx,
        )
        assert ch.calls[0]["attachments"] == ["a.png", "b.txt"]
