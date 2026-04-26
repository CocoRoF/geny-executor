"""Operator tools tests — Config / Monitor / SendUserFile (PR-A.3.6)."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from geny_executor.channels import UserFileChannel
from geny_executor.tools.base import ToolContext
from geny_executor.tools.built_in import (
    BUILT_IN_TOOL_CLASSES,
    ConfigTool,
    MonitorTool,
    SendUserFileTool,
)


def test_all_three_registered():
    for name in ("Config", "Monitor", "SendUserFile"):
        assert name in BUILT_IN_TOOL_CLASSES


# ── Config ───────────────────────────────────────────────────────────


class _FakeStage:
    def __init__(self, name, slots):
        self.name = name
        self._slots = slots

    def get_strategy_slots(self):
        return self._slots


class _FakePipeline:
    def __init__(self, stages=None):
        self.stages = stages or []


class _FakeMutator:
    def __init__(self):
        self.store: Dict = {}

    def get(self, section, key):
        return self.store.get((section, key))

    def set(self, section, key, value):
        self.store[(section, key)] = value


class TestConfig:
    @pytest.mark.asyncio
    async def test_list_active(self):
        slot = type("S", (), {"strategy": object()})()
        pipeline = _FakePipeline(stages=[_FakeStage("api", {"completion": slot})])
        ctx = ToolContext(extras={"pipeline": pipeline})
        result = await ConfigTool().execute({"action": "list_active"}, ctx)
        assert result.is_error is False
        assert result.content["active"][0]["stage"] == "api"

    @pytest.mark.asyncio
    async def test_list_active_no_pipeline(self):
        ctx = ToolContext(extras={})
        result = await ConfigTool().execute({"action": "list_active"}, ctx)
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_set_via_mutator(self):
        mutator = _FakeMutator()
        ctx = ToolContext(extras={"pipeline_mutator": mutator})
        result = await ConfigTool().execute(
            {"action": "set", "section": "model", "key": "default", "value": "claude-haiku"},
            ctx,
        )
        assert result.is_error is False
        assert mutator.store[("model", "default")] == "claude-haiku"

    @pytest.mark.asyncio
    async def test_set_no_mutator(self):
        ctx = ToolContext(extras={})
        result = await ConfigTool().execute(
            {"action": "set", "section": "x", "key": "y", "value": 1}, ctx,
        )
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_get_via_mutator(self):
        mutator = _FakeMutator()
        mutator.set("model", "default", "claude-haiku")
        ctx = ToolContext(extras={"pipeline_mutator": mutator})
        result = await ConfigTool().execute(
            {"action": "get", "section": "model", "key": "default"}, ctx,
        )
        assert result.content["value"] == "claude-haiku"


# ── Monitor ──────────────────────────────────────────────────────────


class _FakeEvent:
    def __init__(self, type_, data):
        self.type = type_
        self.ts = datetime(2026, 4, 26, tzinfo=timezone.utc)
        self.data = data


class _FakeBus:
    def __init__(self, events: List[_FakeEvent]):
        self._events = events

    def subscribe(self, *args, **kwargs):
        events = self._events

        @asynccontextmanager
        async def _ctx():
            async def _stream():
                for e in events:
                    yield e
            yield _stream()
        return _ctx()


class TestMonitor:
    @pytest.mark.asyncio
    async def test_collects_events(self):
        bus = _FakeBus([_FakeEvent("a", {"x": 1}), _FakeEvent("b", {"y": 2})])
        ctx = ToolContext(extras={"event_bus": bus})
        result = await MonitorTool().execute({"duration_seconds": 1, "max_events": 10}, ctx)
        assert result.content["count"] == 2
        assert result.content["events"][0]["type"] == "a"

    @pytest.mark.asyncio
    async def test_max_events_caps(self):
        bus = _FakeBus([_FakeEvent("a", None) for _ in range(10)])
        ctx = ToolContext(extras={"event_bus": bus})
        result = await MonitorTool().execute({"duration_seconds": 1, "max_events": 3}, ctx)
        assert result.content["count"] == 3

    @pytest.mark.asyncio
    async def test_no_bus(self):
        ctx = ToolContext(extras={})
        result = await MonitorTool().execute({}, ctx)
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_bus_without_subscribe(self):
        ctx = ToolContext(extras={"event_bus": object()})
        result = await MonitorTool().execute({}, ctx)
        assert result.is_error is True


# ── SendUserFile ─────────────────────────────────────────────────────


class _FakeChannel(UserFileChannel):
    def __init__(self, response=None, raise_exc=None):
        self.response = response or {"download_url": "https://x/download"}
        self.raise_exc = raise_exc
        self.last_call = None

    async def send(self, path, *, filename=None, content_type=None, description=None):
        self.last_call = (path, filename, content_type, description)
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.response


class TestSendUserFile:
    @pytest.mark.asyncio
    async def test_delivers(self, tmp_path: Path):
        f = tmp_path / "report.txt"
        f.write_text("hello\n")
        channel = _FakeChannel()
        ctx = ToolContext(working_dir=str(tmp_path), extras={"user_file_channel": channel})
        result = await SendUserFileTool().execute({"file_path": "report.txt"}, ctx)
        assert result.is_error is False
        assert result.content["delivered"] is True
        assert channel.last_call[1] == "report.txt"

    @pytest.mark.asyncio
    async def test_missing_file(self, tmp_path: Path):
        channel = _FakeChannel()
        ctx = ToolContext(working_dir=str(tmp_path), extras={"user_file_channel": channel})
        result = await SendUserFileTool().execute({"file_path": "ghost.txt"}, ctx)
        assert result.is_error is True
        assert result.content["error"]["code"] == "FILE_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_no_channel(self, tmp_path: Path):
        ctx = ToolContext(working_dir=str(tmp_path), extras={})
        result = await SendUserFileTool().execute({"file_path": "x.txt"}, ctx)
        assert result.is_error is True
        assert result.content["error"]["code"] == "NO_CHANNEL"

    @pytest.mark.asyncio
    async def test_channel_failure(self, tmp_path: Path):
        f = tmp_path / "x.txt"
        f.write_text("x")
        channel = _FakeChannel(raise_exc=RuntimeError("bucket full"))
        ctx = ToolContext(working_dir=str(tmp_path), extras={"user_file_channel": channel})
        result = await SendUserFileTool().execute({"file_path": "x.txt"}, ctx)
        assert result.is_error is True
        assert result.content["error"]["code"] == "SEND_FAILED"

    @pytest.mark.asyncio
    async def test_directory_rejected(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        ctx = ToolContext(working_dir=str(tmp_path), extras={"user_file_channel": _FakeChannel()})
        result = await SendUserFileTool().execute({"file_path": "sub"}, ctx)
        assert result.is_error is True
        assert result.content["error"]["code"] == "NOT_A_FILE"
