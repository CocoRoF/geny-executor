"""Six task lifecycle tools tests (PR-A.1.5)."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, List

import pytest

from geny_executor.runtime import (
    BackgroundTaskExecutor,
    BackgroundTaskRunner,
)
from geny_executor.stages.s13_task_registry import (
    InMemoryRegistry,
    TaskFilter,
    TaskRecord,
    TaskStatus,
)
from geny_executor.tools.base import ToolContext
from geny_executor.tools.built_in import (
    BUILT_IN_TOOL_CLASSES,
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskOutputTool,
    TaskStopTool,
    TaskUpdateTool,
)


# ── Fixtures ─────────────────────────────────────────────────────────


class _StubExecutor(BackgroundTaskExecutor):
    def __init__(self, chunks: List[bytes], delay: float = 0.0):
        self.chunks = chunks
        self.delay = delay

    async def execute(self, record: TaskRecord) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield chunk


class _SlowExecutor(BackgroundTaskExecutor):
    async def execute(self, record: TaskRecord) -> AsyncIterator[bytes]:
        await asyncio.sleep(60)
        yield b"never"


@pytest.fixture
def registry() -> InMemoryRegistry:
    return InMemoryRegistry()


@pytest.fixture
def runner(registry: InMemoryRegistry) -> BackgroundTaskRunner:
    return BackgroundTaskRunner(
        registry,
        {
            "echo": _StubExecutor([b"hello"], delay=0.01),
            "slow": _SlowExecutor(),
        },
    )


def _ctx(registry, runner=None) -> ToolContext:
    extras = {"task_registry": registry}
    if runner is not None:
        extras["task_runner"] = runner
    return ToolContext(extras=extras)


# ── Registration ─────────────────────────────────────────────────────


def test_six_tools_registered():
    for name in ("TaskCreate", "TaskGet", "TaskList", "TaskUpdate", "TaskOutput", "TaskStop"):
        assert name in BUILT_IN_TOOL_CLASSES


# ── TaskCreate ───────────────────────────────────────────────────────


class TestTaskCreate:
    @pytest.mark.asyncio
    async def test_creates_and_submits(self, registry, runner):
        ctx = _ctx(registry, runner)
        result = await TaskCreateTool().execute({"kind": "echo"}, ctx)
        assert result.is_error is False
        task_id = result.content["task_id"]
        # Wait for completion.
        for _ in range(100):
            r = registry.get(task_id)
            if r and r.is_terminal:
                break
            await asyncio.sleep(0.01)
        assert registry.get(task_id).status == TaskStatus.DONE

    @pytest.mark.asyncio
    async def test_explicit_task_id_used(self, registry, runner):
        ctx = _ctx(registry, runner)
        result = await TaskCreateTool().execute(
            {"kind": "echo", "task_id": "custom-123"}, ctx,
        )
        assert result.content["task_id"] == "custom-123"

    @pytest.mark.asyncio
    async def test_no_runner(self, registry):
        ctx = _ctx(registry)
        result = await TaskCreateTool().execute({"kind": "echo"}, ctx)
        assert result.is_error is True
        assert result.content["error"]["code"] == "NO_RUNNER"

    @pytest.mark.asyncio
    async def test_missing_kind(self, registry, runner):
        ctx = _ctx(registry, runner)
        result = await TaskCreateTool().execute({"kind": ""}, ctx)
        assert result.is_error is True
        assert result.content["error"]["code"] == "BAD_INPUT"

    @pytest.mark.asyncio
    async def test_unknown_kind_marks_failed(self, registry, runner):
        ctx = _ctx(registry, runner)
        result = await TaskCreateTool().execute({"kind": "ghost"}, ctx)
        assert result.is_error is False
        rec = registry.get(result.content["task_id"])
        assert rec.status == TaskStatus.FAILED


# ── TaskGet ──────────────────────────────────────────────────────────


class TestTaskGet:
    @pytest.mark.asyncio
    async def test_returns_record(self, registry):
        registry.register(TaskRecord(task_id="t1", kind="K", payload={"x": 1}))
        result = await TaskGetTool().execute({"task_id": "t1"}, _ctx(registry))
        assert result.is_error is False
        assert result.content["task_id"] == "t1"
        assert result.content["kind"] == "K"

    @pytest.mark.asyncio
    async def test_not_found(self, registry):
        result = await TaskGetTool().execute({"task_id": "ghost"}, _ctx(registry))
        assert result.is_error is True
        assert result.content["error"]["code"] == "NOT_FOUND"

    @pytest.mark.asyncio
    async def test_no_registry(self):
        ctx = ToolContext(extras={})
        result = await TaskGetTool().execute({"task_id": "x"}, ctx)
        assert result.is_error is True
        assert result.content["error"]["code"] == "NO_REGISTRY"


# ── TaskList ─────────────────────────────────────────────────────────


class TestTaskList:
    @pytest.mark.asyncio
    async def test_returns_all_when_no_filter(self, registry):
        for i in range(3):
            registry.register(TaskRecord(task_id=f"t{i}"))
        result = await TaskListTool().execute({}, _ctx(registry))
        ids = [r["task_id"] for r in result.content["tasks"]]
        assert sorted(ids) == ["t0", "t1", "t2"]

    @pytest.mark.asyncio
    async def test_filter_by_status(self, registry):
        a = TaskRecord(task_id="a")
        b = TaskRecord(task_id="b")
        b.mark(TaskStatus.RUNNING)
        registry.register(a)
        registry.register(b)
        result = await TaskListTool().execute({"status": "running"}, _ctx(registry))
        ids = [r["task_id"] for r in result.content["tasks"]]
        assert ids == ["b"]

    @pytest.mark.asyncio
    async def test_filter_by_kind_and_limit(self, registry):
        for i in range(5):
            registry.register(TaskRecord(task_id=f"t{i}", kind="K"))
        result = await TaskListTool().execute({"kind": "K", "limit": 2}, _ctx(registry))
        assert len(result.content["tasks"]) == 2


# ── TaskUpdate ───────────────────────────────────────────────────────


class TestTaskUpdate:
    @pytest.mark.asyncio
    async def test_updates_payload(self, registry):
        registry.register(TaskRecord(task_id="t1", payload={"a": 1}))
        result = await TaskUpdateTool().execute(
            {"task_id": "t1", "payload": {"a": 2, "b": 3}}, _ctx(registry),
        )
        assert result.is_error is False
        assert result.content["payload"] == {"a": 2, "b": 3}

    @pytest.mark.asyncio
    async def test_rejects_non_user_mutable_field(self, registry):
        registry.register(TaskRecord(task_id="t1"))
        result = await TaskUpdateTool().execute(
            {"task_id": "t1", "status": "done"}, _ctx(registry),
        )
        assert result.is_error is True
        assert result.content["error"]["code"] == "FIELD_NOT_MUTABLE"
        # Status was NOT changed.
        assert registry.get("t1").status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_not_found(self, registry):
        result = await TaskUpdateTool().execute(
            {"task_id": "ghost"}, _ctx(registry),
        )
        assert result.is_error is True
        assert result.content["error"]["code"] == "NOT_FOUND"


# ── TaskOutput ───────────────────────────────────────────────────────


class TestTaskOutput:
    @pytest.mark.asyncio
    async def test_reads_buffered_output(self, registry):
        registry.register(TaskRecord(task_id="t1"))
        await registry.append_output("t1", b"hello world")
        result = await TaskOutputTool().execute({"task_id": "t1"}, _ctx(registry))
        assert result.content["text"] == "hello world"
        assert result.content["byte_count"] == 11
        assert result.content["truncated"] is False

    @pytest.mark.asyncio
    async def test_offset_and_limit(self, registry):
        registry.register(TaskRecord(task_id="t1"))
        await registry.append_output("t1", b"abcdefghij")
        result = await TaskOutputTool().execute(
            {"task_id": "t1", "offset": 2, "limit": 4}, _ctx(registry),
        )
        assert result.content["text"] == "cdef"
        assert result.content["truncated"] is True

    @pytest.mark.asyncio
    async def test_empty_output(self, registry):
        registry.register(TaskRecord(task_id="t1"))
        result = await TaskOutputTool().execute({"task_id": "t1"}, _ctx(registry))
        assert result.content["text"] == ""
        assert result.content["byte_count"] == 0

    @pytest.mark.asyncio
    async def test_caps_oversize_limit(self, registry):
        registry.register(TaskRecord(task_id="t1"))
        # Request way more than the cap; should silently cap.
        result = await TaskOutputTool().execute(
            {"task_id": "t1", "limit": 999_999_999_999}, _ctx(registry),
        )
        assert result.is_error is False  # capped silently


# ── TaskStop ─────────────────────────────────────────────────────────


class TestTaskStop:
    @pytest.mark.asyncio
    async def test_stops_running_task(self, registry, runner):
        ctx = _ctx(registry, runner)
        # Submit a slow task.
        sub = await TaskCreateTool().execute({"kind": "slow"}, ctx)
        task_id = sub.content["task_id"]
        await asyncio.sleep(0.05)
        result = await TaskStopTool().execute({"task_id": task_id}, ctx)
        assert result.is_error is False
        assert result.content["stopped"] is True
        assert registry.get(task_id).status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_stop_unknown_returns_false(self, registry, runner):
        ctx = _ctx(registry, runner)
        result = await TaskStopTool().execute({"task_id": "ghost"}, ctx)
        assert result.is_error is False
        assert result.content["stopped"] is False

    @pytest.mark.asyncio
    async def test_no_runner(self, registry):
        ctx = _ctx(registry)
        result = await TaskStopTool().execute({"task_id": "x"}, ctx)
        assert result.is_error is True
        assert result.content["error"]["code"] == "NO_RUNNER"
