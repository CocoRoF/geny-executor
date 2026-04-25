"""BackgroundTaskRunner + executor tests (PR-A.1.3)."""

from __future__ import annotations

import asyncio
import sys
from typing import AsyncIterator, List

import pytest

from geny_executor.runtime import (
    BackgroundTaskExecutor,
    BackgroundTaskRunner,
    LocalAgentExecutor,
    LocalBashExecutor,
)
from geny_executor.stages.s13_task_registry import (
    InMemoryRegistry,
    TaskFilter,
    TaskRecord,
    TaskStatus,
)


# ── Test executor doubles ────────────────────────────────────────────


class _StubExecutor(BackgroundTaskExecutor):
    """Yields ``chunks`` then optionally raises."""

    def __init__(
        self,
        chunks: List[bytes],
        *,
        raise_after: bool = False,
        delay: float = 0.0,
    ) -> None:
        self.chunks = chunks
        self.raise_after = raise_after
        self.delay = delay

    async def execute(self, record: TaskRecord) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield chunk
        if self.raise_after:
            raise RuntimeError("stub failure")


class _SlowExecutor(BackgroundTaskExecutor):
    """Sleeps long enough that the test can stop() it mid-flight."""

    async def execute(self, record: TaskRecord) -> AsyncIterator[bytes]:
        try:
            await asyncio.sleep(60)
            yield b"never"
        except asyncio.CancelledError:
            raise


# ── BackgroundTaskRunner ─────────────────────────────────────────────


class TestSubmit:
    @pytest.mark.asyncio
    async def test_submit_runs_to_completion(self):
        registry = InMemoryRegistry()
        runner = BackgroundTaskRunner(
            registry,
            {"k": _StubExecutor([b"hello", b" world"])},
        )
        task_id = await runner.submit(TaskRecord(task_id="t1", kind="k"))
        # Wait until terminal — give the asyncio.Task a chance.
        for _ in range(50):
            rec = registry.get(task_id)
            if rec and rec.is_terminal:
                break
            await asyncio.sleep(0.01)
        rec = registry.get(task_id)
        assert rec is not None
        assert rec.status == TaskStatus.DONE
        assert await registry.read_output(task_id) == b"hello world"

    @pytest.mark.asyncio
    async def test_submit_unknown_kind_marks_failed(self):
        registry = InMemoryRegistry()
        runner = BackgroundTaskRunner(registry, {})
        task_id = await runner.submit(TaskRecord(task_id="t1", kind="ghost"))
        rec = registry.get(task_id)
        assert rec is not None
        assert rec.status == TaskStatus.FAILED
        assert "no_executor_for_kind:ghost" in (rec.error or "")

    @pytest.mark.asyncio
    async def test_submit_idempotent_when_already_in_flight(self):
        registry = InMemoryRegistry()
        runner = BackgroundTaskRunner(
            registry,
            {"k": _SlowExecutor()},
        )
        await runner.submit(TaskRecord(task_id="t1", kind="k"))
        # Re-submit while still running — should not double-spawn.
        await runner.submit(TaskRecord(task_id="t1", kind="k"))
        # Stop and clean up.
        await runner.shutdown(timeout=2)

    @pytest.mark.asyncio
    async def test_executor_failure_marks_failed_with_error(self):
        registry = InMemoryRegistry()
        runner = BackgroundTaskRunner(
            registry,
            {"k": _StubExecutor([b"chunk"], raise_after=True)},
        )
        task_id = await runner.submit(TaskRecord(task_id="t1", kind="k"))
        for _ in range(50):
            rec = registry.get(task_id)
            if rec and rec.is_terminal:
                break
            await asyncio.sleep(0.01)
        rec = registry.get(task_id)
        assert rec.status == TaskStatus.FAILED
        assert rec.error == "stub failure"
        # Output written before failure is preserved.
        assert await registry.read_output(task_id) == b"chunk"


class TestStop:
    @pytest.mark.asyncio
    async def test_stop_cancels_in_flight(self):
        registry = InMemoryRegistry()
        runner = BackgroundTaskRunner(registry, {"k": _SlowExecutor()})
        task_id = await runner.submit(TaskRecord(task_id="t1", kind="k"))
        # Give the task a tick to start.
        await asyncio.sleep(0.05)
        ok = await runner.stop(task_id)
        assert ok is True
        rec = registry.get(task_id)
        assert rec.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_stop_unknown_returns_false(self):
        registry = InMemoryRegistry()
        runner = BackgroundTaskRunner(registry, {})
        assert await runner.stop("ghost") is False


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_cancels_all(self):
        registry = InMemoryRegistry()
        runner = BackgroundTaskRunner(registry, {"k": _SlowExecutor()})
        await runner.submit(TaskRecord(task_id="a", kind="k"))
        await runner.submit(TaskRecord(task_id="b", kind="k"))
        await asyncio.sleep(0.02)
        await runner.shutdown(timeout=2)
        for tid in ("a", "b"):
            rec = registry.get(tid)
            assert rec.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self):
        registry = InMemoryRegistry()
        runner = BackgroundTaskRunner(registry, {})
        await runner.shutdown()
        await runner.shutdown()  # no error


class TestStart:
    @pytest.mark.asyncio
    async def test_start_marks_orphaned_running_as_failed(self):
        registry = InMemoryRegistry()
        # Pre-seed a leftover RUNNING task (simulates crashed previous proc).
        record = TaskRecord(task_id="orphan", kind="k")
        registry.register(record)
        registry.update_status("orphan", TaskStatus.RUNNING)
        runner = BackgroundTaskRunner(registry, {})
        await runner.start()
        rec = registry.get("orphan")
        assert rec.status == TaskStatus.FAILED
        assert rec.error == "restarted_during_run"

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        registry = InMemoryRegistry()
        runner = BackgroundTaskRunner(registry, {})
        await runner.start()
        await runner.start()  # no error


class TestConcurrencyLimit:
    @pytest.mark.asyncio
    async def test_max_concurrent_respected(self):
        # 5 long-running tasks but max_concurrent=2 → only 2 reach
        # RUNNING immediately; the rest stay PENDING until a slot frees.
        registry = InMemoryRegistry()
        runner = BackgroundTaskRunner(
            registry,
            {"k": _StubExecutor([b"x"], delay=0.2)},
            max_concurrent=2,
        )
        for i in range(5):
            await runner.submit(TaskRecord(task_id=f"t{i}", kind="k"))
        await asyncio.sleep(0.05)
        running = registry.list_filtered(TaskFilter(status=TaskStatus.RUNNING))
        assert len(running) == 2
        # Drain.
        for _ in range(200):
            done = registry.list_filtered(TaskFilter(status=TaskStatus.DONE))
            if len(done) == 5:
                break
            await asyncio.sleep(0.05)
        assert len(registry.list_filtered(TaskFilter(status=TaskStatus.DONE))) == 5


# ── LocalBashExecutor ────────────────────────────────────────────────


class TestLocalBashExecutor:
    @pytest.mark.asyncio
    async def test_runs_command_streams_stdout(self):
        executor = LocalBashExecutor()
        record = TaskRecord(
            task_id="t1",
            kind="local_bash",
            payload={"command": "echo hello"},
        )
        chunks = [c async for c in executor.execute(record)]
        assert b"hello" in b"".join(chunks)

    @pytest.mark.asyncio
    async def test_missing_command_raises(self):
        executor = LocalBashExecutor()
        record = TaskRecord(task_id="t1", kind="local_bash", payload={})
        with pytest.raises(ValueError):
            async for _ in executor.execute(record):
                pass

    @pytest.mark.asyncio
    async def test_nonzero_exit_raises(self):
        executor = LocalBashExecutor()
        record = TaskRecord(
            task_id="t1",
            kind="local_bash",
            payload={"command": "exit 7"},
        )
        with pytest.raises(RuntimeError, match="rc=7"):
            async for _ in executor.execute(record):
                pass

    @pytest.mark.asyncio
    async def test_max_output_bytes_enforced(self):
        executor = LocalBashExecutor(max_output_bytes=10)
        record = TaskRecord(
            task_id="t1",
            kind="local_bash",
            payload={"command": "yes hello | head -c 1000"},
        )
        with pytest.raises(RuntimeError, match="max_output_bytes"):
            async for _ in executor.execute(record):
                pass


# ── LocalAgentExecutor ───────────────────────────────────────────────


class _FakeOrch:
    def __init__(self, response="ok"):
        self.response = response
        self.last_call = None

    async def run_subagent(self, subagent_type, prompt, *, model=None):
        self.last_call = (subagent_type, prompt, model)
        return self.response


class TestLocalAgentExecutor:
    @pytest.mark.asyncio
    async def test_dispatches_to_orchestrator(self):
        orch = _FakeOrch(response="hello")
        executor = LocalAgentExecutor(lambda: orch)
        record = TaskRecord(
            task_id="t1",
            kind="local_agent",
            payload={"subagent_type": "researcher", "prompt": "go"},
        )
        chunks = [c async for c in executor.execute(record)]
        assert b"hello" in b"".join(chunks)
        assert orch.last_call == ("researcher", "go", None)

    @pytest.mark.asyncio
    async def test_serializes_dict_response_as_json(self):
        executor = LocalAgentExecutor(lambda: _FakeOrch(response={"score": 0.9}))
        record = TaskRecord(
            task_id="t1",
            kind="local_agent",
            payload={"subagent_type": "x", "prompt": ""},
        )
        out = b"".join([c async for c in executor.execute(record)])
        assert b'"score"' in out

    @pytest.mark.asyncio
    async def test_passes_model_override(self):
        orch = _FakeOrch()
        executor = LocalAgentExecutor(lambda: orch)
        record = TaskRecord(
            task_id="t1",
            kind="local_agent",
            payload={"subagent_type": "x", "prompt": "hi", "model": "claude-haiku"},
        )
        async for _ in executor.execute(record):
            pass
        assert orch.last_call[2] == "claude-haiku"

    @pytest.mark.asyncio
    async def test_missing_subagent_type_raises(self):
        executor = LocalAgentExecutor(lambda: _FakeOrch())
        record = TaskRecord(task_id="t1", kind="local_agent", payload={"prompt": "x"})
        with pytest.raises(ValueError):
            async for _ in executor.execute(record):
                pass

    @pytest.mark.asyncio
    async def test_missing_prompt_raises(self):
        executor = LocalAgentExecutor(lambda: _FakeOrch())
        record = TaskRecord(
            task_id="t1",
            kind="local_agent",
            payload={"subagent_type": "x"},
        )
        with pytest.raises(ValueError):
            async for _ in executor.execute(record):
                pass
