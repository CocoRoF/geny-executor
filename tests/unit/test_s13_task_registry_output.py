"""Output-streaming + filtering tests for TaskRegistry (PR-A.1.1).

Augments ``test_s9b2_task_registry.py`` with the new TaskFilter +
output streaming surface introduced by the new-executor-uplift cycle A.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from geny_executor.stages.s13_task_registry import (
    InMemoryRegistry,
    TaskFilter,
    TaskRecord,
    TaskStatus,
)


# ── TaskFilter / list_filtered ────────────────────────────────────────


class TestListFiltered:
    def test_no_filter_returns_all_sorted_desc(self):
        reg = InMemoryRegistry()
        a = TaskRecord(task_id="a", created_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
        b = TaskRecord(task_id="b", created_at=datetime(2026, 2, 1, tzinfo=timezone.utc))
        reg.register(a)
        reg.register(b)
        rows = reg.list_filtered(TaskFilter())
        assert [r.task_id for r in rows] == ["b", "a"]

    def test_filter_by_status(self):
        reg = InMemoryRegistry()
        reg.register(TaskRecord(task_id="p"))
        running = TaskRecord(task_id="r")
        running.mark(TaskStatus.RUNNING)
        reg.register(running)
        out = reg.list_filtered(TaskFilter(status=TaskStatus.RUNNING))
        assert [r.task_id for r in out] == ["r"]

    def test_filter_by_kind(self):
        reg = InMemoryRegistry()
        reg.register(TaskRecord(task_id="x", kind="local_bash"))
        reg.register(TaskRecord(task_id="y", kind="local_agent"))
        out = reg.list_filtered(TaskFilter(kind="local_agent"))
        assert [r.task_id for r in out] == ["y"]

    def test_filter_by_created_after(self):
        reg = InMemoryRegistry()
        reg.register(TaskRecord(task_id="old", created_at=datetime(2020, 1, 1, tzinfo=timezone.utc)))
        reg.register(TaskRecord(task_id="new", created_at=datetime(2026, 1, 1, tzinfo=timezone.utc)))
        cutoff = datetime(2025, 1, 1, tzinfo=timezone.utc)
        out = reg.list_filtered(TaskFilter(created_after=cutoff))
        assert [r.task_id for r in out] == ["new"]

    def test_filter_limit_caps_result(self):
        reg = InMemoryRegistry()
        for i in range(5):
            reg.register(TaskRecord(
                task_id=f"t{i}",
                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=i),
            ))
        out = reg.list_filtered(TaskFilter(limit=2))
        assert len(out) == 2
        assert out[0].task_id == "t4"  # most recent first
        assert out[1].task_id == "t3"

    def test_filter_combines_with_and_semantics(self):
        reg = InMemoryRegistry()
        reg.register(TaskRecord(task_id="a", kind="bash"))  # pending
        running_bash = TaskRecord(task_id="b", kind="bash")
        running_bash.mark(TaskStatus.RUNNING)
        reg.register(running_bash)
        running_agent = TaskRecord(task_id="c", kind="agent")
        running_agent.mark(TaskStatus.RUNNING)
        reg.register(running_agent)
        out = reg.list_filtered(TaskFilter(kind="bash", status=TaskStatus.RUNNING))
        assert [r.task_id for r in out] == ["b"]


# ── append_output / read_output ───────────────────────────────────────


class TestOutputBuffer:
    @pytest.mark.asyncio
    async def test_append_then_read_full(self):
        reg = InMemoryRegistry()
        reg.register(TaskRecord(task_id="t1"))
        await reg.append_output("t1", b"hello ")
        await reg.append_output("t1", b"world")
        assert await reg.read_output("t1") == b"hello world"

    @pytest.mark.asyncio
    async def test_read_with_offset(self):
        reg = InMemoryRegistry()
        reg.register(TaskRecord(task_id="t1"))
        await reg.append_output("t1", b"hello world")
        assert await reg.read_output("t1", offset=6) == b"world"

    @pytest.mark.asyncio
    async def test_read_with_offset_and_limit(self):
        reg = InMemoryRegistry()
        reg.register(TaskRecord(task_id="t1"))
        await reg.append_output("t1", b"abcdefghij")
        assert await reg.read_output("t1", offset=2, limit=3) == b"cde"

    @pytest.mark.asyncio
    async def test_read_offset_beyond_len_returns_empty(self):
        reg = InMemoryRegistry()
        reg.register(TaskRecord(task_id="t1"))
        await reg.append_output("t1", b"abc")
        assert await reg.read_output("t1", offset=99) == b""

    @pytest.mark.asyncio
    async def test_read_unknown_task_returns_empty(self):
        reg = InMemoryRegistry()
        assert await reg.read_output("ghost") == b""

    @pytest.mark.asyncio
    async def test_append_empty_chunk_noop(self):
        reg = InMemoryRegistry()
        reg.register(TaskRecord(task_id="t1"))
        await reg.append_output("t1", b"abc")
        await reg.append_output("t1", b"")  # no-op
        assert await reg.read_output("t1") == b"abc"

    @pytest.mark.asyncio
    async def test_remove_clears_output(self):
        reg = InMemoryRegistry()
        reg.register(TaskRecord(task_id="t1"))
        await reg.append_output("t1", b"data")
        assert reg.remove("t1") is True
        assert await reg.read_output("t1") == b""


# ── stream_output ─────────────────────────────────────────────────────


class TestStreamOutput:
    @pytest.mark.asyncio
    async def test_stream_yields_existing_then_returns_when_terminal(self):
        reg = InMemoryRegistry()
        reg.register(TaskRecord(task_id="t1"))
        await reg.append_output("t1", b"chunk1")
        await reg.append_output("t1", b"chunk2")
        reg.update_status("t1", TaskStatus.DONE, result="ok")

        chunks = []
        async for c in reg.stream_output("t1"):
            chunks.append(c)
        assert b"".join(chunks) == b"chunk1chunk2"

    @pytest.mark.asyncio
    async def test_stream_wakes_on_append(self):
        reg = InMemoryRegistry()
        reg.register(TaskRecord(task_id="t1"))

        async def producer():
            await asyncio.sleep(0.05)
            await reg.append_output("t1", b"first")
            await asyncio.sleep(0.05)
            await reg.append_output("t1", b"second")
            await asyncio.sleep(0.05)
            reg.update_status("t1", TaskStatus.DONE)

        async def consumer():
            chunks = []
            async for c in reg.stream_output("t1"):
                chunks.append(c)
            return chunks

        producer_task = asyncio.create_task(producer())
        consumer_chunks = await asyncio.wait_for(consumer(), timeout=2.0)
        await producer_task
        assert b"".join(consumer_chunks) == b"firstsecond"

    @pytest.mark.asyncio
    async def test_stream_returns_immediately_when_already_terminal(self):
        reg = InMemoryRegistry()
        reg.register(TaskRecord(task_id="t1"))
        reg.update_status("t1", TaskStatus.FAILED, error="boom")
        chunks = [c async for c in reg.stream_output("t1")]
        assert chunks == []

    @pytest.mark.asyncio
    async def test_stream_unknown_task_returns_immediately(self):
        reg = InMemoryRegistry()
        chunks = [c async for c in reg.stream_output("ghost")]
        assert chunks == []

    @pytest.mark.asyncio
    async def test_stream_drains_tail_written_during_terminal_transition(self):
        reg = InMemoryRegistry()
        reg.register(TaskRecord(task_id="t1"))

        async def producer():
            await asyncio.sleep(0.02)
            await reg.append_output("t1", b"early")
            # Append + terminal in the same scheduling slice — consumer
            # must still see the trailing bytes via the drain.
            await reg.append_output("t1", b"late")
            reg.update_status("t1", TaskStatus.DONE)

        producer_task = asyncio.create_task(producer())
        chunks = [c async for c in reg.stream_output("t1")]
        await producer_task
        assert b"".join(chunks) == b"earlylate"


# ── TaskRecord.output_path ────────────────────────────────────────────


class TestTaskRecordOutputPath:
    def test_default_output_path_none(self):
        assert TaskRecord(task_id="t1").output_path is None

    def test_to_dict_includes_output_path(self):
        r = TaskRecord(task_id="t1", output_path="/var/tasks/t1.bin")
        assert r.to_dict()["output_path"] == "/var/tasks/t1.bin"
