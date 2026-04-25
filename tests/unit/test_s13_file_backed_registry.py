"""FileBackedRegistry tests (PR-A.1.2)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from geny_executor.stages.s13_task_registry import (
    FileBackedRegistry,
    TaskFilter,
    TaskRecord,
    TaskStatus,
)


# ── Round trip + persistence ─────────────────────────────────────────


class TestRoundTrip:
    def test_register_then_get(self, tmp_path: Path):
        reg = FileBackedRegistry(tmp_path)
        rec = TaskRecord(task_id="t1", kind="local_bash", payload={"cmd": "ls"})
        reg.register(rec)
        loaded = reg.get("t1")
        assert loaded is not None
        assert loaded.task_id == "t1"
        assert loaded.kind == "local_bash"
        assert loaded.payload == {"cmd": "ls"}

    def test_survives_restart(self, tmp_path: Path):
        reg1 = FileBackedRegistry(tmp_path)
        rec = TaskRecord(task_id="t1", kind="local_agent", payload={"x": 1})
        reg1.register(rec)
        reg1.update_status("t1", TaskStatus.RUNNING)
        reg1.update_status("t1", TaskStatus.DONE, result="ok")

        # Fresh instance — must reload from disk.
        reg2 = FileBackedRegistry(tmp_path)
        loaded = reg2.get("t1")
        assert loaded is not None
        assert loaded.status == TaskStatus.DONE
        assert loaded.result == "ok"

    def test_remove_persists_across_restart(self, tmp_path: Path):
        reg1 = FileBackedRegistry(tmp_path)
        reg1.register(TaskRecord(task_id="t1"))
        assert reg1.remove("t1") is True

        reg2 = FileBackedRegistry(tmp_path)
        assert reg2.get("t1") is None

    def test_re_register_overwrites(self, tmp_path: Path):
        reg = FileBackedRegistry(tmp_path)
        reg.register(TaskRecord(task_id="t1", kind="A"))
        reg.register(TaskRecord(task_id="t1", kind="B"))
        loaded = reg.get("t1")
        assert loaded is not None
        assert loaded.kind == "B"

    def test_remove_unknown_returns_false(self, tmp_path: Path):
        reg = FileBackedRegistry(tmp_path)
        assert reg.remove("ghost") is False

    def test_list_all_after_reload(self, tmp_path: Path):
        reg1 = FileBackedRegistry(tmp_path)
        reg1.register(TaskRecord(task_id="a"))
        reg1.register(TaskRecord(task_id="b"))
        reg1.register(TaskRecord(task_id="c"))
        reg1.remove("b")

        reg2 = FileBackedRegistry(tmp_path)
        ids = sorted(r.task_id for r in reg2.list_all())
        assert ids == ["a", "c"]


# ── Corrupt file tolerance ────────────────────────────────────────────


class TestCorruption:
    def test_corrupt_line_skipped(self, tmp_path: Path, caplog):
        # Pre-seed a registry.jsonl with one good + one corrupt line.
        (tmp_path / "registry.jsonl").write_text(
            json.dumps({"task_id": "good", "kind": "K"}) + "\n"
            "this is not json\n"
            + json.dumps({"task_id": "good2", "kind": "L"}) + "\n",
            encoding="utf-8",
        )
        reg = FileBackedRegistry(tmp_path)
        ids = sorted(r.task_id for r in reg.list_all())
        assert ids == ["good", "good2"]

    def test_bad_record_shape_skipped(self, tmp_path: Path):
        (tmp_path / "registry.jsonl").write_text(
            json.dumps({"missing_task_id": True}) + "\n"
            + json.dumps({"task_id": "ok"}) + "\n",
            encoding="utf-8",
        )
        reg = FileBackedRegistry(tmp_path)
        ids = [r.task_id for r in reg.list_all()]
        assert ids == ["ok"]

    def test_empty_lines_ignored(self, tmp_path: Path):
        (tmp_path / "registry.jsonl").write_text(
            "\n\n" + json.dumps({"task_id": "x"}) + "\n\n",
            encoding="utf-8",
        )
        reg = FileBackedRegistry(tmp_path)
        assert reg.get("x") is not None


# ── Filtering inheritance ─────────────────────────────────────────────


class TestListFilteredOnDisk:
    def test_filter_works_after_reload(self, tmp_path: Path):
        reg1 = FileBackedRegistry(tmp_path)
        reg1.register(TaskRecord(task_id="a", kind="bash"))
        running = TaskRecord(task_id="b", kind="agent")
        running.mark(TaskStatus.RUNNING)
        reg1.register(running)

        reg2 = FileBackedRegistry(tmp_path)
        out = reg2.list_filtered(TaskFilter(status=TaskStatus.RUNNING))
        assert [r.task_id for r in out] == ["b"]


# ── Output streaming on disk ──────────────────────────────────────────


class TestOutputOnDisk:
    @pytest.mark.asyncio
    async def test_append_then_read(self, tmp_path: Path):
        reg = FileBackedRegistry(tmp_path)
        reg.register(TaskRecord(task_id="t1"))
        await reg.append_output("t1", b"hello ")
        await reg.append_output("t1", b"world")
        assert await reg.read_output("t1") == b"hello world"

    @pytest.mark.asyncio
    async def test_output_persists_across_restart(self, tmp_path: Path):
        reg1 = FileBackedRegistry(tmp_path)
        reg1.register(TaskRecord(task_id="t1"))
        await reg1.append_output("t1", b"data1")
        await reg1.append_output("t1", b"data2")

        reg2 = FileBackedRegistry(tmp_path)
        assert await reg2.read_output("t1") == b"data1data2"

    @pytest.mark.asyncio
    async def test_remove_deletes_output_file(self, tmp_path: Path):
        reg = FileBackedRegistry(tmp_path)
        reg.register(TaskRecord(task_id="t1"))
        await reg.append_output("t1", b"data")
        out_path = tmp_path / "outputs" / "t1.bin"
        assert out_path.exists()

        reg.remove("t1")
        assert not out_path.exists()

    @pytest.mark.asyncio
    async def test_read_unknown_task_returns_empty(self, tmp_path: Path):
        reg = FileBackedRegistry(tmp_path)
        assert await reg.read_output("ghost") == b""

    @pytest.mark.asyncio
    async def test_stream_yields_then_completes(self, tmp_path: Path):
        reg = FileBackedRegistry(tmp_path)
        reg.register(TaskRecord(task_id="t1"))
        await reg.append_output("t1", b"chunk1")
        reg.update_status("t1", TaskStatus.DONE)
        chunks = [c async for c in reg.stream_output("t1")]
        assert b"".join(chunks) == b"chunk1"

    @pytest.mark.asyncio
    async def test_stream_wakes_on_late_append(self, tmp_path: Path):
        reg = FileBackedRegistry(tmp_path)
        reg.register(TaskRecord(task_id="t1"))

        async def producer():
            await asyncio.sleep(0.05)
            await reg.append_output("t1", b"late")
            reg.update_status("t1", TaskStatus.DONE)

        producer_task = asyncio.create_task(producer())
        chunks = [c async for c in reg.stream_output("t1")]
        await producer_task
        assert b"".join(chunks) == b"late"

    @pytest.mark.asyncio
    async def test_path_traversal_safety(self, tmp_path: Path):
        reg = FileBackedRegistry(tmp_path)
        reg.register(TaskRecord(task_id="../escape"))
        await reg.append_output("../escape", b"x")
        # Output landed under outputs/ directory, not above it.
        out_files = list((tmp_path / "outputs").glob("*.bin"))
        assert len(out_files) == 1
        assert ".." not in out_files[0].name
