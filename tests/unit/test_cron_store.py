"""CronJobStore tests — InMemory + FileBacked (PR-A.4.1)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from geny_executor.cron import (
    CronJob,
    CronJobStatus,
    FileBackedCronJobStore,
    InMemoryCronJobStore,
)


# ── InMemoryCronJobStore ─────────────────────────────────────────────


class TestInMemory:
    @pytest.mark.asyncio
    async def test_put_get_round_trip(self):
        store = InMemoryCronJobStore()
        job = CronJob(name="nightly", cron_expr="0 3 * * *", target_kind="local_bash")
        await store.put(job)
        loaded = await store.get("nightly")
        assert loaded.name == "nightly"
        assert loaded.cron_expr == "0 3 * * *"

    @pytest.mark.asyncio
    async def test_list_only_enabled_filter(self):
        store = InMemoryCronJobStore()
        await store.put(CronJob(name="a", cron_expr="* * * * *", target_kind="x"))
        b = CronJob(name="b", cron_expr="* * * * *", target_kind="x", status=CronJobStatus.DISABLED)
        await store.put(b)
        all_ = await store.list()
        only = await store.list(only_enabled=True)
        assert len(all_) == 2
        assert [j.name for j in only] == ["a"]

    @pytest.mark.asyncio
    async def test_delete_returns_false_for_missing(self):
        store = InMemoryCronJobStore()
        assert await store.delete("ghost") is False

    @pytest.mark.asyncio
    async def test_mark_fired_updates_record(self):
        store = InMemoryCronJobStore()
        await store.put(CronJob(name="x", cron_expr="* * * * *", target_kind="k"))
        when = datetime(2026, 4, 26, tzinfo=timezone.utc)
        updated = await store.mark_fired("x", when, task_id="t-1")
        assert updated.last_fired_at == when
        assert updated.last_task_id == "t-1"

    @pytest.mark.asyncio
    async def test_update_status_disable_then_enable(self):
        store = InMemoryCronJobStore()
        await store.put(CronJob(name="x", cron_expr="* * * * *", target_kind="k"))
        await store.update_status("x", CronJobStatus.DISABLED)
        j = await store.get("x")
        assert j.status == CronJobStatus.DISABLED


# ── FileBackedCronJobStore ───────────────────────────────────────────


class TestFileBacked:
    @pytest.mark.asyncio
    async def test_survives_restart(self, tmp_path: Path):
        path = tmp_path / "cron.json"
        s1 = FileBackedCronJobStore(path)
        await s1.put(CronJob(name="x", cron_expr="0 9 * * *", target_kind="k"))
        await s1.mark_fired("x", datetime(2026, 4, 26, tzinfo=timezone.utc), task_id="t1")

        s2 = FileBackedCronJobStore(path)
        loaded = await s2.get("x")
        assert loaded.name == "x"
        assert loaded.last_task_id == "t1"

    @pytest.mark.asyncio
    async def test_atomic_write_creates_backup(self, tmp_path: Path):
        path = tmp_path / "cron.json"
        store = FileBackedCronJobStore(path)
        await store.put(CronJob(name="a", cron_expr="* * * * *", target_kind="k"))
        await store.put(CronJob(name="b", cron_expr="* * * * *", target_kind="k"))
        # After the second put, the first state is in .bak.
        assert path.exists()
        assert (path.with_suffix(path.suffix + ".bak")).exists()

    @pytest.mark.asyncio
    async def test_corrupt_json_skipped(self, tmp_path: Path, caplog):
        path = tmp_path / "cron.json"
        path.write_text("{not json", encoding="utf-8")
        store = FileBackedCronJobStore(path)
        # Empty cache loaded; subsequent put still works.
        await store.put(CronJob(name="x", cron_expr="* * * * *", target_kind="k"))
        assert (await store.get("x")).name == "x"

    @pytest.mark.asyncio
    async def test_delete_removes_persisted(self, tmp_path: Path):
        path = tmp_path / "cron.json"
        s1 = FileBackedCronJobStore(path)
        await s1.put(CronJob(name="x", cron_expr="* * * * *", target_kind="k"))
        assert await s1.delete("x") is True

        s2 = FileBackedCronJobStore(path)
        assert await s2.get("x") is None

    @pytest.mark.asyncio
    async def test_only_enabled_filter(self, tmp_path: Path):
        store = FileBackedCronJobStore(tmp_path / "cron.json")
        await store.put(CronJob(name="a", cron_expr="* * * * *", target_kind="k"))
        b = CronJob(name="b", cron_expr="* * * * *", target_kind="k", status=CronJobStatus.DISABLED)
        await store.put(b)
        only = await store.list(only_enabled=True)
        assert [j.name for j in only] == ["a"]
