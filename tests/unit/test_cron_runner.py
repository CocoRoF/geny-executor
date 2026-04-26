"""CronRunner tests (PR-A.4.3).

Skips the whole module when ``croniter`` isn't installed — that's the
[cron] extra and CI's [dev] install doesn't pull it. The runner
itself accepts a missing croniter at runtime (logs + skips); the
tests need it because they assert real fire times.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import List

import pytest

pytest.importorskip(
    "croniter",
    reason="cron extra not installed (pip install -e .[cron])",
)

from geny_executor.cron import (  # noqa: E402
    CronJob,
    CronJobStatus,
    CronRunner,
    InMemoryCronJobStore,
)


class _FakeTaskRunner:
    def __init__(self, submit_id: str = "task-1", raise_exc=None):
        self.submitted = []
        self.submit_id = submit_id
        self.raise_exc = raise_exc

    async def submit(self, record):
        if self.raise_exc is not None:
            raise self.raise_exc
        self.submitted.append(record)
        return self.submit_id


# ── Lifecycle ────────────────────────────────────────────────────────


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_then_shutdown(self):
        runner = CronRunner(InMemoryCronJobStore(), _FakeTaskRunner(), cycle_seconds=1)
        await runner.start()
        await asyncio.sleep(0.05)
        await runner.shutdown(timeout=2)

    @pytest.mark.asyncio
    async def test_double_start_is_noop(self):
        runner = CronRunner(InMemoryCronJobStore(), _FakeTaskRunner(), cycle_seconds=1)
        await runner.start()
        await runner.start()  # no error
        await runner.shutdown(timeout=2)

    @pytest.mark.asyncio
    async def test_shutdown_without_start_is_noop(self):
        runner = CronRunner(InMemoryCronJobStore(), _FakeTaskRunner())
        await runner.shutdown(timeout=1)


# ── Firing logic ─────────────────────────────────────────────────────


class TestFire:
    @pytest.mark.asyncio
    async def test_fires_due_job(self):
        store = InMemoryCronJobStore()
        # Past created_at so * * * * * is "due" right now.
        await store.put(CronJob(
            name="every-minute",
            cron_expr="* * * * *",
            target_kind="local_bash",
            created_at=datetime.now(timezone.utc) - timedelta(minutes=2),
        ))
        task_runner = _FakeTaskRunner()
        runner = CronRunner(store, task_runner, cycle_seconds=60)
        fired = await runner.tick_once()
        assert fired == 1
        assert len(task_runner.submitted) == 1
        # Last-fired persisted.
        j = await store.get("every-minute")
        assert j.last_fired_at is not None
        assert j.last_task_id == "task-1"

    @pytest.mark.asyncio
    async def test_idempotent_no_double_fire(self):
        store = InMemoryCronJobStore()
        await store.put(CronJob(
            name="x", cron_expr="* * * * *", target_kind="k",
            created_at=datetime.now(timezone.utc) - timedelta(minutes=2),
        ))
        task_runner = _FakeTaskRunner()
        runner = CronRunner(store, task_runner, cycle_seconds=60)
        await runner.tick_once()
        await runner.tick_once()  # second tick, same minute → no new fire
        assert len(task_runner.submitted) == 1

    @pytest.mark.asyncio
    async def test_disabled_job_not_fired(self):
        store = InMemoryCronJobStore()
        await store.put(CronJob(
            name="x", cron_expr="* * * * *", target_kind="k",
            status=CronJobStatus.DISABLED,
            created_at=datetime.now(timezone.utc) - timedelta(minutes=2),
        ))
        task_runner = _FakeTaskRunner()
        runner = CronRunner(store, task_runner)
        fired = await runner.tick_once()
        assert fired == 0

    @pytest.mark.asyncio
    async def test_invalid_expr_skipped(self):
        store = InMemoryCronJobStore()
        await store.put(CronJob(name="bad", cron_expr="not a cron", target_kind="k"))
        task_runner = _FakeTaskRunner()
        runner = CronRunner(store, task_runner)
        # Should not raise.
        fired = await runner.tick_once()
        assert fired == 0

    @pytest.mark.asyncio
    async def test_submit_failure_does_not_raise(self):
        store = InMemoryCronJobStore()
        await store.put(CronJob(
            name="x", cron_expr="* * * * *", target_kind="k",
            created_at=datetime.now(timezone.utc) - timedelta(minutes=2),
        ))
        task_runner = _FakeTaskRunner(raise_exc=RuntimeError("runner down"))
        runner = CronRunner(store, task_runner)
        # Should not raise — error is logged + we move on.
        await runner.tick_once()

    @pytest.mark.asyncio
    async def test_payload_includes_cron_metadata(self):
        store = InMemoryCronJobStore()
        await store.put(CronJob(
            name="meta-job", cron_expr="* * * * *", target_kind="local_bash",
            payload={"command": "echo hi"},
            created_at=datetime.now(timezone.utc) - timedelta(minutes=2),
        ))
        task_runner = _FakeTaskRunner()
        runner = CronRunner(store, task_runner)
        await runner.tick_once()
        rec = task_runner.submitted[0]
        assert rec.payload["command"] == "echo hi"
        assert rec.payload["_cron_name"] == "meta-job"

    @pytest.mark.asyncio
    async def test_runner_without_submit_logged(self):
        store = InMemoryCronJobStore()
        await store.put(CronJob(
            name="x", cron_expr="* * * * *", target_kind="k",
            created_at=datetime.now(timezone.utc) - timedelta(minutes=2),
        ))
        runner = CronRunner(store, object())
        # Should not raise.
        await runner.tick_once()
