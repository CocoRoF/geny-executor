"""CronRunner — asyncio daemon that fires due cron jobs (PR-A.4.3).

The runner polls the :class:`CronJobStore` every ``cycle_seconds``,
computes ``next_fire_at`` for each enabled job via croniter, and
fires due jobs by submitting a :class:`TaskRecord` through the
host's :class:`BackgroundTaskRunner`.

Idempotency: a fire is suppressed when ``last_fired_at >= next_fire_at``
so a refresh during the same minute doesn't double-fire.

Lifecycle:

    runner = CronRunner(store=store, task_runner=task_runner)
    await runner.start()
    ...
    await runner.shutdown(timeout=5)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from geny_executor.cron.store_abc import CronJobStore
from geny_executor.cron.types import CronJob

logger = logging.getLogger(__name__)


class CronRunner:
    def __init__(
        self,
        store: CronJobStore,
        task_runner: object,
        *,
        cycle_seconds: int = 60,
    ) -> None:
        if cycle_seconds <= 0:
            raise ValueError("cycle_seconds must be > 0")
        self._store = store
        self._task_runner = task_runner
        self._cycle = cycle_seconds
        self._stop = asyncio.Event()
        self._daemon: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._daemon is not None:
            return
        self._stop.clear()
        self._daemon = asyncio.create_task(self._loop(), name="cron-daemon")

    async def shutdown(self, timeout: float = 5.0) -> None:
        if self._daemon is None:
            return
        self._stop.set()
        try:
            await asyncio.wait_for(self._daemon, timeout=timeout)
        except asyncio.TimeoutError:
            self._daemon.cancel()
            try:
                await self._daemon
            except (asyncio.CancelledError, Exception):
                pass
        self._daemon = None

    async def refresh(self) -> None:
        """Wake the daemon to re-poll the store. Useful after Create/Delete."""
        # The simple impl: do nothing — the daemon polls every cycle anyway.
        # We could pulse self._stop and re-arm, but that complicates lifecycle.
        # Keep it simple; the next cycle picks up changes.
        return None

    async def tick_once(self) -> int:
        """One iteration of the loop. Returns the number of jobs fired.
        Useful for tests."""
        return await self._fire_due_jobs()

    # ── Internals ────────────────────────────────────────────────────

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._fire_due_jobs()
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "cron_loop_error",
                    extra={"error": str(exc)},
                )
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._cycle)
            except asyncio.TimeoutError:
                continue

    async def _fire_due_jobs(self) -> int:
        now = datetime.now(timezone.utc)
        try:
            jobs = await self._store.list(only_enabled=True)
        except Exception as exc:  # noqa: BLE001
            logger.exception("cron_list_failed", extra={"error": str(exc)})
            return 0
        fired = 0
        for job in jobs:
            try:
                next_fire = self._compute_next_fire(job, now)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "cron_invalid_expr",
                    extra={"job_name": job.name, "expr": job.cron_expr, "error": str(exc)},
                )
                continue
            if next_fire is None:
                continue
            if job.last_fired_at and next_fire <= job.last_fired_at:
                continue
            if next_fire > now:
                continue
            task_id = await self._submit(job, next_fire)
            # Stamp last_fired_at = now (not next_fire) so the next
            # _compute_next_fire starts from the actual fire wall-clock
            # and we don't catch up on missed minutes from a prior
            # outage (a daemon down for an hour would otherwise fire 60
            # times on restart for a `* * * * *` job).
            try:
                await self._store.mark_fired(job.name, now, task_id=task_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "cron_mark_fired_failed",
                    extra={"job_name": job.name, "error": str(exc)},
                )
            fired += 1
            logger.info(
                "cron_fired",
                extra={"job_name": job.name, "next_fire": next_fire.isoformat(), "task_id": task_id},
            )
        return fired

    def _compute_next_fire(
        self, job: CronJob, now: datetime,
    ) -> Optional[datetime]:
        from croniter import croniter  # type: ignore[import-not-found]

        base = job.last_fired_at or job.created_at or now
        # croniter is finicky with tz-aware datetimes — strip TZ for the
        # iteration and re-attach UTC after.
        if base.tzinfo is not None:
            base_naive = base.replace(tzinfo=None)
        else:
            base_naive = base
        it = croniter(job.cron_expr, base_naive)
        nxt = it.get_next(datetime)
        return nxt.replace(tzinfo=timezone.utc) if nxt.tzinfo is None else nxt

    async def _submit(self, job: CronJob, fire_time: datetime) -> Optional[str]:
        """Build a TaskRecord and hand it to the task runner. Returns
        the task_id (or None on failure)."""
        from geny_executor.stages.s13_task_registry.types import TaskRecord

        record = TaskRecord(
            task_id=str(uuid.uuid4()),
            kind=job.target_kind,
            payload={
                **dict(job.payload),
                "_cron_name": job.name,
                "_scheduled_for": fire_time.isoformat(),
            },
        )
        submit = getattr(self._task_runner, "submit", None)
        if submit is None:
            logger.warning(
                "cron_runner_no_submit",
                extra={"job_name": job.name, "task_runner": type(self._task_runner).__name__},
            )
            return None
        try:
            result = submit(record)
            if hasattr(result, "__await__"):
                return await result
            return result
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "cron_submit_failed",
                extra={"job_name": job.name, "error": str(exc)},
            )
            return None


__all__ = ["CronRunner"]
