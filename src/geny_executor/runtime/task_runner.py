"""Background task runner (PR-A.1.3).

:class:`BackgroundTaskRunner` is the framework-runtime layer that
turns a queue of :class:`TaskRecord` submissions into actual work.
It owns the :class:`asyncio.Task` futures, applies a concurrency
limit, talks to a :class:`TaskRegistry` for state + output
persistence, and supports cooperative cancellation + graceful
shutdown.

Usage:

    runner = BackgroundTaskRunner(
        registry=registry,
        executors={
            "local_bash":  LocalBashExecutor(),
            "local_agent": LocalAgentExecutor(orchestrator_factory),
        },
        max_concurrent=8,
    )
    await runner.start()      # warm-up: any RUNNING tasks left over
                              # from a crash get marked FAILED
    task_id = await runner.submit(record)
    ...
    await runner.shutdown(timeout=30)

The runner is service-instantiated (FastAPI lifespan, CLI bootstrap).
The pipeline never directly mutates it; tools talk to the registry
which the runner observes.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Optional

from geny_executor.runtime.task_executors import BackgroundTaskExecutor
from geny_executor.stages.s13_task_registry.interface import TaskRegistry
from geny_executor.stages.s13_task_registry.types import (
    TaskFilter,
    TaskRecord,
    TaskStatus,
)

logger = logging.getLogger(__name__)


class BackgroundTaskRunner:
    """Asyncio-based background runner for :class:`TaskRecord` work.

    Lifecycle:
        1. ``start()`` — re-attach pass: any RUNNING records left over
           from a crash get marked FAILED (so consumers don't wait
           forever for a future that never resumed).
        2. ``submit(record)`` — registers + schedules. Returns the
           task_id. Idempotent if the same id is re-submitted while
           still in flight (returns existing id, no double-run).
        3. ``stop(task_id)`` — cancels the in-flight asyncio.Task
           cooperatively. The executor's exception handling marks
           the record CANCELLED.
        4. ``shutdown(timeout)`` — cancel everything, wait up to
           timeout. Pending tasks transition to CANCELLED. Idempotent.
    """

    def __init__(
        self,
        registry: TaskRegistry,
        executors: Dict[str, BackgroundTaskExecutor],
        *,
        max_concurrent: int = 8,
    ) -> None:
        self._registry = registry
        self._executors = dict(executors)
        self._sem = asyncio.Semaphore(max_concurrent)
        self._futures: Dict[str, asyncio.Task] = {}
        self._started = False

    async def start(self) -> None:
        """Re-attach pass for crash recovery. Safe to call once."""
        if self._started:
            return
        # Sweep any tasks left in RUNNING from a previous process.
        # Without this, downstream consumers (TaskGet polling) would
        # never see them resolve.
        running = self._registry.list_filtered(TaskFilter(status=TaskStatus.RUNNING))
        for record in running:
            self._registry.update_status(
                record.task_id,
                TaskStatus.FAILED,
                error="restarted_during_run",
            )
        self._started = True

    async def submit(self, record: TaskRecord) -> str:
        """Register + schedule. Returns the task_id."""
        existing = self._futures.get(record.task_id)
        if existing is not None and not existing.done():
            return record.task_id
        executor = self._executors.get(record.kind)
        if executor is None:
            self._registry.register(record)
            self._registry.update_status(
                record.task_id,
                TaskStatus.FAILED,
                error=f"no_executor_for_kind:{record.kind}",
            )
            return record.task_id
        self._registry.register(record)
        task = asyncio.create_task(
            self._run(record, executor),
            name=f"task-{record.task_id}",
        )
        self._futures[record.task_id] = task
        task.add_done_callback(lambda _t, tid=record.task_id: self._futures.pop(tid, None))
        return record.task_id

    async def stop(self, task_id: str) -> bool:
        """Cooperatively cancel the in-flight task. Returns False if
        the task is unknown or already done."""
        task = self._futures.get(task_id)
        if task is None or task.done():
            return False
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        return True

    async def shutdown(self, timeout: float = 30.0) -> None:
        """Cancel all in-flight tasks; wait up to ``timeout``."""
        if not self._futures:
            return
        tasks = list(self._futures.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)
        self._futures.clear()

    # ── Private ──────────────────────────────────────────────────────

    async def _run(
        self,
        record: TaskRecord,
        executor: BackgroundTaskExecutor,
    ) -> None:
        async with self._sem:
            self._registry.update_status(record.task_id, TaskStatus.RUNNING)
            try:
                async for chunk in executor.execute(record):
                    await self._registry.append_output(record.task_id, chunk)
            except asyncio.CancelledError:
                self._registry.update_status(
                    record.task_id,
                    TaskStatus.CANCELLED,
                )
                raise
            except Exception as exc:  # noqa: BLE001 — per-task isolation
                logger.warning(
                    "background_task_failed",
                    extra={"task_id": record.task_id, "kind": record.kind, "error": str(exc)},
                )
                self._registry.update_status(
                    record.task_id,
                    TaskStatus.FAILED,
                    error=str(exc),
                )
                return
            self._registry.update_status(record.task_id, TaskStatus.DONE)


__all__ = ["BackgroundTaskRunner"]
