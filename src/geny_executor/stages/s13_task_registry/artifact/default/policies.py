"""Default task policies for Stage 13 (S9b.2).

Three flavours covering the common host shapes:

* :class:`FireAndForgetPolicy` (default) — registers tasks and
  returns immediately. Hosts that drive their own scheduler use
  this; the stage just maintains the registry view.
* :class:`EagerWaitPolicy` — synchronously awaits an
  ``executor`` callable for each new task before returning. The
  callable is supplied by the host at construction time so the
  policy doesn't need to know how tasks are actually run.
* :class:`TimedWaitPolicy` — like ``EagerWaitPolicy`` but bounded
  by ``timeout_seconds``. Tasks that don't finish in time stay
  ``PENDING`` (or ``RUNNING`` if the executor started them) and
  the policy moves on so the loop keeps progressing.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, List, Optional

from geny_executor.core.state import PipelineState
from geny_executor.stages.s13_task_registry.interface import TaskPolicy, TaskRegistry
from geny_executor.stages.s13_task_registry.types import TaskRecord, TaskStatus

logger = logging.getLogger(__name__)

TaskExecutor = Callable[[TaskRecord], Awaitable[Any]]
"""Async callable a host wires to actually run one task. Returns the
final result (treated as success) or raises on failure."""


class FireAndForgetPolicy(TaskPolicy):
    """Register the new tasks; do nothing else."""

    @property
    def name(self) -> str:
        return "fire_and_forget"

    @property
    def description(self) -> str:
        return "Register tasks and return immediately"

    async def apply(
        self,
        new_tasks: List[TaskRecord],
        registry: TaskRegistry,
        state: PipelineState,
    ) -> None:
        return None


class EagerWaitPolicy(TaskPolicy):
    """Synchronously await ``executor(task)`` for each new task."""

    def __init__(self, executor: Optional[TaskExecutor] = None) -> None:
        self._executor = executor

    @property
    def name(self) -> str:
        return "eager_wait"

    @property
    def description(self) -> str:
        return "Synchronously run each new task to completion"

    def configure(self, config: dict) -> None:
        executor = config.get("executor")
        if executor is not None:
            self._executor = executor

    async def apply(
        self,
        new_tasks: List[TaskRecord],
        registry: TaskRegistry,
        state: PipelineState,
    ) -> None:
        if self._executor is None:
            for record in new_tasks:
                logger.warning(
                    "EagerWaitPolicy: no executor wired; leaving %s as PENDING",
                    record.task_id,
                )
            return

        for record in new_tasks:
            registry.update_status(record.task_id, TaskStatus.RUNNING)
            try:
                result = await self._executor(record)
            except Exception as exc:  # noqa: BLE001 — per-task isolation
                logger.warning("Task %s failed: %s", record.task_id, exc)
                registry.update_status(record.task_id, TaskStatus.FAILED, error=str(exc))
                state.add_event(
                    "task.failed",
                    {"task_id": record.task_id, "kind": record.kind, "error": str(exc)},
                )
                continue
            registry.update_status(record.task_id, TaskStatus.DONE, result=result)
            state.add_event(
                "task.done",
                {"task_id": record.task_id, "kind": record.kind},
            )


class TimedWaitPolicy(TaskPolicy):
    """Eager wait, but bounded by ``timeout_seconds`` per task.

    Tasks that exceed the timeout stay ``RUNNING`` (executor still
    in flight) so a later iteration can pick them up. ``task.timeout``
    event is emitted so observers know what happened.
    """

    def __init__(
        self,
        executor: Optional[TaskExecutor] = None,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._executor = executor
        self._timeout = float(timeout_seconds)

    @property
    def name(self) -> str:
        return "timed_wait"

    @property
    def description(self) -> str:
        return "Bounded wait: eager but capped by timeout_seconds"

    def configure(self, config: dict) -> None:
        executor = config.get("executor")
        if executor is not None:
            self._executor = executor
        timeout = config.get("timeout_seconds")
        if timeout is not None:
            if timeout <= 0:
                raise ValueError("timeout_seconds must be positive")
            self._timeout = float(timeout)

    async def apply(
        self,
        new_tasks: List[TaskRecord],
        registry: TaskRegistry,
        state: PipelineState,
    ) -> None:
        if self._executor is None:
            for record in new_tasks:
                logger.warning(
                    "TimedWaitPolicy: no executor wired; leaving %s as PENDING",
                    record.task_id,
                )
            return

        for record in new_tasks:
            registry.update_status(record.task_id, TaskStatus.RUNNING)
            try:
                result = await asyncio.wait_for(self._executor(record), timeout=self._timeout)
            except asyncio.TimeoutError:
                state.add_event(
                    "task.timeout",
                    {
                        "task_id": record.task_id,
                        "kind": record.kind,
                        "timeout_seconds": self._timeout,
                    },
                )
                continue
            except Exception as exc:  # noqa: BLE001 — per-task isolation
                logger.warning("Task %s failed: %s", record.task_id, exc)
                registry.update_status(record.task_id, TaskStatus.FAILED, error=str(exc))
                state.add_event(
                    "task.failed",
                    {"task_id": record.task_id, "kind": record.kind, "error": str(exc)},
                )
                continue
            registry.update_status(record.task_id, TaskStatus.DONE, result=result)
            state.add_event(
                "task.done",
                {"task_id": record.task_id, "kind": record.kind},
            )


__all__ = [
    "EagerWaitPolicy",
    "FireAndForgetPolicy",
    "TaskExecutor",
    "TimedWaitPolicy",
]
