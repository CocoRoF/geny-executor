"""Stage 13: Task Registry — interface definitions (S9b.2).

The registry stores :class:`TaskRecord` instances; the policy decides
what the stage *does* with newly registered tasks (block until they
finish, fire-and-forget, time-bounded wait, etc.).

Both abstractions are :class:`Strategy` subclasses so they fit into
the existing slot machinery and can be swapped at runtime.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional

from geny_executor.core.stage import Strategy
from geny_executor.core.state import PipelineState
from geny_executor.stages.s13_task_registry.types import TaskRecord, TaskStatus


# State keys (host code or Stage 12 populates these).
PENDING_TASKS_KEY = "tasks_new_this_turn"
TASKS_BY_STATUS_KEY = "tasks_by_status"


class TaskRegistry(Strategy):
    """Storage backend for :class:`TaskRecord` instances."""

    @abstractmethod
    def register(self, record: TaskRecord) -> None:
        """Insert a new record. Re-registering the same task_id replaces it."""
        ...

    @abstractmethod
    def get(self, task_id: str) -> Optional[TaskRecord]:
        """Return the record by id, or None if unknown."""
        ...

    @abstractmethod
    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: Any = None,
        error: Optional[str] = None,
    ) -> Optional[TaskRecord]:
        """Mutate the named task's status. Returns the record (None if unknown)."""
        ...

    @abstractmethod
    def list_all(self) -> List[TaskRecord]:
        """Snapshot of every record currently in the registry."""
        ...

    @abstractmethod
    def remove(self, task_id: str) -> bool:
        """Drop the named task. Returns False if the id is unknown."""
        ...

    def by_status(self) -> Dict[str, List[TaskRecord]]:
        """Group records by status value (string keys)."""
        out: Dict[str, List[TaskRecord]] = {}
        for record in self.list_all():
            out.setdefault(record.status.value, []).append(record)
        return out


class TaskPolicy(Strategy):
    """Decide how Stage 13 handles a freshly-drained batch of tasks.

    Policies receive the just-registered batch (a list of
    :class:`TaskRecord`) plus the live registry. They may mutate task
    statuses, drive synchronous waits, schedule background work, or
    no-op.

    Returning is *advisory* — the stage's behaviour after a policy
    returns is to refresh ``state.shared[TASKS_BY_STATUS_KEY]`` and
    move on. Policies that need to block (e.g. ``EagerWaitPolicy``)
    do so inside :meth:`apply`.
    """

    @abstractmethod
    async def apply(
        self,
        new_tasks: List[TaskRecord],
        registry: TaskRegistry,
        state: PipelineState,
    ) -> None: ...


__all__ = [
    "PENDING_TASKS_KEY",
    "TASKS_BY_STATUS_KEY",
    "TaskPolicy",
    "TaskRegistry",
]
