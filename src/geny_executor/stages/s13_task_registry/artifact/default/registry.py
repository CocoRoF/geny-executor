"""In-memory task registry backend for Stage 13 (S9b.2)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from geny_executor.stages.s13_task_registry.interface import TaskRegistry
from geny_executor.stages.s13_task_registry.types import TaskRecord, TaskStatus


class InMemoryRegistry(TaskRegistry):
    """Process-lifetime task store.

    Suitable for single-process pipelines. Hosts that need durable
    task state can plug their own :class:`TaskRegistry` (e.g. backed
    by Postgres / Redis) — the policies and the stage don't care
    about the backend.
    """

    def __init__(self) -> None:
        self._records: Dict[str, TaskRecord] = {}

    @property
    def name(self) -> str:
        return "in_memory"

    @property
    def description(self) -> str:
        return "In-memory task registry (process lifetime)"

    def register(self, record: TaskRecord) -> None:
        self._records[record.task_id] = record

    def get(self, task_id: str) -> Optional[TaskRecord]:
        return self._records.get(task_id)

    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: Any = None,
        error: Optional[str] = None,
    ) -> Optional[TaskRecord]:
        record = self._records.get(task_id)
        if record is None:
            return None
        record.mark(status, result=result, error=error)
        return record

    def list_all(self) -> List[TaskRecord]:
        return list(self._records.values())

    def remove(self, task_id: str) -> bool:
        return self._records.pop(task_id, None) is not None


__all__ = ["InMemoryRegistry"]
