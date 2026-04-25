"""Stage 13: Task Registry — interface definitions (S9b.2).

The registry stores :class:`TaskRecord` instances; the policy decides
what the stage *does* with newly registered tasks (block until they
finish, fire-and-forget, time-bounded wait, etc.).

Both abstractions are :class:`Strategy` subclasses so they fit into
the existing slot machinery and can be swapped at runtime.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional

from geny_executor.core.stage import Strategy
from geny_executor.core.state import PipelineState
from geny_executor.stages.s13_task_registry.types import (
    TaskFilter,
    TaskRecord,
    TaskStatus,
)


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

    # ── Optional: filtering + streaming output ────────────────────────
    #
    # Defaults provide a working in-memory implementation on top of
    # ``list_all``. Backends that persist tasks to disk / DB should
    # override for efficiency. Backends that have no concept of output
    # streams keep the default no-ops; tools that try to ``read_output``
    # will get an empty bytes payload.

    def list_filtered(self, filter: TaskFilter) -> List[TaskRecord]:
        """Return records matching ``filter``, ordered by ``created_at`` desc.

        The default implementation builds on ``list_all`` and is suitable
        for in-memory backends. Persistent backends (Postgres / Redis)
        should override to push the filter into the query layer.
        """
        rows = self.list_all()
        if filter.status is not None:
            rows = [r for r in rows if r.status == filter.status]
        if filter.kind is not None:
            rows = [r for r in rows if r.kind == filter.kind]
        if filter.created_after is not None:
            rows = [r for r in rows if r.created_at >= filter.created_after]
        rows.sort(key=lambda r: r.created_at, reverse=True)
        if filter.limit is not None:
            rows = rows[: filter.limit]
        return rows

    async def append_output(self, task_id: str, chunk: bytes) -> None:
        """Append output bytes for a task. No-op by default.

        Backends override to persist to memory / disk / blob storage.
        ``chunk`` may be partial — callers may invoke this many times
        per task.
        """
        return None

    async def read_output(
        self,
        task_id: str,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> bytes:
        """Return previously appended output bytes from ``offset``.

        Returns ``b""`` when no output is recorded for ``task_id``
        (or when the backend does not support output storage).
        """
        return b""

    async def stream_output(self, task_id: str) -> AsyncIterator[bytes]:
        """Async-iterate output chunks until the task reaches a terminal status.

        The default implementation yields what is currently buffered
        and returns once the record is terminal. Backends that maintain
        an ``asyncio.Event`` per task should override to deliver chunks
        as they arrive (no polling).
        """
        offset = 0
        while True:
            chunk = await self.read_output(task_id, offset)
            if chunk:
                yield chunk
                offset += len(chunk)
            record = self.get(task_id)
            if record is None or record.is_terminal:
                # Drain any final bytes the producer wrote between the
                # last read and the terminal transition.
                tail = await self.read_output(task_id, offset)
                if tail:
                    yield tail
                return


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
