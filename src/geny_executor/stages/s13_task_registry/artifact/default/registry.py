"""In-memory task registry backend for Stage 13 (S9b.2)."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

from geny_executor.stages.s13_task_registry.interface import TaskRegistry
from geny_executor.stages.s13_task_registry.types import TaskRecord, TaskStatus


class InMemoryRegistry(TaskRegistry):
    """Process-lifetime task store.

    Suitable for single-process pipelines. Hosts that need durable
    task state can plug their own :class:`TaskRegistry` (e.g. backed
    by Postgres / Redis) — the policies and the stage don't care
    about the backend.

    Output streaming is supported via per-task ``bytearray`` buffers
    plus an ``asyncio.Event`` so :meth:`stream_output` wakes on each
    :meth:`append_output` rather than polling.
    """

    def __init__(self) -> None:
        self._records: Dict[str, TaskRecord] = {}
        self._outputs: Dict[str, bytearray] = {}
        self._output_events: Dict[str, asyncio.Event] = {}

    @property
    def name(self) -> str:
        return "in_memory"

    @property
    def description(self) -> str:
        return "In-memory task registry (process lifetime)"

    def register(self, record: TaskRecord) -> None:
        self._records[record.task_id] = record
        # Buffer is created lazily on first append_output, but we set
        # up the event eagerly so consumers waiting on stream_output
        # before the first chunk arrives don't race.
        self._output_events.setdefault(record.task_id, asyncio.Event())

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
        # On terminal transition, wake any stream_output consumers so
        # they can drain the tail and exit instead of waiting forever.
        if record.is_terminal:
            event = self._output_events.get(task_id)
            if event is not None:
                event.set()
        return record

    def list_all(self) -> List[TaskRecord]:
        return list(self._records.values())

    def remove(self, task_id: str) -> bool:
        existed = self._records.pop(task_id, None) is not None
        self._outputs.pop(task_id, None)
        event = self._output_events.pop(task_id, None)
        if event is not None:
            event.set()
        return existed

    # ── Output streaming ──────────────────────────────────────────────

    async def append_output(self, task_id: str, chunk: bytes) -> None:
        if not chunk:
            return
        buf = self._outputs.setdefault(task_id, bytearray())
        buf.extend(chunk)
        event = self._output_events.setdefault(task_id, asyncio.Event())
        # Wake all current waiters, then immediately rearm so the next
        # append_output can wake fresh waiters without us holding state.
        event.set()
        event.clear()

    async def read_output(
        self,
        task_id: str,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> bytes:
        buf = self._outputs.get(task_id)
        if buf is None or offset >= len(buf):
            return b""
        end = len(buf) if limit is None else min(offset + limit, len(buf))
        return bytes(buf[offset:end])

    async def stream_output(self, task_id: str) -> AsyncIterator[bytes]:
        offset = 0
        while True:
            chunk = await self.read_output(task_id, offset)
            if chunk:
                yield chunk
                offset += len(chunk)
                # Loop back to immediately drain anything else queued.
                continue
            record = self._records.get(task_id)
            if record is None:
                return
            if record.is_terminal:
                # One last drain in case bytes arrived between the read
                # above and the terminal transition.
                tail = await self.read_output(task_id, offset)
                if tail:
                    yield tail
                return
            event = self._output_events.get(task_id)
            if event is None:
                # Record exists but no event registered (manual mutation
                # bypassed register). Bail to avoid hanging.
                return
            try:
                # Cap individual waits so a runaway producer / forgotten
                # terminal transition never hangs the consumer forever.
                await asyncio.wait_for(event.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass


__all__ = ["InMemoryRegistry"]
