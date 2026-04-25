"""File-backed task registry — single-process durable backend (PR-A.1.2).

Layout under ``root``:

    root/
      registry.jsonl     — one TaskRecord per line (last write wins
                           per ``task_id``)
      outputs/<task_id>.bin
                         — raw appended output bytes

The registry loads ``registry.jsonl`` lazily on first access and
keeps an in-memory cache. Mutations append a fresh JSON line so a
crash-then-restart resumes from the most recent write.

Suitable for self-hosted deployments where Postgres / Redis is
overkill but ``InMemoryRegistry`` loses too much on restart. For
multi-process / clustered deployments, plug a real DB-backed
registry instead.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from geny_executor.stages.s13_task_registry.interface import TaskRegistry
from geny_executor.stages.s13_task_registry.types import TaskRecord, TaskStatus

logger = logging.getLogger(__name__)


def _record_to_jsonable(record: TaskRecord) -> Dict[str, Any]:
    return record.to_dict()


def _record_from_jsonable(data: Dict[str, Any]) -> TaskRecord:
    record = TaskRecord(
        task_id=data["task_id"],
        kind=data.get("kind", ""),
        payload=dict(data.get("payload") or {}),
        status=TaskStatus(data.get("status", TaskStatus.PENDING.value)),
        created_at=_parse_dt(data.get("created_at")) or datetime.now(timezone.utc),
        started_at=_parse_dt(data.get("started_at")),
        completed_at=_parse_dt(data.get("completed_at")),
        result=data.get("result"),
        error=data.get("error"),
        iteration_seen=int(data.get("iteration_seen") or 0),
        output_path=data.get("output_path"),
    )
    return record


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class FileBackedRegistry(TaskRegistry):
    """Durable single-process task registry.

    Mutations append to ``registry.jsonl``; on load, the latest line
    per ``task_id`` wins so corrupted / partial writes at the tail
    are tolerated. Output bytes for each task are stored as a side
    file under ``outputs/<task_id>.bin`` and read with normal file
    seek / read.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._registry_path = self._root / "registry.jsonl"
        self._outputs_dir = self._root / "outputs"
        self._cache: Dict[str, TaskRecord] = {}
        self._loaded = False
        self._mutate_lock = asyncio.Lock()
        self._output_events: Dict[str, asyncio.Event] = {}

    @property
    def name(self) -> str:
        return "file_backed"

    @property
    def description(self) -> str:
        return "Durable single-process task registry (jsonl append + side files for output)"

    # ── Loading ──────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._root.mkdir(parents=True, exist_ok=True)
        self._outputs_dir.mkdir(parents=True, exist_ok=True)
        if self._registry_path.exists():
            for line in self._registry_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "file_backed_registry_skipped_corrupt_line",
                        extra={"path": str(self._registry_path), "error": str(exc)},
                    )
                    continue
                try:
                    record = _record_from_jsonable(data)
                except (KeyError, ValueError) as exc:
                    logger.warning(
                        "file_backed_registry_skipped_bad_record",
                        extra={"path": str(self._registry_path), "error": str(exc)},
                    )
                    continue
                self._cache[record.task_id] = record
        self._loaded = True

    def _append_line(self, record: TaskRecord) -> None:
        line = json.dumps(_record_to_jsonable(record), ensure_ascii=False, default=str)
        with self._registry_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    # ── TaskRegistry protocol ────────────────────────────────────────

    def register(self, record: TaskRecord) -> None:
        self._ensure_loaded()
        self._cache[record.task_id] = record
        self._append_line(record)
        self._output_events.setdefault(record.task_id, asyncio.Event())

    def get(self, task_id: str) -> Optional[TaskRecord]:
        self._ensure_loaded()
        return self._cache.get(task_id)

    def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        *,
        result: Any = None,
        error: Optional[str] = None,
    ) -> Optional[TaskRecord]:
        self._ensure_loaded()
        record = self._cache.get(task_id)
        if record is None:
            return None
        record.mark(status, result=result, error=error)
        self._append_line(record)
        if record.is_terminal:
            event = self._output_events.get(task_id)
            if event is not None:
                event.set()
        return record

    def list_all(self) -> List[TaskRecord]:
        self._ensure_loaded()
        return list(self._cache.values())

    def remove(self, task_id: str) -> bool:
        self._ensure_loaded()
        if task_id not in self._cache:
            return False
        del self._cache[task_id]
        # Tombstone the line so reload doesn't resurrect it.
        with self._registry_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"task_id": task_id, "_deleted": True}) + "\n")
        # Drop side output file.
        out_path = self._output_path_for(task_id)
        if out_path.exists():
            out_path.unlink()
        event = self._output_events.pop(task_id, None)
        if event is not None:
            event.set()
        return True

    # ── Output streaming ─────────────────────────────────────────────

    def _output_path_for(self, task_id: str) -> Path:
        # task_id is registered by callers; we still defang directory traversal
        # in case a backend swap injects untrusted ids later.
        safe = task_id.replace("/", "_").replace("..", "_")
        return self._outputs_dir / f"{safe}.bin"

    async def append_output(self, task_id: str, chunk: bytes) -> None:
        if not chunk:
            return
        self._ensure_loaded()
        path = self._output_path_for(task_id)
        async with self._mutate_lock:
            with path.open("ab") as handle:
                handle.write(chunk)
            event = self._output_events.setdefault(task_id, asyncio.Event())
            event.set()
            event.clear()

    async def read_output(
        self,
        task_id: str,
        offset: int = 0,
        limit: Optional[int] = None,
    ) -> bytes:
        path = self._output_path_for(task_id)
        if not path.exists():
            return b""
        with path.open("rb") as handle:
            handle.seek(offset)
            if limit is None:
                return handle.read()
            return handle.read(limit)

    async def stream_output(self, task_id: str) -> AsyncIterator[bytes]:
        offset = 0
        while True:
            chunk = await self.read_output(task_id, offset)
            if chunk:
                yield chunk
                offset += len(chunk)
                continue
            record = self._cache.get(task_id)
            if record is None:
                return
            if record.is_terminal:
                tail = await self.read_output(task_id, offset)
                if tail:
                    yield tail
                return
            event = self._output_events.get(task_id)
            if event is None:
                return
            try:
                await asyncio.wait_for(event.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass


# Tombstone-aware loader hook: ``_ensure_loaded`` already filters the
# cache by last write, but ``register / update_status`` append a normal
# record line. ``remove`` writes a ``{"_deleted": True}`` line; on
# reload, we apply the tombstone explicitly.

_TOMBSTONE_KEY = "_deleted"


def _load_with_tombstones(path: Path) -> Dict[str, TaskRecord]:
    cache: Dict[str, TaskRecord] = {}
    if not path.exists():
        return cache
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if data.get(_TOMBSTONE_KEY):
            cache.pop(data.get("task_id", ""), None)
            continue
        try:
            record = _record_from_jsonable(data)
        except (KeyError, ValueError):
            continue
        cache[record.task_id] = record
    return cache


# Override _ensure_loaded to use the tombstone-aware loader.
def _ensure_loaded(self: FileBackedRegistry) -> None:
    if self._loaded:
        return
    self._root.mkdir(parents=True, exist_ok=True)
    self._outputs_dir.mkdir(parents=True, exist_ok=True)
    self._cache = _load_with_tombstones(self._registry_path)
    self._loaded = True


FileBackedRegistry._ensure_loaded = _ensure_loaded  # type: ignore[method-assign]


__all__ = ["FileBackedRegistry"]
