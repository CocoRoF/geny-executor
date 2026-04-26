"""FileBackedCronJobStore — single-file json store with atomic write."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from geny_executor.cron.store_abc import CronJobStore
from geny_executor.cron.types import CronJob, CronJobStatus

logger = logging.getLogger(__name__)


def _decode(data: Dict[str, Any]) -> CronJob:
    def _dt(v):
        if not v:
            return None
        try:
            return datetime.fromisoformat(v)
        except ValueError:
            return None

    return CronJob(
        name=data["name"],
        cron_expr=data["cron_expr"],
        target_kind=data.get("target_kind", ""),
        payload=dict(data.get("payload") or {}),
        description=data.get("description"),
        status=CronJobStatus(data.get("status", CronJobStatus.ENABLED.value)),
        created_at=_dt(data.get("created_at")) or datetime.now(),
        last_fired_at=_dt(data.get("last_fired_at")),
        last_task_id=data.get("last_task_id"),
        next_fire_at=_dt(data.get("next_fire_at")),
    )


class FileBackedCronJobStore(CronJobStore):
    """Atomic-write json store. Layout: ``<path>`` (jobs.json) +
    ``<path>.bak`` (last successful write)."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._cache: Dict[str, CronJob] = {}
        self._loaded = False
        self._lock = asyncio.Lock()

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        async with self._lock:
            if self._loaded:
                return
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if self._path.exists():
                try:
                    data = json.loads(self._path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "cron_store_load_failed",
                        extra={"path": str(self._path), "error": str(exc)},
                    )
                    data = {"jobs": []}
                for raw in data.get("jobs", []):
                    try:
                        j = _decode(raw)
                        self._cache[j.name] = j
                    except (KeyError, ValueError) as exc:
                        logger.warning(
                            "cron_store_skipped_bad_job",
                            extra={"raw": raw, "error": str(exc)},
                        )
            self._loaded = True

    async def _flush(self) -> None:
        # Atomic write: tmp → rename. Backup retained.
        tmp = self._path.with_suffix(".tmp")
        body = {"jobs": [j.to_dict() for j in self._cache.values()]}
        tmp.write_text(json.dumps(body, indent=2), encoding="utf-8")
        if self._path.exists():
            backup = self._path.with_suffix(self._path.suffix + ".bak")
            self._path.replace(backup)
        tmp.replace(self._path)

    async def put(self, job):
        await self._ensure_loaded()
        async with self._lock:
            self._cache[job.name] = job
            await self._flush()

    async def get(self, name):
        await self._ensure_loaded()
        return self._cache.get(name)

    async def list(self, *, only_enabled: bool = False):
        await self._ensure_loaded()
        jobs = list(self._cache.values())
        if only_enabled:
            jobs = [j for j in jobs if j.status == CronJobStatus.ENABLED]
        return jobs

    async def delete(self, name):
        await self._ensure_loaded()
        async with self._lock:
            if name not in self._cache:
                return False
            del self._cache[name]
            await self._flush()
            return True

    async def mark_fired(self, name, when, task_id=None):
        await self._ensure_loaded()
        async with self._lock:
            j = self._cache.get(name)
            if j is None:
                return None
            j.last_fired_at = when
            j.last_task_id = task_id
            await self._flush()
            return j

    async def update_status(self, name, status):
        await self._ensure_loaded()
        async with self._lock:
            j = self._cache.get(name)
            if j is None:
                return None
            j.status = status
            await self._flush()
            return j


__all__ = ["FileBackedCronJobStore"]
