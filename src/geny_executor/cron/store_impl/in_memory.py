"""InMemoryCronJobStore — process-lifetime store (dev / single-tenant test)."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Dict, List, Optional

from geny_executor.cron.store_abc import CronJobStore
from geny_executor.cron.types import CronJob, CronJobStatus


class InMemoryCronJobStore(CronJobStore):
    def __init__(self) -> None:
        self._jobs: Dict[str, CronJob] = {}
        self._lock = asyncio.Lock()

    async def put(self, job: CronJob) -> None:
        async with self._lock:
            self._jobs[job.name] = deepcopy(job)

    async def get(self, name: str) -> Optional[CronJob]:
        async with self._lock:
            j = self._jobs.get(name)
            return deepcopy(j) if j else None

    async def list(self, *, only_enabled: bool = False) -> List[CronJob]:
        async with self._lock:
            jobs = list(self._jobs.values())
        if only_enabled:
            jobs = [j for j in jobs if j.status == CronJobStatus.ENABLED]
        return [deepcopy(j) for j in jobs]

    async def delete(self, name: str) -> bool:
        async with self._lock:
            return self._jobs.pop(name, None) is not None

    async def mark_fired(self, name, when, task_id=None) -> Optional[CronJob]:
        async with self._lock:
            j = self._jobs.get(name)
            if j is None:
                return None
            j.last_fired_at = when
            j.last_task_id = task_id
            return deepcopy(j)

    async def update_status(self, name, status) -> Optional[CronJob]:
        async with self._lock:
            j = self._jobs.get(name)
            if j is None:
                return None
            j.status = status
            return deepcopy(j)


__all__ = ["InMemoryCronJobStore"]
