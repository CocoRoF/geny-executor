"""CronJobStore ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from geny_executor.cron.types import CronJob, CronJobStatus


class CronJobStore(ABC):
    @abstractmethod
    async def put(self, job: CronJob) -> None: ...

    @abstractmethod
    async def get(self, name: str) -> Optional[CronJob]: ...

    @abstractmethod
    async def list(self, *, only_enabled: bool = False) -> List[CronJob]: ...

    @abstractmethod
    async def delete(self, name: str) -> bool: ...

    @abstractmethod
    async def mark_fired(
        self,
        name: str,
        when: datetime,
        task_id: Optional[str] = None,
    ) -> Optional[CronJob]: ...

    @abstractmethod
    async def update_status(
        self,
        name: str,
        status: CronJobStatus,
    ) -> Optional[CronJob]: ...


__all__ = ["CronJobStore"]
