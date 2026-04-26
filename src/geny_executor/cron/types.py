"""Cron record types."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _now() -> datetime:
    return datetime.now(timezone.utc)


class CronJobStatus(str, enum.Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"


@dataclass
class CronJob:
    """A single scheduled job. ``target_kind`` matches a
    BackgroundTaskExecutor key (e.g. ``"local_bash"``,
    ``"local_agent"``); ``payload`` carries the arguments that
    will land on the synthesised :class:`TaskRecord`.
    """

    name: str
    cron_expr: str
    target_kind: str
    payload: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    status: CronJobStatus = CronJobStatus.ENABLED
    created_at: datetime = field(default_factory=_now)
    last_fired_at: Optional[datetime] = None
    last_task_id: Optional[str] = None
    next_fire_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "cron_expr": self.cron_expr,
            "target_kind": self.target_kind,
            "payload": dict(self.payload),
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_fired_at": self.last_fired_at.isoformat() if self.last_fired_at else None,
            "last_task_id": self.last_task_id,
            "next_fire_at": self.next_fire_at.isoformat() if self.next_fire_at else None,
        }


__all__ = ["CronJob", "CronJobStatus"]
