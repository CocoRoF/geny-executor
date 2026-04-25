"""Task record types for Stage 13: Task Registry (S9b.2)."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class TaskStatus(str, enum.Enum):
    """Lifecycle states for a registered task.

    PENDING  — registered but not yet started.
    RUNNING  — execution in progress.
    DONE     — completed successfully.
    FAILED   — completed with an error.
    CANCELLED — externally cancelled before completion.
    """

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TERMINAL_STATUSES = frozenset({TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED})


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class TaskRecord:
    """A single registered task. Mutable so policies can update status in place.

    ``payload`` carries whatever the host wants — typically the
    sub-agent request dict, a SubagentTypeDescriptor argument
    bundle, or a queue item id. ``result`` is filled when the task
    reaches a terminal status.
    """

    task_id: str
    kind: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=_now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    iteration_seen: int = 0

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL_STATUSES

    def mark(self, status: TaskStatus, *, result: Any = None, error: Optional[str] = None) -> None:
        """Transition the task. Sets started_at / completed_at automatically."""
        if status == TaskStatus.RUNNING and self.started_at is None:
            self.started_at = _now()
        if status in _TERMINAL_STATUSES:
            self.completed_at = _now()
            if result is not None:
                self.result = result
            if error is not None:
                self.error = error
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "payload": dict(self.payload),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "iteration_seen": self.iteration_seen,
        }


__all__ = [
    "TaskRecord",
    "TaskStatus",
]
