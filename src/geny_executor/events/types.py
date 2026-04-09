"""Pipeline event types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict


@dataclass
class PipelineEvent:
    """A single event emitted during pipeline execution."""

    type: str
    stage: str = ""
    iteration: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    data: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        parts = [f"type={self.type!r}"]
        if self.stage:
            parts.append(f"stage={self.stage!r}")
        if self.iteration:
            parts.append(f"iter={self.iteration}")
        if self.data:
            keys = ", ".join(self.data.keys())
            parts.append(f"data=[{keys}]")
        return f"PipelineEvent({', '.join(parts)})"
