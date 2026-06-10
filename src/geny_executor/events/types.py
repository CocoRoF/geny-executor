"""Pipeline event types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict


@dataclass
class PipelineEvent:
    """A single event emitted during pipeline execution.

    Correlation fields (2.2.0, audit 2026-06-09 §3.2 — hosts running
    several sessions over one process had no way to attribute an event
    to its conversation without wrapping every emit site):

    - ``session_id``: copied from ``PipelineState.session_id`` (the
      host-assigned conversation id; empty when the host never set one).
    - ``run_id``: uuid hex minted per ``run()`` / ``run_stream()``
      invocation — one conversational turn. Events from overlapping
      runs on a shared pipeline are distinguishable by this alone.
    - ``seq``: monotonically increasing per-pipeline sequence number,
      assigned at publish time. ``0`` means "never published through a
      pipeline" (e.g. a host-constructed event emitted straight onto a
      bare :class:`EventBus`). The ``Pipeline.events()`` replay cursor
      is expressed in ``seq``.

    All three are dataclass field *appends* with defaults — positional
    construction of the pre-2.2.0 field set keeps working.
    """

    type: str
    stage: str = ""
    iteration: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    data: Dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    run_id: str = ""
    seq: int = 0

    def __repr__(self) -> str:
        parts = [f"type={self.type!r}"]
        if self.seq:
            parts.append(f"seq={self.seq}")
        if self.stage:
            parts.append(f"stage={self.stage!r}")
        if self.iteration:
            parts.append(f"iter={self.iteration}")
        if self.data:
            keys = ", ".join(self.data.keys())
            parts.append(f"data=[{keys}]")
        return f"PipelineEvent({', '.join(parts)})"
