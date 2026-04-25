"""Checkpoint record types for Stage 20 (S9b.5)."""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_checkpoint_id() -> str:
    return f"ckpt_{secrets.token_urlsafe(8)}"


@dataclass
class CheckpointRecord:
    """A single state-snapshot the persister wrote.

    The ``payload`` is whatever the persister chose to serialise —
    typically a subset of :class:`PipelineState` (messages, shared,
    metadata, iteration counter). Pipelines that want crash-recovery
    plug a persister that knows how to round-trip the payload back
    through :meth:`PipelineState`-like reconstruction; the executor
    itself doesn't yet ship a ``Pipeline.resume_from_checkpoint``
    API (that lands in a follow-up sprint).
    """

    checkpoint_id: str = field(default_factory=_new_checkpoint_id)
    session_id: str = ""
    iteration: int = 0
    created_at: datetime = field(default_factory=_now)
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "session_id": self.session_id,
            "iteration": self.iteration,
            "created_at": self.created_at.isoformat(),
            "payload": dict(self.payload),
        }


__all__ = ["CheckpointRecord"]
