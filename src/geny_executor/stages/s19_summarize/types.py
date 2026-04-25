"""Summary record types for Stage 19 (S9b.4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

from geny_executor.memory.provider import Importance


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class SummaryRecord:
    """One turn's distilled summary. Mutable so importance scorers
    can update the record after the summarizer runs.

    Attributes:
        turn_id: Identifier for the turn this summary covers (typically
            ``f"{session_id}:{iteration}"``).
        abstract: ~3-sentence prose summary.
        key_facts: List of short standalone fact strings (one per
            line in UI renderings).
        entities: List of entity strings mentioned this turn.
        tags: Free-form tag tokens for retrieval.
        importance: :class:`Importance` grade (re-uses the existing
            memory.provider enum so retrieval boosts apply uniformly).
        created_at: Wall-clock timestamp.
    """

    turn_id: str
    abstract: str = ""
    key_facts: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    importance: Importance = Importance.MEDIUM
    created_at: datetime = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "abstract": self.abstract,
            "key_facts": list(self.key_facts),
            "entities": list(self.entities),
            "tags": list(self.tags),
            "importance": self.importance.value,
            "created_at": self.created_at.isoformat(),
        }


__all__ = ["SummaryRecord"]
