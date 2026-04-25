"""Stage 20: Persist — interface definitions (S9b.5)."""

from __future__ import annotations

from abc import abstractmethod
from typing import List, Optional

from geny_executor.core.stage import Strategy
from geny_executor.core.state import PipelineState
from geny_executor.stages.s20_persist.types import CheckpointRecord


# state.shared keys.
LAST_CHECKPOINT_KEY = "last_checkpoint"
CHECKPOINT_HISTORY_KEY = "checkpoint_history"


class Persister(Strategy):
    """Write a checkpoint snapshot somewhere durable.

    The :class:`NoPersister` default is a no-op so existing pipelines
    pay zero cost. :class:`FilePersister` writes JSON to a directory.
    Hosts that need Postgres / Redis / S3 plug their own implementation
    — the contract is a single ``write`` method.
    """

    @abstractmethod
    async def write(self, record: CheckpointRecord, state: PipelineState) -> None: ...

    async def read(self, checkpoint_id: str) -> Optional[CheckpointRecord]:
        """Optional read-back. Returns ``None`` when not implemented."""
        return None

    async def list_checkpoints(self, session_id: str = "") -> List[CheckpointRecord]:
        """Optional listing. Returns ``[]`` when not implemented."""
        return []


class FrequencyPolicy(Strategy):
    """Decide whether the stage should write a checkpoint this turn."""

    @abstractmethod
    def should_persist(self, state: PipelineState) -> bool: ...


__all__ = [
    "CHECKPOINT_HISTORY_KEY",
    "FrequencyPolicy",
    "LAST_CHECKPOINT_KEY",
    "Persister",
]
