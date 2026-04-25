"""Default artifact for Stage 20: Persist (S9b.5)."""

from geny_executor.stages.s20_persist.artifact.default.frequencies import (
    EveryNTurnsFrequency,
    EveryTurnFrequency,
    OnSignificantFrequency,
)
from geny_executor.stages.s20_persist.artifact.default.persisters import (
    FilePersister,
    NoPersister,
)
from geny_executor.stages.s20_persist.artifact.default.stage import PersistStage

Stage = PersistStage

__all__ = [
    "EveryNTurnsFrequency",
    "EveryTurnFrequency",
    "FilePersister",
    "NoPersister",
    "OnSignificantFrequency",
    "PersistStage",
    "Stage",
]
