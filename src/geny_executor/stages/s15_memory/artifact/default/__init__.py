"""Default artifact for Stage 15: Memory."""

from geny_executor.stages.s15_memory.artifact.default.stage import MemoryStage
from geny_executor.stages.s15_memory.artifact.default.strategies import (
    AppendOnlyStrategy,
    NoMemoryStrategy,
    ReflectiveStrategy,
)
from geny_executor.stages.s15_memory.artifact.default.persistence import (
    InMemoryPersistence,
    FilePersistence,
)

Stage = MemoryStage

__all__ = [
    "Stage",
    "MemoryStage",
    "AppendOnlyStrategy",
    "NoMemoryStrategy",
    "ReflectiveStrategy",
    "InMemoryPersistence",
    "FilePersistence",
]
