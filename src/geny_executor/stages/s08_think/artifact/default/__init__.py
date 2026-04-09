"""Stage 8: Think — default artifact."""

from geny_executor.stages.s08_think.artifact.default.stage import ThinkStage
from geny_executor.stages.s08_think.artifact.default.processors import (
    PassthroughProcessor,
    ExtractAndStoreProcessor,
    ThinkingFilterProcessor,
)

Stage = ThinkStage

__all__ = [
    "Stage",
    "ThinkStage",
    "PassthroughProcessor",
    "ExtractAndStoreProcessor",
    "ThinkingFilterProcessor",
]
