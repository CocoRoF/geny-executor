"""Default artifact for Stage 1: Input."""

from geny_executor.stages.s01_input.artifact.default.stage import InputStage
from geny_executor.stages.s01_input.artifact.default.validators import (
    DefaultValidator,
    PassthroughValidator,
    StrictValidator,
    SchemaValidator,
)
from geny_executor.stages.s01_input.artifact.default.normalizers import (
    DefaultNormalizer,
    MultimodalNormalizer,
)

# Convention: every artifact exports ``Stage``
Stage = InputStage

__all__ = [
    "Stage",
    "InputStage",
    "DefaultValidator",
    "PassthroughValidator",
    "StrictValidator",
    "SchemaValidator",
    "DefaultNormalizer",
    "MultimodalNormalizer",
]
