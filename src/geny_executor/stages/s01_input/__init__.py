"""Stage 1: Input — receive, validate, normalize user input."""

# Interfaces (ABCs)
from geny_executor.stages.s01_input.interface import InputValidator, InputNormalizer

# Types (shared)
from geny_executor.stages.s01_input.types import NormalizedInput

# Default artifact (backward-compatible)
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

__all__ = [
    "InputStage",
    "InputValidator",
    "DefaultValidator",
    "PassthroughValidator",
    "StrictValidator",
    "SchemaValidator",
    "InputNormalizer",
    "DefaultNormalizer",
    "MultimodalNormalizer",
    "NormalizedInput",
]
