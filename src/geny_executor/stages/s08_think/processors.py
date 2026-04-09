"""Think stage — processors. Backward-compatible re-exports."""

from geny_executor.stages.s08_think.types import ThinkingBlock, ThinkingResult
from geny_executor.stages.s08_think.interface import ThinkingProcessor
from geny_executor.stages.s08_think.artifact.default.processors import (
    PassthroughProcessor,
    ExtractAndStoreProcessor,
    ThinkingFilterProcessor,
)

__all__ = [
    "ThinkingBlock",
    "ThinkingResult",
    "ThinkingProcessor",
    "PassthroughProcessor",
    "ExtractAndStoreProcessor",
    "ThinkingFilterProcessor",
]
