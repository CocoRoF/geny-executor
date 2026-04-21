"""Stage 2: Context — default artifact."""

from geny_executor.stages.s02_context.artifact.default.stage import ContextStage
from geny_executor.stages.s02_context.artifact.default.strategies import (
    SimpleLoadStrategy,
    HybridStrategy,
    ProgressiveDisclosureStrategy,
)
from geny_executor.stages.s02_context.artifact.default.compactors import (
    TruncateCompactor,
    SummaryCompactor,
    LLMSummaryCompactor,
    SlidingWindowCompactor,
)
from geny_executor.stages.s02_context.artifact.default.retrievers import (
    NullRetriever,
    StaticRetriever,
)

Stage = ContextStage

__all__ = [
    "Stage",
    "ContextStage",
    "SimpleLoadStrategy",
    "HybridStrategy",
    "ProgressiveDisclosureStrategy",
    "TruncateCompactor",
    "SummaryCompactor",
    "LLMSummaryCompactor",
    "SlidingWindowCompactor",
    "NullRetriever",
    "StaticRetriever",
]
