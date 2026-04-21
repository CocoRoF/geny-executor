"""Stage 2: Context — collect history, memory, references."""

from geny_executor.stages.s02_context.interface import (
    ContextStrategy,
    HistoryCompactor,
    MemoryRetriever,
)
from geny_executor.stages.s02_context.types import MemoryChunk
from geny_executor.stages.s02_context.artifact.default import (
    ContextStage,
    SimpleLoadStrategy,
    HybridStrategy,
    ProgressiveDisclosureStrategy,
    TruncateCompactor,
    SummaryCompactor,
    LLMSummaryCompactor,
    SlidingWindowCompactor,
    NullRetriever,
    StaticRetriever,
)

__all__ = [
    "ContextStage",
    "ContextStrategy",
    "SimpleLoadStrategy",
    "HybridStrategy",
    "ProgressiveDisclosureStrategy",
    "HistoryCompactor",
    "TruncateCompactor",
    "SummaryCompactor",
    "LLMSummaryCompactor",
    "SlidingWindowCompactor",
    "MemoryChunk",
    "MemoryRetriever",
    "NullRetriever",
    "StaticRetriever",
]
