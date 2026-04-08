"""Stage 2: Context — collect history, memory, references."""

from geny_executor.stages.s02_context.stage import ContextStage
from geny_executor.stages.s02_context.strategies import (
    ContextStrategy,
    SimpleLoadStrategy,
    HybridStrategy,
    ProgressiveDisclosureStrategy,
)
from geny_executor.stages.s02_context.compactors import (
    HistoryCompactor,
    TruncateCompactor,
    SummaryCompactor,
    SlidingWindowCompactor,
)
from geny_executor.stages.s02_context.retrievers import (
    MemoryRetriever,
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
    "SlidingWindowCompactor",
    "MemoryRetriever",
    "NullRetriever",
    "StaticRetriever",
]
