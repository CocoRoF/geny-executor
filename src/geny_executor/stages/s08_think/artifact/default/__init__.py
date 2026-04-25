"""Stage 8: Think — default artifact."""

from geny_executor.stages.s08_think.artifact.default.stage import ThinkStage
from geny_executor.stages.s08_think.artifact.default.processors import (
    PassthroughProcessor,
    ExtractAndStoreProcessor,
    ThinkingFilterProcessor,
)
from geny_executor.stages.s08_think.artifact.default.budget import (
    AdaptiveThinkingBudget,
    StaticThinkingBudget,
    apply_thinking_budget,
    make_planner,
)

Stage = ThinkStage

__all__ = [
    "Stage",
    "ThinkStage",
    "PassthroughProcessor",
    "ExtractAndStoreProcessor",
    "ThinkingFilterProcessor",
    "AdaptiveThinkingBudget",
    "StaticThinkingBudget",
    "apply_thinking_budget",
    "make_planner",
]
