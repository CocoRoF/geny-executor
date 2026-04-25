"""Stage 8: Think — interface definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from geny_executor.core.stage import Strategy
from geny_executor.core.state import PipelineState
from geny_executor.stages.s08_think.types import ThinkingBlock


class ThinkingProcessor(Strategy, ABC):
    """Level 2 strategy: how to process thinking content blocks."""

    @abstractmethod
    async def process(
        self,
        thinking_blocks: List[ThinkingBlock],
        state: PipelineState,
    ) -> List[ThinkingBlock]:
        """Process thinking blocks. Return processed blocks."""
        ...


class ThinkingBudgetPlanner(Strategy, ABC):
    """Level 2 strategy: choose ``thinking_budget_tokens`` per turn.

    Runs *before* the API call (Stage 6) — the configured budget is
    written back onto :attr:`PipelineState.thinking_budget_tokens` so
    Stage 6's :meth:`Stage.resolve_model_config` picks it up. Pure
    function of state; planners must not mutate anything other than
    reading state and returning the new budget.

    The default :class:`StaticThinkingBudget` returns a fixed value so
    wiring the slot is safe even when no real adaptive logic is plugged
    in. :class:`AdaptiveThinkingBudget` (also in this artifact) sizes
    the budget based on cheap heuristics: tool advertising, message
    size, recent reflection signals.
    """

    @abstractmethod
    def plan(self, state: PipelineState) -> int:
        """Return the desired ``thinking_budget_tokens`` for this turn."""
        ...
