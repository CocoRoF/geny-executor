"""Default artifact controllers for Stage 13: Loop."""

from __future__ import annotations

from typing import Optional

from geny_executor.core.state import PipelineState
from geny_executor.stages.s13_loop.interface import LoopController, LoopDecision


class StandardLoopController(LoopController):
    """Standard loop controller — tool_use continues, signals decide."""

    def __init__(self, max_turns: Optional[int] = None):
        self._max_turns = max_turns

    @property
    def name(self) -> str:
        return "standard"

    @property
    def description(self) -> str:
        return "Standard loop: tool_use continues, signals decide"

    def decide(self, state: PipelineState) -> str:
        if state.tool_results:
            return LoopDecision.CONTINUE

        signal = state.completion_signal
        if signal == "complete":
            return LoopDecision.COMPLETE
        if signal == "blocked":
            return LoopDecision.ESCALATE
        if signal == "error":
            return LoopDecision.ERROR

        if not state.pending_tool_calls:
            return LoopDecision.COMPLETE

        max_t = self._max_turns or state.max_iterations
        if state.iteration >= max_t:
            return LoopDecision.COMPLETE

        return LoopDecision.CONTINUE


class SingleTurnController(LoopController):
    """Single turn — always complete after one pass."""

    @property
    def name(self) -> str:
        return "single_turn"

    @property
    def description(self) -> str:
        return "Always complete after one turn (no loop)"

    def decide(self, state: PipelineState) -> str:
        return LoopDecision.COMPLETE


class BudgetAwareLoopController(LoopController):
    """Budget-aware — stops if cost/token budget is low."""

    def __init__(self, cost_threshold_ratio: float = 0.9, token_threshold_ratio: float = 0.85):
        self._cost_ratio = cost_threshold_ratio
        self._token_ratio = token_threshold_ratio

    @property
    def name(self) -> str:
        return "budget_aware"

    @property
    def description(self) -> str:
        return "Stops when approaching budget limits"

    def decide(self, state: PipelineState) -> str:
        if (
            state.cost_budget_usd
            and state.total_cost_usd >= state.cost_budget_usd * self._cost_ratio
        ):
            return LoopDecision.COMPLETE

        used = state.token_usage.total_tokens
        if used >= state.context_window_budget * self._token_ratio:
            return LoopDecision.COMPLETE

        if state.tool_results:
            return LoopDecision.CONTINUE

        signal = state.completion_signal
        if signal == "complete":
            return LoopDecision.COMPLETE
        if signal == "blocked":
            return LoopDecision.ESCALATE

        if not state.pending_tool_calls:
            return LoopDecision.COMPLETE

        return LoopDecision.CONTINUE
