"""Loop controllers — Level 2 strategies for loop decision."""

from __future__ import annotations

from abc import abstractmethod
from typing import Optional

from geny_executor.core.stage import Strategy
from geny_executor.core.state import PipelineState


class LoopDecision:
    CONTINUE = "continue"
    COMPLETE = "complete"
    ERROR = "error"
    ESCALATE = "escalate"


class LoopController(Strategy):
    """Base interface for loop control decisions."""

    @abstractmethod
    def decide(self, state: PipelineState) -> str:
        """Decide whether to continue looping.

        Returns: "continue", "complete", "error", or "escalate"
        """
        ...


class StandardLoopController(LoopController):
    """Standard loop controller — tool_use continues, signals decide.

    Logic:
      - If tool results were just produced → continue (need API to process them)
      - If completion signal is "complete" → complete
      - If completion signal is "blocked" → escalate
      - If completion signal is "error" → error
      - If stop_reason is "end_turn" and no tool calls → complete
      - If max iterations reached → force complete
    """

    def __init__(self, max_turns: Optional[int] = None):
        self._max_turns = max_turns

    @property
    def name(self) -> str:
        return "standard"

    @property
    def description(self) -> str:
        return "Standard loop: tool_use continues, signals decide"

    def decide(self, state: PipelineState) -> str:
        # Tool results were just added — need another API call
        if state.tool_results:
            return LoopDecision.CONTINUE

        # Completion signal takes priority
        signal = state.completion_signal
        if signal == "complete":
            return LoopDecision.COMPLETE
        if signal == "blocked":
            return LoopDecision.ESCALATE
        if signal == "error":
            return LoopDecision.ERROR

        # No tool calls and end_turn → complete
        if not state.pending_tool_calls:
            return LoopDecision.COMPLETE

        # Max turns
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
        # Check cost budget
        if state.cost_budget_usd and state.total_cost_usd >= state.cost_budget_usd * self._cost_ratio:
            return LoopDecision.COMPLETE

        # Check token budget
        used = state.token_usage.total_tokens
        if used >= state.context_window_budget * self._token_ratio:
            return LoopDecision.COMPLETE

        # Tool results need processing
        if state.tool_results:
            return LoopDecision.CONTINUE

        # Signal-based
        signal = state.completion_signal
        if signal == "complete":
            return LoopDecision.COMPLETE
        if signal == "blocked":
            return LoopDecision.ESCALATE

        if not state.pending_tool_calls:
            return LoopDecision.COMPLETE

        return LoopDecision.CONTINUE
