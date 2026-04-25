"""Default artifact controllers for Stage 13: Loop."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import List, Optional

from geny_executor.core.state import PipelineState
from geny_executor.stages.s16_loop.interface import LoopController, LoopDecision

logger = logging.getLogger(__name__)


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


# ─────────────────────────────────────────────────────────────────
# Phase 7 Sprint S7.7 — Multi-dimensional budget
# ─────────────────────────────────────────────────────────────────


class BudgetDimension(ABC):
    """One dimension of a multi-dimensional loop budget.

    Subclasses inspect ``state`` and return ``True`` when their budget
    has been exhausted. ``MultiDimensionalBudgetController`` consults
    every registered dimension and stops the loop the moment any one
    of them returns True. ``name`` is surfaced in the
    ``loop.budget_exceeded`` event payload so admins can see *which*
    budget tripped.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable short identifier used in logs + events."""
        ...

    @property
    def description(self) -> str:
        """Human-readable summary; used in stage descriptions."""
        return self.name

    @abstractmethod
    def is_exceeded(self, state: PipelineState) -> bool:
        """Return True when this dimension has been exhausted."""
        ...


class IterationBudget(BudgetDimension):
    """Hard cap on loop iterations."""

    def __init__(self, max_iterations: int):
        self._max = max(1, int(max_iterations))

    @property
    def name(self) -> str:
        return "iteration"

    @property
    def description(self) -> str:
        return f"≤ {self._max} iterations"

    def is_exceeded(self, state: PipelineState) -> bool:
        return state.iteration >= self._max


class CostBudget(BudgetDimension):
    """Soft cap on cumulative USD cost.

    ``threshold_ratio`` lets hosts stop short of the absolute ceiling
    so the next turn doesn't blow past it. Default ``0.9`` matches
    the legacy ``BudgetAwareLoopController``.
    """

    def __init__(self, max_usd: float, *, threshold_ratio: float = 0.9):
        self._max_usd = float(max_usd)
        self._ratio = float(threshold_ratio)

    @property
    def name(self) -> str:
        return "cost"

    @property
    def description(self) -> str:
        return f"≤ ${self._max_usd:.2f} (stop at {self._ratio:.0%})"

    def is_exceeded(self, state: PipelineState) -> bool:
        if self._max_usd <= 0:
            return False
        return state.total_cost_usd >= self._max_usd * self._ratio


class TokenBudget(BudgetDimension):
    """Soft cap on the context window.

    Reads the cumulative ``token_usage.total_tokens`` and compares
    against ``state.context_window_budget`` (the model's window) by
    default, OR an explicit ``max_tokens`` override.
    """

    def __init__(
        self,
        *,
        max_tokens: Optional[int] = None,
        threshold_ratio: float = 0.85,
    ):
        self._max_tokens = int(max_tokens) if max_tokens is not None else None
        self._ratio = float(threshold_ratio)

    @property
    def name(self) -> str:
        return "tokens"

    @property
    def description(self) -> str:
        if self._max_tokens is not None:
            return f"≤ {self._max_tokens} tokens (stop at {self._ratio:.0%})"
        return f"≤ context_window_budget (stop at {self._ratio:.0%})"

    def is_exceeded(self, state: PipelineState) -> bool:
        used = state.token_usage.total_tokens
        cap = self._max_tokens if self._max_tokens is not None else state.context_window_budget
        if cap <= 0:
            return False
        return used >= cap * self._ratio


class WallClockBudget(BudgetDimension):
    """Cap on real time elapsed since session start.

    Reads ``state.created_at`` (set at PipelineState construction)
    so the budget covers everything from session creation through
    the current loop check — including any time the host spent
    setting things up before run().
    """

    def __init__(
        self,
        max_seconds: float,
        *,
        clock: Optional[callable] = None,
    ):
        self._max_seconds = float(max_seconds)
        # ``clock`` is injectable for deterministic tests.
        self._clock = clock or time.monotonic
        # Capture a startup-relative origin in case the state's
        # ``created_at`` (datetime) drifts under clock skew. We
        # compare against monotonic time deltas at evaluation time.
        self._origin = self._clock()

    @property
    def name(self) -> str:
        return "wall_clock"

    @property
    def description(self) -> str:
        return f"≤ {self._max_seconds:.1f}s wall clock"

    def is_exceeded(self, state: PipelineState) -> bool:
        if self._max_seconds <= 0:
            return False
        elapsed = self._clock() - self._origin
        return elapsed >= self._max_seconds


class ToolCallBudget(BudgetDimension):
    """Hard cap on cumulative tool calls executed.

    Counts entries in ``state.tool_results`` (Stage 10 appends each
    completed call). Useful for guarding against runaway tool
    invocation in agent-driven sessions.
    """

    def __init__(self, max_calls: int):
        self._max = max(1, int(max_calls))

    @property
    def name(self) -> str:
        return "tool_calls"

    @property
    def description(self) -> str:
        return f"≤ {self._max} tool calls"

    def is_exceeded(self, state: PipelineState) -> bool:
        return len(state.tool_results) >= self._max


class MultiDimensionalBudgetController(LoopController):
    """Loop controller backed by a list of pluggable budget dimensions.

    Cycle 20260424 executor uplift — Phase 7 Sprint S7.7.

    The pre-S7.7 :class:`BudgetAwareLoopController` hard-coded two
    dimensions (cost + tokens) at fixed ratios. Hosts that needed a
    third (wall-clock for SLA, tool-call count for spam guards) had
    to subclass or fork.

    The multi-dimensional controller flips that around: build a list
    of :class:`BudgetDimension` instances, pass them in, and the
    controller stops the loop the moment ANY one of them reports
    exceeded. The active dimension's name lands in the
    ``loop.budget_exceeded`` event so admin UIs can render
    "stopped because: tokens" etc.

    When no dimension trips, the controller delegates to standard
    signal-driven loop logic (matches ``StandardLoopController``):
    pending tool results → continue, ``complete`` signal → complete,
    ``blocked`` → escalate, no pending tool calls → complete.

    An empty dimension list is allowed — the controller behaves like
    ``StandardLoopController`` in that case.
    """

    def __init__(self, dimensions: Optional[List[BudgetDimension]] = None):
        self._dimensions: List[BudgetDimension] = list(dimensions or [])
        self._last_exceeded: Optional[str] = None

    @property
    def name(self) -> str:
        return "multi_dim_budget"

    @property
    def description(self) -> str:
        if not self._dimensions:
            return "Multi-dim budget (no dimensions registered)"
        return "Multi-dim budget: " + ", ".join(d.description for d in self._dimensions)

    @property
    def dimensions(self) -> List[BudgetDimension]:
        """Defensive copy of the registered dimensions in declared order."""
        return list(self._dimensions)

    @property
    def last_exceeded_dimension(self) -> Optional[str]:
        """Name of the most recently exceeded dimension, or ``None``.

        Useful for downstream observability (events, audit logs)
        without re-walking the dimension list.
        """
        return self._last_exceeded

    def add(self, dimension: BudgetDimension) -> "MultiDimensionalBudgetController":
        """Append a dimension and return self for fluent composition."""
        self._dimensions.append(dimension)
        return self

    def decide(self, state: PipelineState) -> str:
        # Walk dimensions in declared order; first exceeded wins.
        for dim in self._dimensions:
            try:
                if dim.is_exceeded(state):
                    self._last_exceeded = dim.name
                    logger.info(
                        "MultiDimensionalBudgetController: %s exhausted — stopping loop",
                        dim.name,
                    )
                    return LoopDecision.COMPLETE
            except Exception:
                # A broken dimension must not crash the loop —
                # log + skip; if all dimensions are broken the
                # controller falls through to the default signal
                # logic, which is the safe fallback.
                logger.warning(
                    "MultiDimensionalBudgetController: dim %r raised; skipping",
                    getattr(dim, "name", "?"),
                    exc_info=True,
                )
        self._last_exceeded = None

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

        return LoopDecision.CONTINUE
