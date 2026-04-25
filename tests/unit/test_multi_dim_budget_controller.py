"""Phase 7 Sprint S7.7 — MultiDimensionalBudgetController tests."""

from __future__ import annotations

from typing import List

import pytest

from geny_executor.core.state import PipelineState, TokenUsage
from geny_executor.stages.s16_loop import (
    BudgetDimension,
    CostBudget,
    IterationBudget,
    LoopStage,
    MultiDimensionalBudgetController,
    TokenBudget,
    ToolCallBudget,
    WallClockBudget,
)
from geny_executor.stages.s16_loop.interface import LoopDecision


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────


def _state(
    *,
    iteration: int = 0,
    tool_results: list | None = None,
    pending_tool_calls: list | None = None,
    completion_signal: str | None = None,
    total_cost_usd: float = 0.0,
    cost_budget_usd: float | None = None,
    total_tokens: int = 0,
    context_window_budget: int = 200_000,
) -> PipelineState:
    s = PipelineState(session_id="s")
    s.iteration = iteration
    s.tool_results = list(tool_results or [])
    s.pending_tool_calls = list(pending_tool_calls or [])
    s.completion_signal = completion_signal
    s.total_cost_usd = total_cost_usd
    s.cost_budget_usd = cost_budget_usd
    s.token_usage = TokenUsage(input_tokens=total_tokens, output_tokens=0)
    s.context_window_budget = context_window_budget
    return s


# ─────────────────────────────────────────────────────────────────
# Individual dimensions
# ─────────────────────────────────────────────────────────────────


class TestIterationBudget:
    def test_under_cap_not_exceeded(self):
        assert IterationBudget(5).is_exceeded(_state(iteration=4)) is False

    def test_at_cap_exceeded(self):
        assert IterationBudget(5).is_exceeded(_state(iteration=5)) is True

    def test_over_cap_exceeded(self):
        assert IterationBudget(5).is_exceeded(_state(iteration=10)) is True

    def test_clamps_to_min_one(self):
        # Construction with 0 clamps to 1; iteration=0 still under
        b = IterationBudget(0)
        assert b.is_exceeded(_state(iteration=0)) is False
        assert b.is_exceeded(_state(iteration=1)) is True


class TestCostBudget:
    def test_under_threshold_not_exceeded(self):
        b = CostBudget(10.0, threshold_ratio=0.9)
        assert b.is_exceeded(_state(total_cost_usd=8.0)) is False

    def test_at_threshold_exceeded(self):
        b = CostBudget(10.0, threshold_ratio=0.9)
        assert b.is_exceeded(_state(total_cost_usd=9.0)) is True

    def test_zero_max_disables(self):
        b = CostBudget(0.0)
        assert b.is_exceeded(_state(total_cost_usd=999.0)) is False


class TestTokenBudget:
    def test_uses_state_window_by_default(self):
        b = TokenBudget(threshold_ratio=0.85)
        assert b.is_exceeded(_state(total_tokens=100_000, context_window_budget=200_000)) is False
        assert b.is_exceeded(_state(total_tokens=170_001, context_window_budget=200_000)) is True

    def test_explicit_max_overrides_state(self):
        b = TokenBudget(max_tokens=1000, threshold_ratio=0.5)
        assert b.is_exceeded(_state(total_tokens=400)) is False
        assert b.is_exceeded(_state(total_tokens=600)) is True

    def test_zero_cap_disables(self):
        b = TokenBudget(max_tokens=0)
        assert b.is_exceeded(_state(total_tokens=999_999)) is False


class TestWallClockBudget:
    def test_under_limit_not_exceeded(self):
        clock = iter([0.0, 1.0])  # origin=0, eval=1
        b = WallClockBudget(10.0, clock=lambda: next(clock))
        assert b.is_exceeded(_state()) is False

    def test_at_limit_exceeded(self):
        clock = iter([0.0, 11.0])
        b = WallClockBudget(10.0, clock=lambda: next(clock))
        assert b.is_exceeded(_state()) is True

    def test_zero_seconds_disables(self):
        clock = iter([0.0, 999.0])
        b = WallClockBudget(0.0, clock=lambda: next(clock))
        assert b.is_exceeded(_state()) is False


class TestToolCallBudget:
    def test_under_cap(self):
        b = ToolCallBudget(5)
        assert b.is_exceeded(_state(tool_results=[{"a": 1}])) is False

    def test_at_cap_exceeded(self):
        b = ToolCallBudget(3)
        assert (
            b.is_exceeded(_state(tool_results=[{"a": 1}, {"a": 2}, {"a": 3}])) is True
        )


# ─────────────────────────────────────────────────────────────────
# Controller — empty / signal-driven defaults
# ─────────────────────────────────────────────────────────────────


class TestEmptyController:
    def test_empty_dim_list_with_pending_continues(self):
        c = MultiDimensionalBudgetController()
        # Pending tool calls + no completion signal → continue
        assert (
            c.decide(_state(pending_tool_calls=[{"x": 1}]))
            == LoopDecision.CONTINUE
        )

    def test_no_dims_no_pending_completes(self):
        c = MultiDimensionalBudgetController()
        # No tool calls / signals → loop completes naturally
        assert c.decide(_state()) == LoopDecision.COMPLETE

    def test_complete_signal_completes(self):
        c = MultiDimensionalBudgetController()
        assert (
            c.decide(_state(completion_signal="complete"))
            == LoopDecision.COMPLETE
        )

    def test_blocked_signal_escalates(self):
        c = MultiDimensionalBudgetController()
        assert (
            c.decide(_state(completion_signal="blocked"))
            == LoopDecision.ESCALATE
        )

    def test_error_signal_errors(self):
        c = MultiDimensionalBudgetController()
        assert c.decide(_state(completion_signal="error")) == LoopDecision.ERROR

    def test_tool_results_continue(self):
        c = MultiDimensionalBudgetController()
        # Even with no pending calls, fresh tool_results → continue
        # (model gets a chance to reason about them).
        assert (
            c.decide(_state(tool_results=[{"a": 1}]))
            == LoopDecision.CONTINUE
        )


# ─────────────────────────────────────────────────────────────────
# Controller — dimension trips short-circuits to COMPLETE
# ─────────────────────────────────────────────────────────────────


class TestDimensionShortCircuit:
    def test_iteration_budget_stops_loop(self):
        c = MultiDimensionalBudgetController([IterationBudget(3)])
        # iteration=3 with pending tool calls would normally CONTINUE
        result = c.decide(
            _state(iteration=3, pending_tool_calls=[{"x": 1}])
        )
        assert result == LoopDecision.COMPLETE
        assert c.last_exceeded_dimension == "iteration"

    def test_cost_budget_stops_loop(self):
        c = MultiDimensionalBudgetController(
            [CostBudget(10.0, threshold_ratio=0.5)]
        )
        result = c.decide(
            _state(total_cost_usd=6.0, pending_tool_calls=[{"x": 1}])
        )
        assert result == LoopDecision.COMPLETE
        assert c.last_exceeded_dimension == "cost"

    def test_token_budget_stops_loop(self):
        c = MultiDimensionalBudgetController(
            [TokenBudget(max_tokens=1000, threshold_ratio=0.5)]
        )
        result = c.decide(
            _state(total_tokens=600, pending_tool_calls=[{"x": 1}])
        )
        assert result == LoopDecision.COMPLETE
        assert c.last_exceeded_dimension == "tokens"

    def test_first_exceeded_wins(self):
        # Both dimensions exceeded — first declared one is recorded.
        c = MultiDimensionalBudgetController(
            [
                IterationBudget(2),
                CostBudget(1.0, threshold_ratio=0.5),
            ]
        )
        result = c.decide(
            _state(iteration=5, total_cost_usd=10.0, pending_tool_calls=[{"x": 1}])
        )
        assert result == LoopDecision.COMPLETE
        assert c.last_exceeded_dimension == "iteration"

    def test_no_dim_exceeded_falls_through(self):
        c = MultiDimensionalBudgetController(
            [IterationBudget(100), CostBudget(100.0)]
        )
        # No dimension trips → standard signal logic kicks in
        # (pending → continue, complete → complete, etc.)
        assert (
            c.decide(_state(pending_tool_calls=[{"x": 1}]))
            == LoopDecision.CONTINUE
        )
        assert c.last_exceeded_dimension is None


# ─────────────────────────────────────────────────────────────────
# Controller — failure isolation
# ─────────────────────────────────────────────────────────────────


class _CrashyDim(BudgetDimension):
    @property
    def name(self) -> str:
        return "crashy"

    def is_exceeded(self, state: PipelineState) -> bool:
        raise RuntimeError("boom")


class TestFailureIsolation:
    def test_crashy_dim_skipped_others_evaluated(self, caplog):
        c = MultiDimensionalBudgetController(
            [_CrashyDim(), IterationBudget(2)]
        )
        caplog.set_level("WARNING")
        result = c.decide(_state(iteration=5, pending_tool_calls=[{"x": 1}]))
        # IterationBudget still tripped despite the crashy dim above it.
        assert result == LoopDecision.COMPLETE
        assert c.last_exceeded_dimension == "iteration"
        assert any("crashy" in r.message for r in caplog.records)

    def test_all_crashy_falls_through_to_signal_logic(self, caplog):
        c = MultiDimensionalBudgetController(
            [_CrashyDim(), _CrashyDim()]
        )
        caplog.set_level("WARNING")
        # No dimension successfully tripped → standard signal logic
        result = c.decide(_state(pending_tool_calls=[{"x": 1}]))
        assert result == LoopDecision.CONTINUE


# ─────────────────────────────────────────────────────────────────
# Strategy metadata + fluent API
# ─────────────────────────────────────────────────────────────────


class TestMetadata:
    def test_name(self):
        assert MultiDimensionalBudgetController().name == "multi_dim_budget"

    def test_description_lists_dimensions(self):
        c = MultiDimensionalBudgetController(
            [IterationBudget(50), CostBudget(5.0)]
        )
        d = c.description
        assert "50 iterations" in d
        assert "5.00" in d

    def test_empty_description(self):
        assert (
            "no dimensions"
            in MultiDimensionalBudgetController().description.lower()
        )

    def test_dimensions_property_is_defensive_copy(self):
        c = MultiDimensionalBudgetController([IterationBudget(5)])
        out = c.dimensions
        out.append(IterationBudget(99))
        assert len(c.dimensions) == 1

    def test_add_returns_self_for_chaining(self):
        c = MultiDimensionalBudgetController()
        out = c.add(IterationBudget(3)).add(CostBudget(10.0))
        assert out is c
        assert [d.name for d in c.dimensions] == ["iteration", "cost"]


# ─────────────────────────────────────────────────────────────────
# Stage 13 strategy registry wiring
# ─────────────────────────────────────────────────────────────────


class TestStageRegistration:
    def test_registry_includes_multi_dim_budget(self):
        stage = LoopStage()
        registry = stage.get_strategy_slots()["controller"].registry
        assert "multi_dim_budget" in registry
        assert registry["multi_dim_budget"] is MultiDimensionalBudgetController

    def test_can_inject_pre_built_controller(self):
        ctrl = MultiDimensionalBudgetController(
            [IterationBudget(7), CostBudget(2.5)]
        )
        stage = LoopStage(controller=ctrl)
        assert stage._slots["controller"].strategy is ctrl
