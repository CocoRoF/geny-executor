"""2.2.0 Wave 1 — MultiDimensionalBudgetController.configure() (audit §2.1).

Companion to the EvaluationChain fix: Geny prod declares
``strategy_configs={"controller": {"dimensions": ["iterations"]}}`` plus a
stage-level ``config={"max_turns": N}``. Before this wave the dimension
list vanished (base no-op configure) AND the stage-level max_turns was
forwarded only to controllers with a ``_max_turns`` attribute — which
this controller doesn't have. Both halves are pinned here.
"""

from __future__ import annotations

from geny_executor.core.state import PipelineState
from geny_executor.stages.s16_loop import (
    IterationBudget,
    LoopStage,
    MultiDimensionalBudgetController,
    StandardLoopController,
    ToolCallBudget,
)
from geny_executor.stages.s16_loop.interface import LoopDecision


def _state(*, iteration: int = 0, pending: bool = True, **overrides) -> PipelineState:
    state = PipelineState(session_id="s")
    state.iteration = iteration
    # Pending tool calls keep the standard signal logic on "continue",
    # so any COMPLETE we observe is the budget dimension talking.
    state.pending_tool_calls = (
        [{"tool_name": "Bash", "tool_use_id": "t", "tool_input": {}}] if pending else []
    )
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


class TestConfiguredDimensionsEnforced:
    def test_iterations_dimension_with_max_turns(self):
        controller = MultiDimensionalBudgetController()
        controller.configure({"dimensions": ["iterations"], "max_turns": 2})

        assert controller.decide(_state(iteration=1)) == LoopDecision.CONTINUE
        assert controller.decide(_state(iteration=2)) == LoopDecision.COMPLETE
        assert controller.last_exceeded_dimension == "iteration"

    def test_iterations_dimension_defers_to_state_when_uncapped(self):
        """Geny's manifest declares the dimension WITHOUT a cap in
        strategy_configs — the cap lives at the session level. The
        dimension must fall back to state.max_iterations, not be inert."""
        controller = MultiDimensionalBudgetController()
        controller.configure({"dimensions": ["iterations"]})

        state = _state(iteration=5)
        state.max_iterations = 5
        assert controller.decide(state) == LoopDecision.COMPLETE

        state = _state(iteration=4)
        state.max_iterations = 5
        assert controller.decide(state) == LoopDecision.CONTINUE

    def test_limit_arriving_after_dimensions_updates_in_place(self):
        """Manifest restore order: slot swap applies strategy_configs first,
        the stage's update_config forwards max_turns through a SECOND
        configure call. The already-built dimension must pick it up."""
        controller = MultiDimensionalBudgetController()
        controller.configure({"dimensions": ["iterations"]})
        controller.configure({"max_turns": 3})

        assert controller.decide(_state(iteration=3)) == LoopDecision.COMPLETE
        assert controller.decide(_state(iteration=2)) == LoopDecision.CONTINUE
        # Round-trip keeps the merged view.
        cfg = controller.get_config()
        assert cfg["dimensions"] == ["iterations"]
        assert cfg["max_turns"] == 3

    def test_tool_calls_dimension(self):
        controller = MultiDimensionalBudgetController.from_config(
            {"dimensions": ["tool_calls"], "max_tool_calls": 2}
        )
        state = _state()
        state.tool_results = [{"tool_use_id": "a"}, {"tool_use_id": "b"}]
        assert controller.decide(state) == LoopDecision.COMPLETE
        assert controller.last_exceeded_dimension == "tool_calls"

    def test_cost_dimension_with_explicit_cap(self):
        controller = MultiDimensionalBudgetController.from_config(
            {"dimensions": ["cost_usd"], "max_cost_usd": 1.0, "cost_threshold_ratio": 0.5}
        )
        assert controller.decide(_state(total_cost_usd=0.6)) == LoopDecision.COMPLETE
        assert controller.decide(_state(total_cost_usd=0.4)) == LoopDecision.CONTINUE

    def test_unconfigured_controller_back_compat(self):
        """No dimensions registered → behaves like StandardLoopController
        (the documented pre-2.2.0 behaviour). Must not start enforcing
        phantom budgets."""
        controller = MultiDimensionalBudgetController()
        assert controller.decide(_state(iteration=999)) == LoopDecision.CONTINUE
        assert controller.get_config() == {}


class TestStageMaxTurnsForwarding:
    """s16 stage fix: config={'max_turns': N} reaches the controller via
    configure() for every controller whose schema declares max_turns —
    the old hasattr('_max_turns') poke skipped MultiDimensionalBudgetController."""

    def test_update_config_reaches_multi_dim_controller(self):
        stage = LoopStage()
        stage.set_strategy("controller", "multi_dim_budget", {"dimensions": ["iterations"]})
        stage.update_config({"max_turns": 2})

        controller = stage.get_strategy_slots()["controller"].strategy
        assert isinstance(controller, MultiDimensionalBudgetController)
        assert controller.get_config()["max_turns"] == 2
        assert controller.decide(_state(iteration=2)) == LoopDecision.COMPLETE

    def test_update_config_still_reaches_standard_controller(self):
        stage = LoopStage()  # default slot strategy is StandardLoopController
        stage.update_config({"max_turns": 3})

        controller = stage.get_strategy_slots()["controller"].strategy
        assert isinstance(controller, StandardLoopController)
        assert controller.get_config()["max_turns"] == 3

    def test_update_config_zero_clears_to_state_fallback(self):
        stage = LoopStage()
        stage.update_config({"max_turns": 3})
        stage.update_config({"max_turns": 0})

        controller = stage.get_strategy_slots()["controller"].strategy
        assert controller.get_config()["max_turns"] == 0  # defer to state

    def test_legacy_private_attr_fallback_for_host_controllers(self):
        """Host-supplied controllers that predate the configure() contract
        (no schema, but a _max_turns attribute) must keep working."""

        class LegacyController(StandardLoopController):
            @classmethod
            def config_schema(cls):
                return None

        legacy = LegacyController()
        stage = LoopStage(controller=legacy)
        stage.update_config({"max_turns": 9})
        assert legacy._max_turns == 9


class TestProgrammaticDimensionsUntouched:
    def test_instance_built_dimensions_accept_limit_updates(self):
        controller = MultiDimensionalBudgetController(
            [IterationBudget(10), ToolCallBudget(5)]
        )
        controller.configure({"max_turns": 2})
        assert controller.decide(_state(iteration=2)) == LoopDecision.COMPLETE

    def test_get_config_reports_live_dimension_names(self):
        controller = MultiDimensionalBudgetController([IterationBudget(10)])
        assert controller.get_config()["dimensions"] == ["iteration"]
