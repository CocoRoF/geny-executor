"""A turn must not end on the same iteration its tools ran.

Stage 10 executes the pending tool calls and then empties
``state.pending_tool_calls``. Every stage after it — the adaptive
classifier, the signal evaluator, the loop controllers — asked "did this
iteration use tools?" by looking at that list, and got "no" on precisely
the iteration that did. So a first turn that called a tool was classified
``easy`` and completed: the tool ran, its result was appended to the
history, and the model never saw it. The reply was whatever the model had
written *before* dispatching the call — a guess, delivered as an answer.

The retired ``claude_code_cli`` backend hid this for a long time because the
CLI runs its own loop internally; every API backend (and the token-only
Claude Code client) sat on it.

``state.has_fresh_tool_results`` is the missing evidence: tool results
exist AND Stage 10 stamped this iteration.
"""

from __future__ import annotations

import pytest

from geny_executor.core.state import PipelineState
from geny_executor.stages.s14_evaluate.artifact.adaptive.strategy import (
    BinaryClassifyEvaluation,
)
from geny_executor.stages.s14_evaluate.artifact.default.strategies import (
    SignalBasedEvaluation,
)
from geny_executor.stages.s16_loop.artifact.default.controllers import (
    BudgetAwareLoopController,
)


def _state_after_stage10(*, iteration: int = 1, text: str = "Let me read that file.") -> PipelineState:
    """Exactly what Stage 14 receives once Stage 10 has run the calls."""
    state = PipelineState()
    state.iteration = iteration
    state.final_text = text
    state.pending_tool_calls = []          # Stage 10 emptied it
    state.tool_results = [{"tool_use_id": "t1", "content": "hello"}]
    state.tool_iteration = iteration       # ...and stamped the iteration
    return state


class TestFreshResultsAreVisible:
    def test_state_reports_fresh_results(self) -> None:
        assert _state_after_stage10().has_fresh_tool_results is True

    def test_results_from_an_earlier_iteration_are_not_fresh(self) -> None:
        state = _state_after_stage10(iteration=1)
        state.iteration = 2  # the model has since read them
        assert state.has_fresh_tool_results is False

    def test_no_tools_no_freshness(self) -> None:
        state = PipelineState()
        state.iteration = 1
        assert state.has_fresh_tool_results is False

    def test_a_turn_reset_clears_the_stamp(self) -> None:
        state = _state_after_stage10()
        state.begin_turn()
        assert state.tool_iteration == 0
        assert state.has_fresh_tool_results is False


class TestAdaptiveClassifier:
    @pytest.mark.asyncio
    async def test_a_tool_using_first_turn_is_not_easy(self) -> None:
        state = _state_after_stage10()
        result = await BinaryClassifyEvaluation().evaluate(state)
        assert result.decision == "continue"
        assert state.metadata["task_class"] == "not_easy"

    @pytest.mark.asyncio
    async def test_a_plain_first_turn_is_still_easy(self) -> None:
        state = PipelineState()
        state.iteration = 1
        state.final_text = "2 + 2 is 4."
        result = await BinaryClassifyEvaluation().evaluate(state)
        assert result.decision == "complete"
        assert state.metadata["task_class"] == "easy"

    @pytest.mark.asyncio
    async def test_a_later_tool_iteration_continues(self) -> None:
        state = _state_after_stage10(iteration=3)
        state.metadata["task_class"] = "not_easy"
        result = await BinaryClassifyEvaluation().evaluate(state)
        assert result.decision == "continue"

    @pytest.mark.asyncio
    async def test_stale_results_do_not_trap_the_loop(self) -> None:
        """The bug's mirror image: a finished answer must still complete
        even though earlier iterations left tool results on the state."""
        state = _state_after_stage10(iteration=2, text="The file says hello.")
        state.tool_iteration = 1          # results are from the previous pass
        state.metadata["task_class"] = "not_easy"
        result = await BinaryClassifyEvaluation().evaluate(state)
        assert result.decision == "complete"


class TestSignalEvaluator:
    @pytest.mark.asyncio
    async def test_fresh_results_continue(self) -> None:
        result = await SignalBasedEvaluation().evaluate(_state_after_stage10())
        assert result.decision == "continue"


class TestLoopController:
    def test_controller_does_not_complete_on_fresh_results(self) -> None:
        assert BudgetAwareLoopController().decide(_state_after_stage10()) == "continue"

    def test_controller_completes_without_tool_work(self) -> None:
        state = PipelineState()
        state.iteration = 1
        state.final_text = "done"
        assert BudgetAwareLoopController().decide(state) == "complete"
