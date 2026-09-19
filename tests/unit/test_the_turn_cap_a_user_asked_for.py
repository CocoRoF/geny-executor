"""Two caps, and which one binds.

A manifest can name a turn cap ("this environment never runs longer than N")
and a session can name one ("this agent never runs longer than M"). They are
different statements and the binding one is the smaller.

Until 2.73.0 the manifest's replaced the session's outright, which broke it in
both directions: a host that raised the session cap found the control did
nothing, and a user who asked for a SHORT leash was quietly given the long
one. Geny shipped a manifest cap of 30 next to a UI that displayed 50.
"""

from __future__ import annotations

import pytest

from geny_executor.stages.s16_loop.artifact.default.controllers import (
    CostBudget,
    IterationBudget,
    MultiDimensionalBudgetController,
    StandardLoopController,
    _binding_cap,
)


class _State:
    def __init__(self, iteration: int = 0, max_iterations: int = 0,
                 cost_budget_usd=None, total_cost_usd: float = 0.0) -> None:
        self.iteration = iteration
        self.max_iterations = max_iterations
        self.cost_budget_usd = cost_budget_usd
        self.total_cost_usd = total_cost_usd
        # A turn mid-tool-loop: the controller only reaches its cap check
        # when there is still work pending.
        self.tool_results = []
        self.completion_signal = ""
        self.pending_tool_calls = [{"id": "t"}]
        self.has_fresh_tool_results = False


class TestWhichCapBinds:
    @pytest.mark.parametrize(
        "manifest,session,expected",
        [
            (30, 50, 30),   # the environment is stricter
            (30, 10, 10),   # the USER is stricter — this is the half that was lost
            (None, 50, 50),
            (0, 50, 50),    # 0 means "I am not naming a cap"
            (30, None, 30),
            (None, None, 0),
        ],
    )
    def test_the_smaller_one(self, manifest, session, expected) -> None:
        assert _binding_cap(manifest, session) == expected


class TestTheIterationDimension:
    def test_a_users_shorter_leash_is_honoured(self) -> None:
        """The failure this guards: asking for 10 turns and getting 30."""
        dim = IterationBudget(max_iterations=30)
        assert dim.is_exceeded(_State(iteration=10, max_iterations=10)) is True

    def test_a_users_longer_leash_does_not_escape_the_environment(self) -> None:
        dim = IterationBudget(max_iterations=30)
        assert dim.is_exceeded(_State(iteration=29, max_iterations=50)) is False
        assert dim.is_exceeded(_State(iteration=30, max_iterations=50)) is True

    def test_a_manifest_that_names_nothing_defers(self) -> None:
        dim = IterationBudget()
        assert dim.is_exceeded(_State(iteration=49, max_iterations=50)) is False
        assert dim.is_exceeded(_State(iteration=50, max_iterations=50)) is True

    def test_nobody_naming_a_cap_is_no_cap(self) -> None:
        assert IterationBudget().is_exceeded(_State(iteration=9999)) is False


class TestTheStandardController:
    def test_the_users_shorter_leash_is_honoured(self) -> None:
        from geny_executor.stages.s16_loop.interface import LoopDecision

        ctrl = StandardLoopController(max_turns=30)
        state = _State(iteration=10, max_iterations=10)
        assert ctrl.decide(state) is LoopDecision.COMPLETE

    def test_and_the_environment_still_binds_above_it(self) -> None:
        from geny_executor.stages.s16_loop.interface import LoopDecision

        ctrl = StandardLoopController(max_turns=30)
        assert ctrl.decide(_State(iteration=29, max_iterations=50)) is LoopDecision.CONTINUE
        assert ctrl.decide(_State(iteration=30, max_iterations=50)) is LoopDecision.COMPLETE


class TestTheCostDimensionCostsNothingUntilAsked:
    """Adding the dimension to a manifest must not start stopping turns on a
    host that never sets a budget."""

    def test_no_budget_never_stops(self) -> None:
        assert CostBudget().is_exceeded(_State(total_cost_usd=999.0)) is False

    def test_a_session_budget_is_honoured(self) -> None:
        dim = CostBudget()
        assert dim.is_exceeded(_State(cost_budget_usd=1.0, total_cost_usd=0.5)) is False
        assert dim.is_exceeded(_State(cost_budget_usd=1.0, total_cost_usd=0.95)) is True

    def test_both_dimensions_compose(self) -> None:
        ctrl = MultiDimensionalBudgetController()
        ctrl.configure({"dimensions": ["iterations", "cost_usd"], "max_turns": 0})
        names = [d.name for d in ctrl.dimensions]
        assert names == ["iteration", "cost"]


class TestTheClassifierNarrowsAndNeverRaises:
    """Stage 14's binary_classify writes ``state.max_iterations`` on the turn
    it classifies. It used to ASSIGN its own cap, which is the same bug one
    stage over: a session asking for 10 turns was handed 30, and one asking
    for 100 was cut to 30 with nothing said."""

    @staticmethod
    def _state(max_iterations: int, *, tools: bool = True):
        from geny_executor.core.state import PipelineState

        state = PipelineState(session_id="s")
        state.max_iterations = max_iterations
        state.iteration = 1
        state.pending_tool_calls = [{"id": "t"}] if tools else []
        state.completion_signal = ""
        return state

    @staticmethod
    def _strategy(not_easy: int = 30, easy: int = 1):
        from geny_executor.stages.s14_evaluate.artifact.adaptive.strategy import (
            BinaryClassifyEvaluation,
        )

        strategy = BinaryClassifyEvaluation()
        strategy.configure({"easy_max_turns": easy, "not_easy_max_turns": not_easy})
        return strategy

    @pytest.mark.asyncio
    async def test_a_shorter_session_cap_survives_classification(self) -> None:
        state = self._state(10)
        await self._strategy().evaluate(state)
        assert state.max_iterations == 10, "the classifier raised a user's short leash"

    @pytest.mark.asyncio
    async def test_a_longer_session_cap_is_narrowed_not_ignored(self) -> None:
        state = self._state(100)
        await self._strategy().evaluate(state)
        assert state.max_iterations == 30

    @pytest.mark.asyncio
    async def test_an_unset_session_cap_takes_the_classifiers(self) -> None:
        state = self._state(0)
        await self._strategy().evaluate(state)
        assert state.max_iterations == 30

    @pytest.mark.asyncio
    async def test_easy_still_ends_the_turn(self) -> None:
        """Narrowing must not weaken the classifier's own call."""
        state = self._state(100, tools=False)
        result = await self._strategy().evaluate(state)
        assert result.decision == "complete"
        assert state.max_iterations == 1
