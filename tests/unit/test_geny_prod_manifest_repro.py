"""2.2.0 Wave 1 — reproduction of the Geny prod worker-loop bug (audit §2.1).

Geny's production worker manifest (Geny/backend/service/executor/
default_manifest.py, loop + evaluate entries) declares:

    s14: strategies={"strategy": "evaluation_chain", "scorer": "no_scorer"}
         strategy_configs={"strategy": {"evaluators": ["binary_classify",
             "signal_based"], "easy_max_turns": 1, "not_easy_max_turns": 30}}
    s16: strategies={"controller": "multi_dim_budget"}
         config={"max_turns": 30}
         strategy_configs={"controller": {"dimensions": ["iterations"]}}

Before this wave, the base no-op ``Strategy.configure`` swallowed both
``strategy_configs`` blocks on the slot-swap + configure path used by
``PipelineMutator.restore`` (which ``Pipeline.from_manifest`` delegates
to). The result in prod: an EMPTY evaluation chain returning
``decision="complete"`` every turn — the worker session died after one
iteration regardless of [CONTINUE] signals — and an empty dimension
list on the budget controller. This module is the regression test: it
drives the exact manifest shape through the real restore path and
asserts the live objects.
"""

from __future__ import annotations

import pytest

from geny_executor.core.environment import (
    EnvironmentManifest,
    EnvironmentMetadata,
    StageManifestEntry,
    ToolsSnapshot,
)
from geny_executor.core.pipeline import Pipeline
from geny_executor.core.state import PipelineState
from geny_executor.stages.s14_evaluate import BinaryClassifyEvaluation, EvaluationChain
from geny_executor.stages.s14_evaluate.artifact.default.strategies import (
    SignalBasedEvaluation,
)
from geny_executor.stages.s16_loop import MultiDimensionalBudgetController
from geny_executor.stages.s16_loop.interface import LoopDecision


def _geny_worker_manifest() -> EnvironmentManifest:
    entries = [
        StageManifestEntry(order=1, name="input"),
        StageManifestEntry(
            order=14,
            name="evaluate",
            strategies={"strategy": "evaluation_chain", "scorer": "no_scorer"},
            strategy_configs={
                "strategy": {
                    "evaluators": ["binary_classify", "signal_based"],
                    "easy_max_turns": 1,
                    "not_easy_max_turns": 30,
                },
            },
        ),
        StageManifestEntry(
            order=16,
            name="loop",
            strategies={"controller": "multi_dim_budget"},
            config={"max_turns": 30},
            strategy_configs={
                "controller": {
                    "dimensions": ["iterations"],
                },
            },
        ),
    ]
    return EnvironmentManifest(
        metadata=EnvironmentMetadata(
            id="",
            name="geny-worker-repro",
            description="",
            base_preset="worker_adaptive",
        ),
        model={},
        pipeline={},
        stages=[e.to_dict() for e in entries],
        tools=ToolsSnapshot(built_in=[], external=[]),
    )


@pytest.fixture()
def pipeline() -> Pipeline:
    return Pipeline.from_manifest(_geny_worker_manifest(), api_key="sk-test", strict=False)


class TestEvaluationChainRestored:
    def test_chain_is_not_empty(self, pipeline):
        """THE prod bug: the chain restored empty. It must hold the two
        declared evaluators, in order."""
        chain = pipeline.get_stage(14).get_strategy_slots()["strategy"].strategy
        assert isinstance(chain, EvaluationChain)
        assert [type(ev) for ev in chain.evaluators] == [
            BinaryClassifyEvaluation,
            SignalBasedEvaluation,
        ]

    def test_turn_limits_reached_binary_classify(self, pipeline):
        chain = pipeline.get_stage(14).get_strategy_slots()["strategy"].strategy
        binary = chain.evaluators[0]
        assert binary.get_config() == {"easy_max_turns": 1, "not_easy_max_turns": 30}

    @pytest.mark.asyncio
    async def test_continue_turn_keeps_loop_alive(self, pipeline):
        """The observable prod symptom: a turn with pending tool calls
        returned 'complete' from the empty chain and the session ended
        after one iteration. The restored chain must say 'continue'."""
        chain = pipeline.get_stage(14).get_strategy_slots()["strategy"].strategy
        state = PipelineState(session_id="s")
        state.iteration = 1
        state.pending_tool_calls = [
            {"tool_name": "memory_search", "tool_use_id": "t1", "tool_input": {}}
        ]

        result = await chain.evaluate(state)

        assert result.decision == "continue"
        assert state.metadata["task_class"] == "not_easy"
        assert state.max_iterations == 30  # not_easy_max_turns landed


class TestBudgetControllerRestored:
    def test_controller_has_iterations_dimension(self, pipeline):
        controller = pipeline.get_stage(16).get_strategy_slots()["controller"].strategy
        assert isinstance(controller, MultiDimensionalBudgetController)
        assert [d.name for d in controller.dimensions] == ["iteration"]

    def test_stage_max_turns_reached_the_dimension(self, pipeline):
        """The second inert path: stage config={'max_turns': 30} used to be
        forwarded only to controllers with a `_max_turns` attribute, which
        this controller lacks. It must now arrive via configure()."""
        controller = pipeline.get_stage(16).get_strategy_slots()["controller"].strategy
        cfg = controller.get_config()
        assert cfg["dimensions"] == ["iterations"]
        assert cfg["max_turns"] == 30

    def test_dimension_enforced_at_the_declared_cap(self, pipeline):
        controller = pipeline.get_stage(16).get_strategy_slots()["controller"].strategy

        def state_at(iteration: int) -> PipelineState:
            s = PipelineState(session_id="s")
            s.iteration = iteration
            s.pending_tool_calls = [
                {"tool_name": "memory_search", "tool_use_id": "t", "tool_input": {}}
            ]
            return s

        assert controller.decide(state_at(29)) == LoopDecision.CONTINUE
        assert controller.decide(state_at(30)) == LoopDecision.COMPLETE
        assert controller.last_exceeded_dimension == "iteration"


class TestManifestRoundTrip:
    def test_snapshot_after_restore_preserves_strategy_configs(self, pipeline):
        """Snapshot → manifest → pipeline → snapshot must keep the configs:
        this is what lets operators export a live environment and get back
        what is actually running (manifest = source of truth)."""
        from geny_executor.core.mutation import PipelineMutator

        snapshot = PipelineMutator(pipeline).snapshot()
        s14 = next(s for s in snapshot.stages if s.order == 14)
        assert s14.strategy_configs["strategy"]["evaluators"] == [
            "binary_classify",
            "signal_based",
        ]
        s16 = next(s for s in snapshot.stages if s.order == 16)
        assert s16.strategy_configs["controller"]["dimensions"] == ["iterations"]
        assert s16.strategy_configs["controller"]["max_turns"] == 30
