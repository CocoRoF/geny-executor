"""2.2.0 Wave 1 — EvaluationChain.configure() behaviour (audit §2.1).

The live prod bug: Geny's worker manifest declared
``strategy_configs={"strategy": {"evaluators": ["binary_classify",
"signal_based"], ...}}`` and the base no-op ``Strategy.configure``
dropped it, leaving an empty chain whose ``evaluate()`` returns
``decision="complete"`` unconditionally — the worker loop died after
one iteration. These tests pin that a configured chain actually
consults the named evaluators (and that the unconfigured empty-chain
default is unchanged, since hosts rely on it as a safe no-op).
"""

from __future__ import annotations

import pytest

from geny_executor.core.state import PipelineState
from geny_executor.stages.s14_evaluate import (
    BinaryClassifyEvaluation,
    EvaluateStage,
    EvaluationChain,
    SignalBasedEvaluation,
)

GENY_PROD_CHAIN_CONFIG = {
    "evaluators": ["binary_classify", "signal_based"],
    "easy_max_turns": 1,
    "not_easy_max_turns": 30,
}


def _state(**overrides) -> PipelineState:
    state = PipelineState(session_id="s")
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


class TestConfiguredChainRunsEvaluators:
    @pytest.mark.asyncio
    async def test_configured_chain_resolves_named_instances_in_order(self):
        chain = EvaluationChain()
        chain.configure(GENY_PROD_CHAIN_CONFIG)

        evaluators = chain.evaluators
        assert len(evaluators) == 2
        assert isinstance(evaluators[0], BinaryClassifyEvaluation)
        assert isinstance(evaluators[1], SignalBasedEvaluation)

    @pytest.mark.asyncio
    async def test_turn_limits_forwarded_to_binary_classify(self):
        chain = EvaluationChain()
        chain.configure(GENY_PROD_CHAIN_CONFIG)

        binary = chain.evaluators[0]
        assert binary.get_config() == {"easy_max_turns": 1, "not_easy_max_turns": 30}

    @pytest.mark.asyncio
    async def test_pending_tool_calls_continue_not_complete(self):
        """The exact prod symptom: a turn with pending tool calls must keep
        the loop alive. The empty chain said 'complete'; the configured
        chain consults binary_classify, which classifies not_easy and
        continues."""
        chain = EvaluationChain()
        chain.configure(GENY_PROD_CHAIN_CONFIG)

        state = _state(
            iteration=1,
            pending_tool_calls=[{"tool_name": "Bash", "tool_use_id": "t1", "tool_input": {}}],
        )
        result = await chain.evaluate(state)

        assert result.decision == "continue"
        # binary_classify ran for real: it stamps the classification and
        # rewrites max_iterations from the forwarded not_easy_max_turns.
        assert state.metadata["task_class"] == "not_easy"
        assert state.max_iterations == 30

    @pytest.mark.asyncio
    async def test_easy_classification_uses_forwarded_easy_turns(self):
        chain = EvaluationChain()
        chain.configure(
            {
                "evaluators": ["binary_classify"],
                "easy_max_turns": 2,
                "not_easy_max_turns": 7,
            }
        )

        state = _state(iteration=1)  # no tools, no signal → easy
        result = await chain.evaluate(state)

        assert result.decision == "complete"
        assert state.metadata["task_class"] == "easy"
        assert state.max_iterations == 2

    @pytest.mark.asyncio
    async def test_reconfigure_replaces_the_chain(self):
        chain = EvaluationChain()
        chain.configure({"evaluators": ["signal_based", "binary_classify"]})
        chain.configure({"evaluators": ["signal_based"]})

        assert [type(ev) for ev in chain.evaluators] == [SignalBasedEvaluation]
        assert chain.get_config()["evaluators"] == ["signal_based"]


class TestUnconfiguredBackCompat:
    @pytest.mark.asyncio
    async def test_unconfigured_chain_keeps_safe_default(self):
        """Back-compat: an empty chain still returns the documented no-op
        verdict (complete). 2.2.0 makes the *configured* path real; it
        must not change what unconfigured chains do."""
        chain = EvaluationChain()
        result = await chain.evaluate(_state())
        assert result.passed is True
        assert result.decision == "complete"
        assert "empty" in result.feedback.lower()

    def test_unconfigured_chain_get_config_is_empty(self):
        # Snapshots must not invent a config for a chain nobody configured.
        assert EvaluationChain().get_config() == {}


class TestSlotSwapPath:
    """``StrategySlot.swap(impl, config)`` is the path PipelineMutator.restore
    and Pipeline.from_manifest use — the chain must come out non-empty."""

    def test_set_strategy_with_config_builds_populated_chain(self):
        stage = EvaluateStage()
        stage.set_strategy("strategy", "evaluation_chain", GENY_PROD_CHAIN_CONFIG)

        chain = stage.get_strategy_slots()["strategy"].strategy
        assert isinstance(chain, EvaluationChain)
        assert [ev.name for ev in chain.evaluators] == ["binary_classify", "signal_based"]

    def test_slot_describe_round_trips_strategy_config(self):
        stage = EvaluateStage()
        stage.set_strategy("strategy", "evaluation_chain", GENY_PROD_CHAIN_CONFIG)

        info = stage.get_strategy_slots()["strategy"].describe()
        assert info.config["evaluators"] == ["binary_classify", "signal_based"]
        assert info.config["easy_max_turns"] == 1
        assert info.config["not_easy_max_turns"] == 30

    def test_nested_chain_is_spellable(self):
        # The chain registers itself — nested chains are documented as
        # supported. The inner chain arrives empty (safe no-op verdict).
        chain = EvaluationChain.from_config({"evaluators": ["evaluation_chain"]})
        assert [ev.name for ev in chain.evaluators] == ["evaluation_chain"]
