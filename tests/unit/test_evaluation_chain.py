"""Phase 7 Sprint S7.6 — EvaluationChain tests."""

from __future__ import annotations

from typing import List

import pytest

from geny_executor.core.state import PipelineState
from geny_executor.stages.s12_evaluate import (
    EvaluateStage,
    EvaluationChain,
    EvaluationResult,
    EvaluationStrategy,
)


# ─────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────


class _Static(EvaluationStrategy):
    """Returns a fixed result; records that it was called."""

    def __init__(self, result: EvaluationResult, *, label: str = "static"):
        self._result = result
        self._label = label
        self.called = False

    @property
    def name(self) -> str:
        return self._label

    async def evaluate(self, state: PipelineState) -> EvaluationResult:
        self.called = True
        return self._result


class _Crashy(EvaluationStrategy):
    def __init__(self, exc: Exception, *, label: str = "crashy"):
        self._exc = exc
        self._label = label

    @property
    def name(self) -> str:
        return self._label

    async def evaluate(self, state: PipelineState) -> EvaluationResult:
        raise self._exc


def _state() -> PipelineState:
    return PipelineState(session_id="s")


# ─────────────────────────────────────────────────────────────────
# Empty chain
# ─────────────────────────────────────────────────────────────────


class TestEmptyChain:
    @pytest.mark.asyncio
    async def test_empty_chain_yields_no_op_complete(self):
        chain = EvaluationChain()
        result = await chain.evaluate(_state())
        # Matches the "no-criteria" CriteriaBasedEvaluation fallback —
        # nothing to say → "complete with no objections".
        assert result.passed is True
        assert result.decision == "complete"
        assert "empty" in result.feedback.lower()


# ─────────────────────────────────────────────────────────────────
# Short-circuit on first definitive verdict
# ─────────────────────────────────────────────────────────────────


class TestShortCircuit:
    @pytest.mark.asyncio
    async def test_first_complete_wins(self):
        first = _Static(
            EvaluationResult(passed=True, decision="complete", feedback="first"),
            label="first",
        )
        second = _Static(
            EvaluationResult(passed=False, decision="retry", feedback="second"),
            label="second",
        )
        chain = EvaluationChain([first, second])

        result = await chain.evaluate(_state())

        assert result.feedback == "first"
        assert result.decision == "complete"
        assert first.called is True
        assert second.called is False

    @pytest.mark.asyncio
    async def test_continue_falls_through_to_next(self):
        first = _Static(
            EvaluationResult(passed=True, decision="continue", feedback="defer"),
            label="first",
        )
        second = _Static(
            EvaluationResult(passed=True, decision="complete", feedback="done"),
            label="second",
        )
        chain = EvaluationChain([first, second])

        result = await chain.evaluate(_state())

        assert result.feedback == "done"
        assert first.called is True
        assert second.called is True

    @pytest.mark.asyncio
    async def test_first_escalate_short_circuits(self):
        first = _Static(
            EvaluationResult(passed=False, decision="escalate", feedback="problem"),
            label="first",
        )
        second = _Static(
            EvaluationResult(passed=True, decision="complete", feedback="done"),
            label="second",
        )
        chain = EvaluationChain([first, second])

        result = await chain.evaluate(_state())

        assert result.decision == "escalate"
        assert second.called is False

    @pytest.mark.asyncio
    async def test_first_retry_short_circuits(self):
        first = _Static(
            EvaluationResult(passed=False, decision="retry", feedback="not yet"),
            label="first",
        )
        second = _Static(
            EvaluationResult(passed=True, decision="complete", feedback="done"),
            label="second",
        )
        chain = EvaluationChain([first, second])

        result = await chain.evaluate(_state())

        assert result.decision == "retry"
        assert second.called is False


# ─────────────────────────────────────────────────────────────────
# All-continue → trailing result returned
# ─────────────────────────────────────────────────────────────────


class TestAllContinue:
    @pytest.mark.asyncio
    async def test_returns_last_result_when_no_definitive_verdict(self):
        first = _Static(
            EvaluationResult(passed=True, decision="continue", feedback="A"),
            label="first",
        )
        second = _Static(
            EvaluationResult(passed=True, decision="continue", feedback="B"),
            label="second",
        )
        chain = EvaluationChain([first, second])

        result = await chain.evaluate(_state())

        assert result.decision == "continue"
        # Last evaluator's feedback wins
        assert result.feedback == "B"


# ─────────────────────────────────────────────────────────────────
# Crashy evaluator isolated
# ─────────────────────────────────────────────────────────────────


class TestFailureIsolation:
    @pytest.mark.asyncio
    async def test_raise_does_not_kill_chain(self, caplog):
        crashy = _Crashy(RuntimeError("boom"), label="crashy")
        ok = _Static(
            EvaluationResult(passed=True, decision="complete", feedback="ok"),
            label="ok",
        )
        chain = EvaluationChain([crashy, ok])
        caplog.set_level("WARNING")

        result = await chain.evaluate(_state())

        assert result.feedback == "ok"
        assert any("crashy" in r.message and "raised" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_all_crashy_yields_default_continue(self):
        chain = EvaluationChain(
            [
                _Crashy(RuntimeError("a"), label="a"),
                _Crashy(RuntimeError("b"), label="b"),
            ]
        )
        result = await chain.evaluate(_state())
        # No definitive verdict produced; default-continue placeholder
        assert result.decision == "continue"
        assert "no evaluator" in result.feedback.lower()


# ─────────────────────────────────────────────────────────────────
# Strategy metadata
# ─────────────────────────────────────────────────────────────────


class TestMetadata:
    def test_name(self):
        assert EvaluationChain().name == "evaluation_chain"

    def test_description_lists_inner_evaluators(self):
        chain = EvaluationChain(
            [
                _Static(EvaluationResult(decision="continue"), label="alpha"),
                _Static(EvaluationResult(decision="continue"), label="beta"),
            ]
        )
        assert "alpha" in chain.description and "beta" in chain.description

    def test_empty_description(self):
        assert "(empty)" in EvaluationChain().description

    def test_evaluators_property_is_defensive_copy(self):
        inner = _Static(EvaluationResult(decision="continue"))
        chain = EvaluationChain([inner])
        out = chain.evaluators
        out.append(_Static(EvaluationResult(decision="continue"), label="x"))
        # Mutating the copy must not mutate the chain
        assert len(chain.evaluators) == 1

    def test_add_returns_self_for_chaining(self):
        chain = EvaluationChain()
        a = _Static(EvaluationResult(decision="continue"), label="a")
        b = _Static(EvaluationResult(decision="continue"), label="b")
        out = chain.add(a).add(b)
        assert out is chain
        assert [ev.name for ev in chain.evaluators] == ["a", "b"]


# ─────────────────────────────────────────────────────────────────
# Stage 12 strategy registry wiring
# ─────────────────────────────────────────────────────────────────


class TestStageRegistration:
    def test_registry_includes_evaluation_chain(self):
        stage = EvaluateStage()
        registry = stage.get_strategy_slots()["strategy"].registry
        assert "evaluation_chain" in registry
        assert registry["evaluation_chain"] is EvaluationChain

    @pytest.mark.asyncio
    async def test_chain_drives_loop_decision_through_stage(self):
        chain = EvaluationChain(
            [
                _Static(
                    EvaluationResult(passed=True, decision="complete", feedback="done"),
                    label="terminal",
                )
            ]
        )
        stage = EvaluateStage(strategy=chain)
        state = PipelineState(session_id="s")

        await stage.execute(None, state)

        assert state.loop_decision == "complete"


# ─────────────────────────────────────────────────────────────────
# Nested chains
# ─────────────────────────────────────────────────────────────────


class TestNestedChain:
    @pytest.mark.asyncio
    async def test_inner_chain_short_circuit_propagates(self):
        inner = EvaluationChain(
            [
                _Static(
                    EvaluationResult(passed=True, decision="continue", feedback="i1"),
                    label="i1",
                ),
                _Static(
                    EvaluationResult(passed=True, decision="complete", feedback="i2"),
                    label="i2",
                ),
            ]
        )
        outer = EvaluationChain(
            [
                _Static(
                    EvaluationResult(passed=True, decision="continue", feedback="o1"),
                    label="o1",
                ),
                inner,
                _Static(
                    EvaluationResult(passed=True, decision="complete", feedback="o3"),
                    label="o3",
                ),
            ]
        )
        result = await outer.evaluate(_state())
        # Inner chain's i2 returned 'complete' → outer short-circuits there
        assert result.feedback == "i2"
