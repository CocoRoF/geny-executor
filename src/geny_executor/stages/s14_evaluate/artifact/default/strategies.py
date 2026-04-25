"""Default artifact strategies for Stage 12: Evaluate."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from geny_executor.core.state import PipelineState
from geny_executor.stages.s14_evaluate.interface import EvaluationStrategy, QualityScorer
from geny_executor.stages.s14_evaluate.types import EvaluationResult, QualityCriterion

logger = logging.getLogger(__name__)


class SignalBasedEvaluation(EvaluationStrategy):
    """Evaluate based on completion signals from Stage 9 (Parse)."""

    @property
    def name(self) -> str:
        return "signal_based"

    async def evaluate(self, state: PipelineState) -> EvaluationResult:
        signal = state.completion_signal

        if signal is None or signal == "continue":
            if state.pending_tool_calls:
                return EvaluationResult(
                    passed=True, decision="continue", feedback="Tool calls pending."
                )
            return EvaluationResult(
                passed=True, decision="continue", feedback="No completion signal detected."
            )

        if signal == "complete":
            return EvaluationResult(
                passed=True,
                score=1.0,
                decision="complete",
                feedback=state.completion_detail or "Task completed.",
            )

        if signal == "blocked":
            return EvaluationResult(
                passed=False,
                score=0.0,
                decision="escalate",
                feedback=state.completion_detail or "Task blocked.",
            )

        if signal == "error":
            return EvaluationResult(
                passed=False,
                score=0.0,
                decision="error",
                feedback=state.completion_detail or "Error encountered.",
            )

        if signal == "delegate":
            return EvaluationResult(
                passed=True,
                decision="continue",
                feedback=f"Delegated to: {state.completion_detail or 'unknown'}",
            )

        return EvaluationResult(
            passed=True, decision="continue", feedback=f"Unknown signal: {signal}"
        )


class CriteriaBasedEvaluation(EvaluationStrategy):
    """Evaluate against predefined quality criteria."""

    def __init__(
        self,
        criteria: Optional[List[QualityCriterion]] = None,
        pass_threshold: float = 0.6,
    ):
        self._criteria = criteria or []
        self._pass_threshold = pass_threshold

    @property
    def name(self) -> str:
        return "criteria_based"

    async def evaluate(self, state: PipelineState) -> EvaluationResult:
        if not self._criteria:
            return EvaluationResult(passed=True, decision="complete")

        criteria_results: List[Dict[str, Any]] = []
        total_weight = 0.0
        weighted_sum = 0.0

        for criterion in self._criteria:
            score = 0.0
            if criterion.check is not None:
                try:
                    score = float(criterion.check(state))
                except Exception:
                    score = 0.0

            passed = score >= criterion.threshold
            criteria_results.append(
                {
                    "name": criterion.name,
                    "score": score,
                    "weight": criterion.weight,
                    "passed": passed,
                }
            )
            weighted_sum += score * criterion.weight
            total_weight += criterion.weight

        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        overall_passed = overall_score >= self._pass_threshold

        return EvaluationResult(
            passed=overall_passed,
            score=overall_score,
            decision="complete" if overall_passed else "retry",
            feedback=f"Score: {overall_score:.2f} (threshold: {self._pass_threshold})",
            criteria_results=criteria_results,
        )


class AgentEvaluation(EvaluationStrategy):
    """Use evaluator agent results from Stage 11."""

    @property
    def name(self) -> str:
        return "agent_evaluation"

    async def evaluate(self, state: PipelineState) -> EvaluationResult:
        eval_input = state.metadata.get("evaluation_input")
        if not eval_input:
            return EvaluationResult(passed=True, decision="complete")

        if not eval_input.get("evaluator_success", False):
            return EvaluationResult(
                passed=True,
                decision="complete",
                feedback="Evaluator failed; accepting response as-is.",
                metadata=eval_input,
            )

        evaluator_text = eval_input.get("evaluator_response", "")
        score = self._extract_score(evaluator_text)

        return EvaluationResult(
            passed=score is None or score >= 0.6,
            score=score,
            decision="complete" if (score is None or score >= 0.6) else "retry",
            feedback=evaluator_text[:500],
            metadata=eval_input,
        )

    def _extract_score(self, text: str) -> Optional[float]:
        match = re.search(r"[Ss]core[:\s]+(\d+(?:\.\d+)?)\s*/\s*100", text)
        if match:
            return float(match.group(1)) / 100.0
        match = re.search(r"[Ss]core[:\s]+(\d+(?:\.\d+)?)", text)
        if match:
            val = float(match.group(1))
            return val / 100.0 if val > 1.0 else val
        return None


class NoScorer(QualityScorer):
    """No scoring — always returns 1.0."""

    @property
    def name(self) -> str:
        return "no_scorer"

    def score(self, state: PipelineState) -> float:
        return 1.0


class WeightedScorer(QualityScorer):
    """Weighted average scorer using configurable metrics."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self._weights = weights or {}

    @property
    def name(self) -> str:
        return "weighted"

    def score(self, state: PipelineState) -> float:
        if not self._weights:
            return 1.0
        total_weight = 0.0
        weighted_sum = 0.0
        for key, weight in self._weights.items():
            val = state.metadata.get(key)
            if val is not None:
                weighted_sum += float(val) * weight
                total_weight += weight
        return weighted_sum / total_weight if total_weight > 0 else 1.0


class EvaluationChain(EvaluationStrategy):
    """Run a sequence of evaluators, first definitive verdict wins.

    Cycle 20260424 executor uplift — Phase 7 Sprint S7.6 (Evaluator
    chain). The pre-S7.6 Stage 12 strategy slot held exactly one
    evaluator. Composing layered policy ("trust signals first; fall
    back to criteria; finally call an evaluator agent") meant writing
    a custom strategy that dispatched manually.

    The chain inverts that: hand it a list of evaluators. They run in
    order; the first one returning ``decision != "continue"`` wins.
    A trailing ``continue`` (everyone passed without a definitive
    verdict) returns the last evaluator's result so the chain
    behaves like "no one objected" — same semantics as
    ``GuardChain`` for Stage 4.

    Failure isolation: an evaluator that raises is logged at WARNING
    and skipped. The chain marches on rather than blowing up the
    whole evaluation pass — matches the fail-open semantics the rest
    of Stage 12 already follows.

    Use cases:
        * **Layered policy**: ``[SignalBasedEvaluation(),
          CriteriaBasedEvaluation(...), AgentEvaluation()]`` — cheap
          checks first, expensive evaluator-LLM only when nothing
          earlier said anything definitive.
        * **Multiple criteria packs** (``CriteriaBasedEvaluation``
          for safety, then for quality, then for length) without
          having to merge their criteria lists.
        * **A/B-style fallback** where a primary evaluator can
          decline ("continue") and a backup runs.

    Note: the chain itself is an :class:`EvaluationStrategy`, so
    nested chains are supported.
    """

    def __init__(self, evaluators: Optional[List[EvaluationStrategy]] = None):
        self._evaluators: List[EvaluationStrategy] = list(evaluators or [])

    @property
    def name(self) -> str:
        return "evaluation_chain"

    @property
    def description(self) -> str:
        names = [getattr(ev, "name", "?") for ev in self._evaluators]
        return f"Sequential chain: {', '.join(names) if names else '(empty)'}"

    @property
    def evaluators(self) -> List[EvaluationStrategy]:
        """Read-only view of the wrapped evaluators in declared order."""
        return list(self._evaluators)

    def add(self, evaluator: EvaluationStrategy) -> "EvaluationChain":
        """Append an evaluator and return self for chaining."""
        self._evaluators.append(evaluator)
        return self

    async def evaluate(self, state: PipelineState) -> EvaluationResult:
        if not self._evaluators:
            # Empty chain → no verdict. Match the existing
            # CriteriaBasedEvaluation no-criteria default.
            return EvaluationResult(
                passed=True,
                decision="complete",
                feedback="empty evaluator chain",
            )

        last_result = EvaluationResult(
            passed=True,
            decision="continue",
            feedback="no evaluator produced a definitive decision",
        )
        for evaluator in self._evaluators:
            try:
                result = await evaluator.evaluate(state)
            except Exception as exc:
                logger.warning(
                    "EvaluationChain: evaluator %r raised; skipping (%s)",
                    getattr(evaluator, "name", "?"),
                    exc,
                    exc_info=True,
                )
                continue

            if result.decision != "continue":
                # First definitive verdict wins — short-circuit the rest.
                return result
            last_result = result
        return last_result
