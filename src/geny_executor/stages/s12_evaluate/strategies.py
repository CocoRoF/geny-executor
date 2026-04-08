"""Evaluate stage — evaluation strategies (Level 2)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from geny_executor.core.stage import Strategy
from geny_executor.core.state import PipelineState


@dataclass
class EvaluationResult:
    """Result of response evaluation."""

    passed: bool = True
    score: Optional[float] = None  # 0.0 - 1.0
    feedback: str = ""
    decision: str = "continue"  # continue | complete | retry | escalate
    criteria_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvaluationStrategy(Strategy, ABC):
    """Level 2 strategy: how to evaluate response quality."""

    @abstractmethod
    async def evaluate(self, state: PipelineState) -> EvaluationResult:
        """Evaluate current state and return result."""
        ...


class SignalBasedEvaluation(EvaluationStrategy):
    """Evaluate based on completion signals from Stage 9 (Parse).

    Default strategy. Uses completion_signal from state to decide.
    """

    @property
    def name(self) -> str:
        return "signal_based"

    async def evaluate(self, state: PipelineState) -> EvaluationResult:
        signal = state.completion_signal

        if signal is None or signal == "continue":
            # Tool use or no explicit signal — continue the loop
            if state.pending_tool_calls:
                return EvaluationResult(
                    passed=True,
                    decision="continue",
                    feedback="Tool calls pending.",
                )
            return EvaluationResult(
                passed=True,
                decision="continue",
                feedback="No completion signal detected.",
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

        # Unknown signal — continue
        return EvaluationResult(
            passed=True,
            decision="continue",
            feedback=f"Unknown signal: {signal}",
        )


@dataclass
class QualityCriterion:
    """A single quality criterion for criteria-based evaluation."""

    name: str
    description: str
    weight: float = 1.0
    threshold: float = 0.5
    check: Optional[Any] = None  # Callable[[PipelineState], float]


class CriteriaBasedEvaluation(EvaluationStrategy):
    """Evaluate against predefined quality criteria.

    Follows Anthropic's pattern: "subjective quality → measurable criteria."
    """

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

        criteria_results = []
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
            criteria_results.append({
                "name": criterion.name,
                "score": score,
                "weight": criterion.weight,
                "passed": passed,
            })

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
    """Use evaluator agent results from Stage 11.

    Relies on evaluation_input placed by EvaluatorOrchestrator.
    """

    @property
    def name(self) -> str:
        return "agent_evaluation"

    async def evaluate(self, state: PipelineState) -> EvaluationResult:
        eval_input = state.metadata.get("evaluation_input")
        if not eval_input:
            # No evaluator ran — fall back to pass
            return EvaluationResult(passed=True, decision="complete")

        if not eval_input.get("evaluator_success", False):
            return EvaluationResult(
                passed=True,
                decision="complete",
                feedback="Evaluator failed; accepting response as-is.",
                metadata=eval_input,
            )

        evaluator_text = eval_input.get("evaluator_response", "")

        # Try to extract score from evaluator response
        score = self._extract_score(evaluator_text)

        return EvaluationResult(
            passed=score is None or score >= 0.6,
            score=score,
            decision="complete" if (score is None or score >= 0.6) else "retry",
            feedback=evaluator_text[:500],
            metadata=eval_input,
        )

    def _extract_score(self, text: str) -> Optional[float]:
        """Try to extract a numeric score from evaluator text."""
        import re
        # Look for patterns like "Score: 85/100" or "score: 0.85"
        match = re.search(r'[Ss]core[:\s]+(\d+(?:\.\d+)?)\s*/\s*100', text)
        if match:
            return float(match.group(1)) / 100.0
        match = re.search(r'[Ss]core[:\s]+(\d+(?:\.\d+)?)', text)
        if match:
            val = float(match.group(1))
            return val / 100.0 if val > 1.0 else val
        return None


class QualityScorer(Strategy, ABC):
    """Optional numerical quality scorer."""

    @abstractmethod
    def score(self, state: PipelineState) -> float:
        """Return quality score 0.0 - 1.0."""
        ...


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
        # Score based on available metrics in state.metadata
        total_weight = 0.0
        weighted_sum = 0.0
        for key, weight in self._weights.items():
            val = state.metadata.get(key)
            if val is not None:
                weighted_sum += float(val) * weight
                total_weight += weight
        return weighted_sum / total_weight if total_weight > 0 else 1.0
