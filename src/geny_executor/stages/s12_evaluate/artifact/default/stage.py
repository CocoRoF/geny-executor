"""Default implementation of Stage 12: Evaluate."""

from __future__ import annotations

from typing import Any, Dict, Optional

from geny_executor.core.slot import StrategySlot
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState
from geny_executor.stages.s12_evaluate.interface import EvaluationStrategy, QualityScorer
from geny_executor.stages.s12_evaluate.artifact.adaptive.strategy import (
    BinaryClassifyEvaluation,
)
from geny_executor.stages.s12_evaluate.artifact.default.strategies import (
    AgentEvaluation,
    CriteriaBasedEvaluation,
    EvaluationChain,
    NoScorer,
    SignalBasedEvaluation,
    WeightedScorer,
)


class EvaluateStage(Stage[Any, Any]):
    """Stage 12: Evaluate.

    Dual abstraction:
      - Level 2 strategy: evaluation method (signal/criteria/agent)
      - Level 2 scorer: optional quality scoring
    """

    def __init__(
        self,
        strategy: Optional[EvaluationStrategy] = None,
        scorer: Optional[QualityScorer] = None,
    ):
        self._slots: Dict[str, StrategySlot] = {
            "strategy": StrategySlot(
                name="strategy",
                strategy=strategy or SignalBasedEvaluation(),
                registry={
                    "signal_based": SignalBasedEvaluation,
                    "criteria_based": CriteriaBasedEvaluation,
                    "agent_evaluation": AgentEvaluation,
                    "binary_classify": BinaryClassifyEvaluation,
                    # Phase 7 S7.6 — sequential evaluator chain.
                    # Construct via ``EvaluationChain([ev1, ev2, ...])``;
                    # the slot's zero-arg swap path produces an empty
                    # chain (which acts as a no-op verdict).
                    "evaluation_chain": EvaluationChain,
                },
                description="Evaluation strategy",
            ),
            "scorer": StrategySlot(
                name="scorer",
                strategy=scorer or NoScorer(),
                registry={
                    "no_scorer": NoScorer,
                    "weighted": WeightedScorer,
                },
                description="Quality scorer strategy",
            ),
        }

    @property
    def _strategy(self) -> EvaluationStrategy:
        return self._slots["strategy"].strategy  # type: ignore[return-value]

    @property
    def _scorer(self) -> QualityScorer:
        return self._slots["scorer"].strategy  # type: ignore[return-value]

    @property
    def name(self) -> str:
        return "evaluate"

    @property
    def order(self) -> int:
        return 12

    @property
    def category(self) -> str:
        return "decision"

    def get_strategy_slots(self) -> Dict[str, StrategySlot]:
        return self._slots

    async def execute(self, input: Any, state: PipelineState) -> Any:
        state.add_event("evaluate.start", {"strategy": self._strategy.name})

        result = await self._strategy.evaluate(state)

        quality_score = self._scorer.score(state)
        if result.score is None:
            result.score = quality_score

        state.evaluation_score = result.score
        state.evaluation_feedback = result.feedback

        decision_map = {
            "complete": "complete",
            "continue": "continue",
            "retry": "continue",
            "escalate": "escalate",
            "error": "error",
        }
        state.loop_decision = decision_map.get(result.decision, "continue")

        state.add_event(
            "evaluate.complete",
            {
                "passed": result.passed,
                "score": result.score,
                "decision": result.decision,
                "loop_decision": state.loop_decision,
                "feedback": result.feedback[:200] if result.feedback else "",
            },
        )

        return input
