"""Stage 12: Evaluate — response quality evaluation & completion decision."""

from __future__ import annotations

from typing import Any, List, Optional

from geny_executor.core.stage import Stage, StrategyInfo
from geny_executor.core.state import PipelineState
from geny_executor.stages.s12_evaluate.strategies import (
    EvaluationStrategy,
    SignalBasedEvaluation,
    QualityScorer,
    NoScorer,
    EvaluationResult,
)


class EvaluateStage(Stage[Any, Any]):
    """Stage 12: Evaluate.

    Evaluates response quality and determines whether to continue,
    complete, retry, or escalate.

    Dual abstraction:
      - Level 2 strategy: evaluation method (signal/criteria/agent)
      - Level 2 scorer: optional quality scoring
    """

    def __init__(
        self,
        strategy: Optional[EvaluationStrategy] = None,
        scorer: Optional[QualityScorer] = None,
    ):
        self._strategy = strategy or SignalBasedEvaluation()
        self._scorer = scorer or NoScorer()

    @property
    def name(self) -> str:
        return "evaluate"

    @property
    def order(self) -> int:
        return 12

    @property
    def category(self) -> str:
        return "decision"

    async def execute(self, input: Any, state: PipelineState) -> Any:
        state.add_event("evaluate.start", {
            "strategy": self._strategy.name,
        })

        # Run evaluation strategy
        result = await self._strategy.evaluate(state)

        # Run quality scorer
        quality_score = self._scorer.score(state)
        if result.score is None:
            result.score = quality_score

        # Store evaluation results in state
        state.evaluation_score = result.score
        state.evaluation_feedback = result.feedback

        # Map evaluation decision to loop_decision
        decision_map = {
            "complete": "complete",
            "continue": "continue",
            "retry": "continue",  # Retry = continue the loop
            "escalate": "escalate",
            "error": "error",
        }
        state.loop_decision = decision_map.get(result.decision, "continue")

        state.add_event("evaluate.complete", {
            "passed": result.passed,
            "score": result.score,
            "decision": result.decision,
            "loop_decision": state.loop_decision,
            "feedback": result.feedback[:200] if result.feedback else "",
        })

        return input

    def list_strategies(self) -> List[StrategyInfo]:
        return [
            StrategyInfo(
                slot_name="strategy",
                current_impl=type(self._strategy).__name__,
                available_impls=[
                    "SignalBasedEvaluation",
                    "CriteriaBasedEvaluation",
                    "AgentEvaluation",
                ],
            ),
            StrategyInfo(
                slot_name="scorer",
                current_impl=type(self._scorer).__name__,
                available_impls=["NoScorer", "WeightedScorer"],
            ),
        ]
