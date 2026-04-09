"""Evaluate strategies — backward-compatible re-exports."""

from geny_executor.stages.s12_evaluate.interface import EvaluationStrategy, QualityScorer
from geny_executor.stages.s12_evaluate.types import EvaluationResult, QualityCriterion
from geny_executor.stages.s12_evaluate.artifact.default.strategies import (
    SignalBasedEvaluation,
    CriteriaBasedEvaluation,
    AgentEvaluation,
    NoScorer,
    WeightedScorer,
)

__all__ = [
    "EvaluationStrategy",
    "QualityScorer",
    "EvaluationResult",
    "QualityCriterion",
    "SignalBasedEvaluation",
    "CriteriaBasedEvaluation",
    "AgentEvaluation",
    "NoScorer",
    "WeightedScorer",
]
