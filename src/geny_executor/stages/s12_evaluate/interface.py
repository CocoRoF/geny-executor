"""Stage 12: Evaluate — interface definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod

from geny_executor.core.stage import Strategy
from geny_executor.core.state import PipelineState
from geny_executor.stages.s12_evaluate.types import EvaluationResult


class EvaluationStrategy(Strategy, ABC):
    """Level 2 strategy: how to evaluate response quality."""

    @abstractmethod
    async def evaluate(self, state: PipelineState) -> EvaluationResult:
        """Evaluate current state and return result."""
        ...


class QualityScorer(Strategy, ABC):
    """Optional numerical quality scorer."""

    @abstractmethod
    def score(self, state: PipelineState) -> float:
        """Return quality score 0.0 - 1.0."""
        ...
