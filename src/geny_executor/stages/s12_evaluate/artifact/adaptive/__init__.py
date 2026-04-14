"""Adaptive artifact for Stage 12: Evaluate.

Provides BinaryClassifyEvaluation — an evaluation strategy that
auto-classifies tasks as easy/not_easy on the first turn and adapts
loop behavior accordingly.
"""

from geny_executor.stages.s12_evaluate.artifact.adaptive.strategy import (
    BinaryClassifyEvaluation,
    BinaryClassifyConfig,
)

Stage = None  # Use default EvaluateStage with this strategy injected

__all__ = ["BinaryClassifyEvaluation", "BinaryClassifyConfig"]
