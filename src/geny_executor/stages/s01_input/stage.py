"""Stage 1: Input — backward-compatible re-export.

The concrete implementation has moved to ``artifact.default.stage``.
"""

from geny_executor.stages.s01_input.artifact.default.stage import InputStage

__all__ = ["InputStage"]
