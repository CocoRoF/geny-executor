"""Completion signal detection. Backward-compatible re-exports."""

from geny_executor.stages.s09_parse.interface import (
    CompletionSignal,
    CompletionSignalDetector,
)
from geny_executor.stages.s09_parse.artifact.default.signals import (
    RegexDetector,
    StructuredDetector,
    HybridDetector,
)

__all__ = [
    "CompletionSignal",
    "CompletionSignalDetector",
    "RegexDetector",
    "StructuredDetector",
    "HybridDetector",
]
