"""Result formatters — backward-compatible re-exports."""

from geny_executor.stages.s16_yield.interface import ResultFormatter
from geny_executor.stages.s16_yield.artifact.default.formatters import (
    DefaultFormatter,
    StructuredFormatter,
    StreamingFormatter,
)
from geny_executor.stages.s16_yield.artifact.default.multi_format import (
    MultiFormatFormatter,
    build_markdown,
    build_structured,
)

__all__ = [
    "ResultFormatter",
    "DefaultFormatter",
    "StructuredFormatter",
    "StreamingFormatter",
    "MultiFormatFormatter",
    "build_markdown",
    "build_structured",
]
