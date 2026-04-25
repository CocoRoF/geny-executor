"""Stage 11: Tool Review — chain-driven implementation (S9b.1).

Walks an ordered chain of :class:`Reviewer` strategies and
accumulates :class:`ToolReviewFlag` records on
``state.shared['tool_review_flags']``. The default chain is::

    Schema → Sensitive → Destructive → Network → Size

Stage 14 (Evaluate) consults the flag list to decide whether to
escalate the loop on errors.
"""

from geny_executor.stages.s11_tool_review.artifact.default.reviewers import (
    DestructiveResultReviewer,
    NetworkAuditReviewer,
    SchemaReviewer,
    SensitivePatternReviewer,
    SizeReviewer,
)
from geny_executor.stages.s11_tool_review.artifact.default.stage import ToolReviewStage
from geny_executor.stages.s11_tool_review.interface import (
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARN,
    Reviewer,
    ToolReviewFlag,
    append_flags,
    collect_flags,
    has_error_flag,
    reset_flags,
)

__all__ = [
    "DestructiveResultReviewer",
    "NetworkAuditReviewer",
    "Reviewer",
    "SEVERITY_ERROR",
    "SEVERITY_INFO",
    "SEVERITY_WARN",
    "SchemaReviewer",
    "SensitivePatternReviewer",
    "SizeReviewer",
    "ToolReviewFlag",
    "ToolReviewStage",
    "append_flags",
    "collect_flags",
    "has_error_flag",
    "reset_flags",
]
