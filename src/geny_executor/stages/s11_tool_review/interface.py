"""Stage 11: Tool Review — interface definitions (S9b.1).

A :class:`Reviewer` inspects pending tool calls (or completed tool
results) and emits zero or more :class:`ToolReviewFlag` records into
``state.tool_review_flags``. Stage 14 (Evaluate) downstream may
consult these flags to decide whether to escalate the loop.

Reviewers are run as a chain (``state.tool_review_flags`` accumulates
across reviewers). Each reviewer is failure-isolated by the stage —
an exception only sidelines that reviewer for the turn.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from geny_executor.core.stage import Strategy
from geny_executor.core.state import PipelineState


SEVERITY_INFO = "info"
SEVERITY_WARN = "warn"
SEVERITY_ERROR = "error"

_VALID_SEVERITIES = frozenset({SEVERITY_INFO, SEVERITY_WARN, SEVERITY_ERROR})


@dataclass(frozen=True)
class ToolReviewFlag:
    """A single review verdict for one pending tool call.

    Attributes:
        tool_call_id: The id of the ``tool_use`` block under review.
            Empty string when the flag applies to the turn as a whole.
        reviewer: The reviewer name that produced the flag.
        severity: ``"info"`` / ``"warn"`` / ``"error"``. Stage 14
            treats ``"error"`` as ``escalate``.
        reason: Human-readable summary of why the flag was raised.
        details: Optional structured payload (e.g. matched pattern,
            byte count). Reviewers populate freely.
    """

    tool_call_id: str
    reviewer: str
    severity: str
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in _VALID_SEVERITIES:
            raise ValueError(
                f"severity must be one of {sorted(_VALID_SEVERITIES)}, got {self.severity!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "reviewer": self.reviewer,
            "severity": self.severity,
            "reason": self.reason,
            "details": dict(self.details),
        }


class Reviewer(Strategy, ABC):
    """Inspect pending tool calls and append ToolReviewFlag records."""

    @abstractmethod
    async def review(
        self,
        tool_calls: List[Dict[str, Any]],
        tool_results: List[Dict[str, Any]],
        state: PipelineState,
    ) -> List[ToolReviewFlag]:
        """Return the list of flags this reviewer raises this turn."""
        ...


def collect_flags(state: PipelineState) -> List[ToolReviewFlag]:
    """Read the running flag list off ``state.shared`` (typed)."""
    raw = state.shared.get("tool_review_flags") or []
    out: List[ToolReviewFlag] = []
    for entry in raw:
        if isinstance(entry, ToolReviewFlag):
            out.append(entry)
        elif isinstance(entry, dict):
            try:
                out.append(
                    ToolReviewFlag(
                        tool_call_id=str(entry.get("tool_call_id", "")),
                        reviewer=str(entry.get("reviewer", "")),
                        severity=str(entry.get("severity", "info")),
                        reason=str(entry.get("reason", "")),
                        details=dict(entry.get("details") or {}),
                    )
                )
            except ValueError:
                continue
    return out


def has_error_flag(state: PipelineState) -> bool:
    """Quick predicate for Stage 14 — any error-severity flag present?"""
    return any(f.severity == SEVERITY_ERROR for f in collect_flags(state))


def reset_flags(state: PipelineState) -> None:
    """Clear the running flag list — typically called at the start of a turn."""
    state.shared["tool_review_flags"] = []


def append_flags(state: PipelineState, flags: List[ToolReviewFlag]) -> None:
    """Append flags to the running list, preserving order."""
    bucket: List[Any] = state.shared.setdefault("tool_review_flags", [])
    bucket.extend(flags)


__all__ = [
    "Reviewer",
    "SEVERITY_ERROR",
    "SEVERITY_INFO",
    "SEVERITY_WARN",
    "ToolReviewFlag",
    "append_flags",
    "collect_flags",
    "has_error_flag",
    "reset_flags",
]
