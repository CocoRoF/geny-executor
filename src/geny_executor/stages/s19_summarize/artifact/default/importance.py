"""Default importance scorers for Stage 19 (S9b.4)."""

from __future__ import annotations

from typing import List, Optional

from geny_executor.core.state import PipelineState
from geny_executor.memory.provider import Importance
from geny_executor.stages.s19_summarize.interface import ImportanceScorer
from geny_executor.stages.s19_summarize.types import SummaryRecord


_VALID_IMPORTANCE = {i.value: i for i in Importance}


def _coerce(value) -> Optional[Importance]:
    if isinstance(value, Importance):
        return value
    if isinstance(value, str):
        return _VALID_IMPORTANCE.get(value.lower())
    return None


class FixedImportance(ImportanceScorer):
    """Always returns the configured grade."""

    def __init__(self, grade: Importance = Importance.MEDIUM) -> None:
        coerced = _coerce(grade)
        if coerced is None:
            raise ValueError(f"unknown importance: {grade!r}")
        self._grade = coerced

    @property
    def name(self) -> str:
        return "fixed"

    @property
    def description(self) -> str:
        return "Always assigns a single fixed importance grade"

    def configure(self, config: dict) -> None:
        if "grade" in config:
            coerced = _coerce(config["grade"])
            if coerced is None:
                raise ValueError(f"unknown importance: {config['grade']!r}")
            self._grade = coerced

    async def score(self, record: SummaryRecord, state: PipelineState) -> Importance:
        return self._grade


_DEFAULT_HIGH_KEYWORDS = (
    "critical",
    "urgent",
    "fail",
    "failure",
    "error",
    "bug",
    "incident",
    "outage",
    "deadline",
)
_DEFAULT_LOW_KEYWORDS = (
    "fyi",
    "just checking",
    "no action",
    "trivial",
)


class HeuristicImportance(ImportanceScorer):
    """Importance from cheap content heuristics.

    Decision (clamped to ``[Importance.LOW, Importance.CRITICAL]``):

    * ``high_keywords`` matched in abstract/key_facts → :attr:`HIGH`
      (or :attr:`CRITICAL` when ``escalate_on_tool_review_error`` and
      ``state.shared['tool_review_flags']`` contains an ``error``).
    * ``low_keywords`` matched → :attr:`LOW`.
    * Many key_facts (``>= many_facts_threshold``) or large entity
      count (``>= many_entities_threshold``) → :attr:`HIGH`.
    * Otherwise → ``baseline`` (default :attr:`MEDIUM`).
    """

    def __init__(
        self,
        *,
        baseline: Importance = Importance.MEDIUM,
        high_keywords: Optional[List[str]] = None,
        low_keywords: Optional[List[str]] = None,
        many_facts_threshold: int = 4,
        many_entities_threshold: int = 6,
        escalate_on_tool_review_error: bool = True,
    ) -> None:
        if many_facts_threshold < 1 or many_entities_threshold < 1:
            raise ValueError("threshold values must be >= 1")
        coerced = _coerce(baseline)
        if coerced is None:
            raise ValueError(f"unknown baseline: {baseline!r}")
        self._baseline = coerced
        self._high = tuple(kw.lower() for kw in (high_keywords or _DEFAULT_HIGH_KEYWORDS))
        self._low = tuple(kw.lower() for kw in (low_keywords or _DEFAULT_LOW_KEYWORDS))
        self._many_facts = int(many_facts_threshold)
        self._many_entities = int(many_entities_threshold)
        self._escalate = bool(escalate_on_tool_review_error)

    @property
    def name(self) -> str:
        return "heuristic"

    @property
    def description(self) -> str:
        return "Importance from keyword + size heuristics"

    @staticmethod
    def _haystack(record: SummaryRecord) -> str:
        bits = [record.abstract, *record.key_facts]
        return "\n".join(bits).lower()

    def _has_review_error(self, state: PipelineState) -> bool:
        flags = state.shared.get("tool_review_flags") or []
        for flag in flags:
            if isinstance(flag, dict):
                if flag.get("severity") == "error":
                    return True
            elif getattr(flag, "severity", "") == "error":
                return True
        return False

    async def score(self, record: SummaryRecord, state: PipelineState) -> Importance:
        haystack = self._haystack(record)

        # High keyword path (potentially escalates to CRITICAL).
        if any(kw in haystack for kw in self._high):
            if self._escalate and self._has_review_error(state):
                return Importance.CRITICAL
            return Importance.HIGH

        # Low keyword path.
        if any(kw in haystack for kw in self._low):
            return Importance.LOW

        # Volume-based promotion to HIGH.
        if len(record.key_facts) >= self._many_facts or len(record.entities) >= self._many_entities:
            return Importance.HIGH

        return self._baseline


__all__ = [
    "FixedImportance",
    "HeuristicImportance",
]
