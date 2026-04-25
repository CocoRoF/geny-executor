"""Stage 19: Summarize — real implementation (S9b.4).

Runs the configured :class:`Summarizer` to produce a
:class:`SummaryRecord`, lets the configured :class:`ImportanceScorer`
grade it, and publishes the record to ``state.shared['turn_summary']``
plus an audit list at ``state.shared['summary_history']``.

The default :class:`NoSummarizer` returns ``None`` so existing
pipelines see no behaviour change. Hosts that want LTM index
generation swap in :class:`RuleBasedSummarizer` (cheap, local) or
plug their own LLM-driven summarizer.

If the host attached a memory provider with a ``record_summary``
async method (advisory contract — not all providers implement it),
the stage forwards the record so LTM indexers can ingest it. The
provider is sourced from ``state.session_runtime.memory_provider``
when present; missing or non-callable attributes are silently
ignored.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from geny_executor.core.slot import StrategySlot
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState
from geny_executor.stages.s19_summarize.artifact.default.importance import (
    FixedImportance,
    HeuristicImportance,
)
from geny_executor.stages.s19_summarize.artifact.default.summarizers import (
    NoSummarizer,
    RuleBasedSummarizer,
)
from geny_executor.stages.s19_summarize.interface import (
    SUMMARY_HISTORY_KEY,
    TURN_SUMMARY_KEY,
    ImportanceScorer,
    Summarizer,
)
from geny_executor.stages.s19_summarize.types import SummaryRecord

logger = logging.getLogger(__name__)


class SummarizeStage(Stage[Any, Any]):
    """Stage 19: Summarize.

    Two slots:

    * ``summarizer`` — produces the SummaryRecord (default
      :class:`NoSummarizer`).
    * ``importance`` — grades the record (default
      :class:`FixedImportance(MEDIUM)`).
    """

    def __init__(
        self,
        summarizer: Optional[Summarizer] = None,
        importance: Optional[ImportanceScorer] = None,
    ):
        self._slots: Dict[str, StrategySlot] = {
            "summarizer": StrategySlot(
                name="summarizer",
                strategy=summarizer or NoSummarizer(),
                registry={
                    "no_summary": NoSummarizer,
                    "rule_based": RuleBasedSummarizer,
                },
                description="Produces the SummaryRecord for this turn",
            ),
            "importance": StrategySlot(
                name="importance",
                strategy=importance or FixedImportance(),
                registry={
                    "fixed": FixedImportance,
                    "heuristic": HeuristicImportance,
                },
                description="Assigns an Importance grade to the produced record",
            ),
        }

    @property
    def name(self) -> str:
        return "summarize"

    @property
    def order(self) -> int:
        return 19

    @property
    def category(self) -> str:
        return "finalize"

    @property
    def _summarizer(self) -> Summarizer:
        return self._slots["summarizer"].strategy  # type: ignore[return-value]

    @property
    def _importance(self) -> ImportanceScorer:
        return self._slots["importance"].strategy  # type: ignore[return-value]

    def get_strategy_slots(self) -> Dict[str, StrategySlot]:
        return self._slots

    def should_bypass(self, state: PipelineState) -> bool:
        # NoSummarizer would return None anyway — short-circuit so the
        # stage doesn't even fire its events for the common no-op case.
        return isinstance(self._summarizer, NoSummarizer)

    async def execute(self, input: Any, state: PipelineState) -> Any:
        try:
            record = await self._summarizer.summarize(state)
        except Exception as exc:  # noqa: BLE001 — never wedge the finalize tail
            logger.warning(
                "Summarizer %s raised %s; skipping summary",
                self._summarizer.name,
                exc,
            )
            state.add_event(
                "summary.summarizer_error",
                {"summarizer": self._summarizer.name, "error": str(exc)},
            )
            return input

        if record is None:
            state.add_event(
                "summary.skipped",
                {"summarizer": self._summarizer.name, "reason": "summarizer returned None"},
            )
            return input

        try:
            grade = await self._importance.score(record, state)
        except Exception as exc:  # noqa: BLE001 — keep the summary even on grader bug
            logger.warning(
                "ImportanceScorer %s raised %s; using summarizer default grade",
                self._importance.name,
                exc,
            )
            state.add_event(
                "summary.importance_error",
                {"importance": self._importance.name, "error": str(exc)},
            )
            grade = record.importance

        record.importance = grade

        state.shared[TURN_SUMMARY_KEY] = record
        history: List[Any] = state.shared.setdefault(SUMMARY_HISTORY_KEY, [])
        history.append(record.to_dict())

        state.add_event("summary.written", record.to_dict())

        await self._maybe_forward_to_provider(record, state)
        return input

    async def _maybe_forward_to_provider(self, record: SummaryRecord, state: PipelineState) -> None:
        runtime = getattr(state, "session_runtime", None)
        if runtime is None:
            return
        provider = getattr(runtime, "memory_provider", None)
        if provider is None:
            return
        record_summary = getattr(provider, "record_summary", None)
        if record_summary is None or not callable(record_summary):
            return
        try:
            await record_summary(record)
        except Exception as exc:  # noqa: BLE001 — provider failures don't break the loop
            logger.warning("memory_provider.record_summary failed: %s", exc)
            state.add_event(
                "summary.provider_error",
                {"error": str(exc)},
            )
            return
        state.add_event(
            "summary.provider_recorded",
            {"turn_id": record.turn_id, "importance": record.importance.value},
        )
