"""Stage 19: Summarize — turn-level + session-close summary writer.

Runs the configured :class:`Summarizer` to produce a
:class:`SummaryRecord`, lets the configured :class:`ImportanceScorer`
grade it, and publishes the record to ``state.shared['turn_summary']``
plus an audit list at ``state.shared['summary_history']``.

The default :class:`NoSummarizer` returns ``None`` so existing
pipelines see no behaviour change. Hosts that want LTM index
generation swap in :class:`RuleBasedSummarizer` (cheap, local) or
plug their own LLM-driven summarizer.

Two provider hand-offs:

1. **Per-turn forward** — if the attached provider exposes an
   advisory ``record_summary(record)`` async method, every turn's
   record is forwarded so LTM indexers can ingest it.

2. **Session-close summary** (D1) — when the pipeline reaches a
   terminal decision (``complete`` / ``error`` / ``escalate``), the
   stage assembles the full ``summary_history`` into a markdown
   block and writes it once to
   ``provider.stm().write_summary(body)``. This is the canonical
   ``transcripts/summary.md`` writer; hosts that previously kept
   their own summary-md writer should drop it.
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
        if state.loop_decision in _TERMINAL_DECISIONS:
            await self._maybe_write_session_summary(history, state)
        return input

    async def _maybe_forward_to_provider(self, record: SummaryRecord, state: PipelineState) -> None:
        provider = self._get_provider(state)
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

    async def _maybe_write_session_summary(
        self,
        history: List[Any],
        state: PipelineState,
    ) -> None:
        """Assemble the session's full summary and forward it once to
        ``provider.stm().write_summary(markdown)`` (D1).

        The pipeline reaches this branch when the loop decision turns
        terminal — ``complete`` / ``error`` / ``escalate``. Earlier
        turns simply append to ``state.shared['summary_history']``;
        this method is the single session-close write.
        """
        provider = self._get_provider(state)
        if provider is None or not history:
            return
        stm_factory = getattr(provider, "stm", None)
        if stm_factory is None or not callable(stm_factory):
            return
        try:
            stm_handle = stm_factory()
        except Exception:  # noqa: BLE001
            return
        if stm_handle is None:
            return
        writer = getattr(stm_handle, "write_summary", None)
        if writer is None or not callable(writer):
            return
        body = _compose_session_summary(history)
        if not body:
            return
        try:
            await writer(body)
        except Exception as exc:  # noqa: BLE001
            logger.warning("provider.stm().write_summary failed: %s", exc)
            state.add_event(
                "summary.session_close_error",
                {"error": str(exc)},
            )
            return
        state.add_event(
            "summary.session_closed",
            {
                "chars": len(body),
                "turns": len(history),
                "decision": state.loop_decision,
            },
        )

    @staticmethod
    def _get_provider(state: PipelineState) -> Optional[Any]:
        runtime = getattr(state, "session_runtime", None)
        if runtime is None:
            return None
        return getattr(runtime, "memory_provider", None)


_TERMINAL_DECISIONS = frozenset({"complete", "error", "escalate"})


def _compose_session_summary(history: List[Any]) -> str:
    """Render the accumulated turn-level summaries into a single
    markdown block suitable for ``transcripts/summary.md``.

    Format mirrors what most hosts expect: an ``## Session Summary``
    heading, then a per-turn entry with importance / abstract / key
    facts. ``history`` items are dicts produced by
    ``SummaryRecord.to_dict()``.
    """
    if not history:
        return ""
    lines: List[str] = ["## Session Summary", ""]
    for entry in history:
        if not isinstance(entry, dict):
            continue
        turn_id = str(entry.get("turn_id", "?"))
        importance = str(entry.get("importance", "medium"))
        abstract = str(entry.get("abstract", "")).strip()
        facts = entry.get("key_facts") or []
        tags = entry.get("tags") or []
        lines.append(f"### Turn {turn_id} ({importance})")
        if abstract:
            lines.append("")
            lines.append(abstract)
        if facts:
            lines.append("")
            lines.append("**Key facts:**")
            for fact in facts:
                lines.append(f"- {str(fact).strip()}")
        if tags:
            tag_str = ", ".join(f"`{t}`" for t in tags)
            lines.append("")
            lines.append(f"_Tags: {tag_str}_")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
