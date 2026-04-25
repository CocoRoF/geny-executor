"""Unit tests for Stage 19 Summarize (S9b.4)."""

from __future__ import annotations

from typing import Any, Optional

import pytest

from geny_executor.core.state import PipelineState
from geny_executor.memory.provider import Importance
from geny_executor.stages.s19_summarize import (
    FixedImportance,
    HeuristicImportance,
    NoSummarizer,
    RuleBasedSummarizer,
    SUMMARY_HISTORY_KEY,
    SummarizeStage,
    SummaryRecord,
    TURN_SUMMARY_KEY,
)


# ── helpers ───────────────────────────────────────────────────────────


def _state(*, session_id: str = "s", iteration: int = 1) -> PipelineState:
    state = PipelineState(session_id=session_id)
    state.iteration = iteration
    return state


def _add_assistant(state: PipelineState, text: str) -> None:
    state.messages.append({"role": "assistant", "content": text})


# ── SummaryRecord ────────────────────────────────────────────────────


class TestSummaryRecord:
    def test_to_dict(self):
        r = SummaryRecord(
            turn_id="s:1",
            abstract="abs",
            key_facts=["a"],
            entities=["E"],
            tags=["t"],
            importance=Importance.HIGH,
        )
        d = r.to_dict()
        assert d["turn_id"] == "s:1"
        assert d["abstract"] == "abs"
        assert d["importance"] == "high"
        assert d["key_facts"] == ["a"]
        assert d["entities"] == ["E"]
        assert d["tags"] == ["t"]


# ── NoSummarizer ─────────────────────────────────────────────────────


class TestNoSummarizer:
    @pytest.mark.asyncio
    async def test_returns_none(self):
        s = NoSummarizer()
        assert await s.summarize(_state()) is None


# ── RuleBasedSummarizer ──────────────────────────────────────────────


class TestRuleBasedSummarizer:
    @pytest.mark.asyncio
    async def test_no_assistant_text_returns_none(self):
        s = RuleBasedSummarizer()
        assert await s.summarize(_state()) is None

    @pytest.mark.asyncio
    async def test_blank_assistant_text_returns_none(self):
        state = _state()
        _add_assistant(state, "   ")
        s = RuleBasedSummarizer()
        assert await s.summarize(state) is None

    @pytest.mark.asyncio
    async def test_basic_extraction(self):
        state = _state(session_id="sess", iteration=2)
        _add_assistant(state, "First sentence. Second one. Third here. Fourth fact. Fifth.")
        s = RuleBasedSummarizer(max_sentences=2, max_facts=2)
        record = await s.summarize(state)
        assert record is not None
        assert record.turn_id == "sess:2"
        assert record.abstract.startswith("First sentence.")
        assert "Second one." in record.abstract
        # Remaining sentences capped at max_facts.
        assert record.key_facts == ["Third here.", "Fourth fact."]

    @pytest.mark.asyncio
    async def test_entities_extracted_and_capped(self):
        state = _state()
        _add_assistant(
            state,
            "Alice and Bob met in Paris with Charlie. We also chatted with Alice again.",
        )
        s = RuleBasedSummarizer(max_entities=3)
        record = await s.summarize(state)
        assert record is not None
        # Dedupe + cap.
        assert "Alice" in record.entities
        assert len(record.entities) <= 3

    @pytest.mark.asyncio
    async def test_extra_tags_propagate_and_dedupe(self):
        state = _state()
        _add_assistant(state, "One. Two. Three.")
        s = RuleBasedSummarizer(extra_tags=["audit", "rule_based"])
        record = await s.summarize(state)
        assert "rule_based" in record.tags
        assert "audit" in record.tags
        assert record.tags.count("rule_based") == 1

    def test_validation(self):
        with pytest.raises(ValueError):
            RuleBasedSummarizer(max_sentences=0)
        with pytest.raises(ValueError):
            RuleBasedSummarizer(max_facts=-1)

    @pytest.mark.asyncio
    async def test_picks_last_assistant_message(self):
        state = _state()
        state.messages.append({"role": "user", "content": "Q?"})
        _add_assistant(state, "Old answer.")
        state.messages.append({"role": "user", "content": "Q2?"})
        _add_assistant(state, "Newer answer here.")
        record = await RuleBasedSummarizer().summarize(state)
        assert record is not None
        assert "Newer answer" in record.abstract

    @pytest.mark.asyncio
    async def test_handles_block_content(self):
        state = _state()
        state.messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Block text."}],
            }
        )
        record = await RuleBasedSummarizer().summarize(state)
        assert record is not None
        assert "Block text." in record.abstract


# ── FixedImportance ─────────────────────────────────────────────────


class TestFixedImportance:
    @pytest.mark.asyncio
    async def test_default_medium(self):
        s = FixedImportance()
        record = SummaryRecord(turn_id="t")
        assert await s.score(record, _state()) == Importance.MEDIUM

    @pytest.mark.asyncio
    async def test_custom_grade(self):
        s = FixedImportance(grade=Importance.HIGH)
        assert await s.score(SummaryRecord(turn_id="t"), _state()) == Importance.HIGH

    def test_invalid_grade_rejected(self):
        with pytest.raises(ValueError):
            FixedImportance(grade="bogus")

    def test_configure(self):
        s = FixedImportance()
        s.configure({"grade": "critical"})
        # Internal state changed; assert via score.
        import asyncio

        assert asyncio.run(s.score(SummaryRecord(turn_id="t"), _state())) == Importance.CRITICAL


# ── HeuristicImportance ─────────────────────────────────────────────


class TestHeuristicImportance:
    @pytest.mark.asyncio
    async def test_baseline_when_quiet(self):
        s = HeuristicImportance()
        record = SummaryRecord(turn_id="t", abstract="all calm")
        assert await s.score(record, _state()) == Importance.MEDIUM

    @pytest.mark.asyncio
    async def test_high_keyword_promotes(self):
        s = HeuristicImportance()
        record = SummaryRecord(turn_id="t", abstract="critical bug found")
        assert await s.score(record, _state()) == Importance.HIGH

    @pytest.mark.asyncio
    async def test_high_keyword_with_review_error_critical(self):
        s = HeuristicImportance()
        state = _state()
        state.shared["tool_review_flags"] = [{"severity": "error"}]
        record = SummaryRecord(turn_id="t", abstract="urgent failure")
        assert await s.score(record, state) == Importance.CRITICAL

    @pytest.mark.asyncio
    async def test_low_keyword_demotes(self):
        s = HeuristicImportance()
        record = SummaryRecord(turn_id="t", abstract="fyi only")
        assert await s.score(record, _state()) == Importance.LOW

    @pytest.mark.asyncio
    async def test_many_facts_promotes(self):
        s = HeuristicImportance(many_facts_threshold=3)
        record = SummaryRecord(turn_id="t", key_facts=["a", "b", "c"])
        assert await s.score(record, _state()) == Importance.HIGH

    @pytest.mark.asyncio
    async def test_many_entities_promotes(self):
        s = HeuristicImportance(many_entities_threshold=2)
        record = SummaryRecord(turn_id="t", entities=["A", "B"])
        assert await s.score(record, _state()) == Importance.HIGH

    def test_threshold_validation(self):
        with pytest.raises(ValueError):
            HeuristicImportance(many_facts_threshold=0)
        with pytest.raises(ValueError):
            HeuristicImportance(many_entities_threshold=0)

    def test_baseline_validation(self):
        with pytest.raises(ValueError):
            HeuristicImportance(baseline="bogus")


# ── SummarizeStage ──────────────────────────────────────────────────


class TestSummarizeStage:
    def test_default_bypasses(self):
        stage = SummarizeStage()
        assert stage.should_bypass(_state()) is True

    @pytest.mark.asyncio
    async def test_no_summary_default_no_op(self):
        stage = SummarizeStage()
        state = _state()
        _add_assistant(state, "Hello.")
        await stage.execute(input=None, state=state)
        assert TURN_SUMMARY_KEY not in state.shared

    @pytest.mark.asyncio
    async def test_rule_based_publishes_summary(self):
        stage = SummarizeStage(summarizer=RuleBasedSummarizer())
        state = _state()
        _add_assistant(state, "We launched the rocket. It worked. We celebrated.")
        await stage.execute(input=None, state=state)
        record = state.shared[TURN_SUMMARY_KEY]
        assert isinstance(record, SummaryRecord)
        assert "rocket" in record.abstract.lower()

    @pytest.mark.asyncio
    async def test_history_appends(self):
        stage = SummarizeStage(summarizer=RuleBasedSummarizer())
        state = _state()
        _add_assistant(state, "First turn summary text here.")
        await stage.execute(input=None, state=state)
        _add_assistant(state, "Second turn summary text here.")
        await stage.execute(input=None, state=state)
        history = state.shared[SUMMARY_HISTORY_KEY]
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_importance_applied_to_record(self):
        stage = SummarizeStage(
            summarizer=RuleBasedSummarizer(),
            importance=FixedImportance(grade=Importance.CRITICAL),
        )
        state = _state()
        _add_assistant(state, "Critical bug found in production.")
        await stage.execute(input=None, state=state)
        assert state.shared[TURN_SUMMARY_KEY].importance == Importance.CRITICAL

    @pytest.mark.asyncio
    async def test_summarizer_exception_emits_event_no_summary(self):
        class BoomSummarizer(NoSummarizer):
            async def summarize(self, state):
                raise RuntimeError("kaboom")

        stage = SummarizeStage(summarizer=BoomSummarizer())
        state = _state()
        _add_assistant(state, "x")
        # NoSummarizer subclass would be bypassed by default — call execute directly
        # by registering as the slot strategy via constructor.
        await stage.execute(input=None, state=state)
        assert TURN_SUMMARY_KEY not in state.shared
        errs = [e for e in state.events if e["type"] == "summary.summarizer_error"]
        assert len(errs) == 1

    @pytest.mark.asyncio
    async def test_importance_exception_keeps_summary(self):
        class BoomImportance(FixedImportance):
            async def score(self, record, state):
                raise RuntimeError("kaboom")

        stage = SummarizeStage(
            summarizer=RuleBasedSummarizer(),
            importance=BoomImportance(),
        )
        state = _state()
        _add_assistant(state, "Some content.")
        await stage.execute(input=None, state=state)
        # Record still published with the summarizer's default grade.
        assert isinstance(state.shared[TURN_SUMMARY_KEY], SummaryRecord)
        errs = [e for e in state.events if e["type"] == "summary.importance_error"]
        assert len(errs) == 1

    @pytest.mark.asyncio
    async def test_skipped_event_when_summarizer_returns_none(self):
        class AlwaysNone(NoSummarizer):
            @property
            def name(self) -> str:
                return "always_none"

            async def summarize(self, state):
                return None

        stage = SummarizeStage(summarizer=AlwaysNone())
        state = _state()
        _add_assistant(state, "x")
        await stage.execute(input=None, state=state)
        evts = [e for e in state.events if e["type"] == "summary.skipped"]
        assert len(evts) == 1

    @pytest.mark.asyncio
    async def test_provider_record_summary_called_when_present(self):
        recorded: list[SummaryRecord] = []

        class _Runtime:
            class _Provider:
                async def record_summary(self, record):
                    recorded.append(record)

            memory_provider = _Provider()

        stage = SummarizeStage(summarizer=RuleBasedSummarizer())
        state = _state()
        state.session_runtime = _Runtime()
        _add_assistant(state, "Hello world.")
        await stage.execute(input=None, state=state)
        assert len(recorded) == 1
        evts = [e for e in state.events if e["type"] == "summary.provider_recorded"]
        assert len(evts) == 1

    @pytest.mark.asyncio
    async def test_provider_record_summary_failure_isolated(self):
        class _Runtime:
            class _Provider:
                async def record_summary(self, record):
                    raise RuntimeError("write failed")

            memory_provider = _Provider()

        stage = SummarizeStage(summarizer=RuleBasedSummarizer())
        state = _state()
        state.session_runtime = _Runtime()
        _add_assistant(state, "Hello world.")
        await stage.execute(input=None, state=state)
        # Summary still published; only the provider hop emits the error.
        assert TURN_SUMMARY_KEY in state.shared
        errs = [e for e in state.events if e["type"] == "summary.provider_error"]
        assert len(errs) == 1

    @pytest.mark.asyncio
    async def test_provider_without_record_summary_silently_ignored(self):
        class _Runtime:
            class _Provider:
                pass  # no record_summary

            memory_provider = _Provider()

        stage = SummarizeStage(summarizer=RuleBasedSummarizer())
        state = _state()
        state.session_runtime = _Runtime()
        _add_assistant(state, "Hello.")
        await stage.execute(input=None, state=state)
        # Summary published, no provider events.
        assert TURN_SUMMARY_KEY in state.shared
        provider_evts = [
            e
            for e in state.events
            if e["type"] in {"summary.provider_recorded", "summary.provider_error"}
        ]
        assert provider_evts == []

    def test_slot_registries(self):
        stage = SummarizeStage()
        slots = stage.get_strategy_slots()
        assert set(slots["summarizer"].registry) == {"no_summary", "rule_based"}
        assert set(slots["importance"].registry) == {"fixed", "heuristic"}

    def test_default_strategies(self):
        stage = SummarizeStage()
        slots = stage.get_strategy_slots()
        assert isinstance(slots["summarizer"].strategy, NoSummarizer)
        assert isinstance(slots["importance"].strategy, FixedImportance)
