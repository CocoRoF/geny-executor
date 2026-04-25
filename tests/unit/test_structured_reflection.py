"""Unit tests for the Stage 15 structured reflection schema (S7.9)."""

from __future__ import annotations

import pytest

from geny_executor.core.state import PipelineState
from geny_executor.memory.provider import Importance, Insight
from geny_executor.stages.s15_memory import (
    INSIGHTS_KEY,
    PENDING_INSIGHTS_KEY,
    MemoryStage,
    StructuredReflectiveStrategy,
    coerce_insight,
    drain_pending_insights,
    insights_to_dicts,
    list_recorded_insights,
    record_insight,
)


# ── coerce_insight ──────────────────────────────────────────────────────


class TestCoerceInsight:
    def test_returns_insight_unchanged(self):
        ins = Insight(title="t", content="c")
        assert coerce_insight(ins) is ins

    def test_dict_minimal(self):
        ins = coerce_insight({"title": "t", "content": "c"})
        assert ins.title == "t"
        assert ins.content == "c"
        assert ins.importance == Importance.MEDIUM
        assert ins.category == "general"
        assert ins.tags == []

    def test_dict_full(self):
        ins = coerce_insight(
            {
                "title": "T",
                "content": "C",
                "importance": "high",
                "category": "user",
                "tags": ["a", "b"],
            }
        )
        assert ins.importance == Importance.HIGH
        assert ins.category == "user"
        assert ins.tags == ["a", "b"]

    def test_importance_enum_passthrough(self):
        ins = coerce_insight(
            {"title": "t", "content": "c", "importance": Importance.CRITICAL}
        )
        assert ins.importance == Importance.CRITICAL

    def test_unknown_importance_rejected(self):
        with pytest.raises(ValueError, match="unknown importance"):
            coerce_insight({"title": "t", "content": "c", "importance": "nope"})

    def test_missing_title_rejected(self):
        with pytest.raises(ValueError, match="non-empty 'title'"):
            coerce_insight({"content": "c"})

    def test_blank_title_rejected(self):
        with pytest.raises(ValueError, match="non-empty 'title'"):
            coerce_insight({"title": "   ", "content": "c"})

    def test_missing_content_rejected(self):
        with pytest.raises(ValueError, match="non-empty 'content'"):
            coerce_insight({"title": "t"})

    def test_non_mapping_rejected(self):
        with pytest.raises(TypeError):
            coerce_insight("not a mapping")

    def test_non_list_tags_rejected(self):
        with pytest.raises(ValueError, match="'tags' must be"):
            coerce_insight({"title": "t", "content": "c", "tags": "nope"})

    def test_tags_coerced_to_strings(self):
        ins = coerce_insight({"title": "t", "content": "c", "tags": [1, 2]})
        assert ins.tags == ["1", "2"]

    def test_strips_title_content_whitespace(self):
        ins = coerce_insight({"title": "  t  ", "content": "  c  "})
        assert ins.title == "t" and ins.content == "c"


# ── record_insight & queue plumbing ─────────────────────────────────────


class TestRecordInsight:
    def test_appends_to_pending_queue(self):
        state = PipelineState()
        ins = record_insight(state, title="t", content="c")
        assert isinstance(ins, Insight)
        assert state.metadata[PENDING_INSIGHTS_KEY] == [ins]

    def test_multiple_appends_preserve_order(self):
        state = PipelineState()
        a = record_insight(state, title="a", content="ca")
        b = record_insight(state, title="b", content="cb")
        assert state.metadata[PENDING_INSIGHTS_KEY] == [a, b]

    def test_importance_string(self):
        state = PipelineState()
        ins = record_insight(state, title="t", content="c", importance="critical")
        assert ins.importance == Importance.CRITICAL

    def test_invalid_importance_raises(self):
        state = PipelineState()
        with pytest.raises(ValueError):
            record_insight(state, title="t", content="c", importance="bogus")
        # bad call leaves queue untouched
        assert PENDING_INSIGHTS_KEY not in state.metadata


class TestDrainPending:
    def test_empty_state_returns_empty(self):
        state = PipelineState()
        assert list(drain_pending_insights(state)) == []

    def test_drains_and_clears_queue(self):
        state = PipelineState()
        record_insight(state, title="a", content="ca")
        record_insight(state, title="b", content="cb")
        drained = list(drain_pending_insights(state))
        assert [i.title for i in drained] == ["a", "b"]
        assert state.metadata[PENDING_INSIGHTS_KEY] == []

    def test_accepts_dict_payloads(self):
        state = PipelineState()
        state.metadata[PENDING_INSIGHTS_KEY] = [
            {"title": "x", "content": "y", "importance": "high"}
        ]
        drained = list(drain_pending_insights(state))
        assert drained[0].title == "x"
        assert drained[0].importance == Importance.HIGH

    def test_invalid_payload_clears_queue(self):
        state = PipelineState()
        state.metadata[PENDING_INSIGHTS_KEY] = [{"title": "ok", "content": "ok"}, "junk"]
        with pytest.raises(TypeError):
            list(drain_pending_insights(state))
        # Queue must still be cleared so the same bad payload is not
        # retried on every subsequent run.
        assert state.metadata[PENDING_INSIGHTS_KEY] == []


# ── StructuredReflectiveStrategy ────────────────────────────────────────


class TestStructuredReflectiveStrategy:
    @pytest.mark.asyncio
    async def test_drains_into_recorded_collection(self):
        state = PipelineState()
        record_insight(state, title="t1", content="c1", importance="high")
        record_insight(state, title="t2", content="c2")
        strategy = StructuredReflectiveStrategy()
        await strategy.update(state)

        recorded = list_recorded_insights(state)
        assert len(recorded) == 2
        assert recorded[0].title == "t1"
        assert recorded[0].importance == Importance.HIGH
        assert state.metadata[PENDING_INSIGHTS_KEY] == []

    @pytest.mark.asyncio
    async def test_emits_per_insight_event(self):
        state = PipelineState()
        record_insight(state, title="t", content="c", importance="critical", tags=["x"])
        strategy = StructuredReflectiveStrategy()
        await strategy.update(state)

        evts = [e for e in state.events if e["type"] == "memory.insight_recorded"]
        assert len(evts) == 1
        data = evts[0]["data"]
        assert data["title"] == "t"
        assert data["importance"] == "critical"
        assert data["tags"] == ["x"]

    @pytest.mark.asyncio
    async def test_emits_summary_event(self):
        state = PipelineState()
        record_insight(state, title="t1", content="c1")
        record_insight(state, title="t2", content="c2")
        strategy = StructuredReflectiveStrategy()
        await strategy.update(state)

        summary = [
            e for e in state.events if e["type"] == "memory.structured_reflection_done"
        ]
        assert len(summary) == 1
        data = summary[0]["data"]
        assert data["recorded"] == 2
        assert data["total"] == 2

    @pytest.mark.asyncio
    async def test_clears_needs_reflection_flag(self):
        state = PipelineState()
        state.metadata["needs_reflection"] = True
        record_insight(state, title="t", content="c")
        await StructuredReflectiveStrategy().update(state)
        assert state.metadata["needs_reflection"] is False

    @pytest.mark.asyncio
    async def test_empty_queue_emits_summary_only(self):
        state = PipelineState()
        await StructuredReflectiveStrategy().update(state)
        evts = [e["type"] for e in state.events]
        assert "memory.insight_recorded" not in evts
        assert "memory.structured_reflection_done" in evts
        # Empty drain still cleared the flag.
        assert state.metadata["needs_reflection"] is False

    @pytest.mark.asyncio
    async def test_invalid_payload_emits_error_event(self):
        state = PipelineState()
        state.metadata[PENDING_INSIGHTS_KEY] = [{"title": "missing content"}]
        await StructuredReflectiveStrategy().update(state)

        errs = [e for e in state.events if e["type"] == "memory.insight_invalid"]
        assert len(errs) == 1
        assert "content" in errs[0]["data"]["error"]
        # No partial recording on coercion failure.
        assert list_recorded_insights(state) == []
        assert state.metadata["needs_reflection"] is False

    @pytest.mark.asyncio
    async def test_accumulates_across_runs(self):
        state = PipelineState()
        strategy = StructuredReflectiveStrategy()

        record_insight(state, title="r1", content="x")
        await strategy.update(state)

        record_insight(state, title="r2", content="x")
        await strategy.update(state)

        recorded = list_recorded_insights(state)
        assert [i.title for i in recorded] == ["r1", "r2"]


# ── insights_to_dicts ───────────────────────────────────────────────────


class TestInsightsToDicts:
    def test_round_trip_keys(self):
        ins = Insight(
            title="t",
            content="c",
            category="cat",
            tags=["a"],
            importance=Importance.HIGH,
        )
        out = insights_to_dicts([ins])
        assert out == [
            {
                "title": "t",
                "content": "c",
                "category": "cat",
                "tags": ["a"],
                "importance": "high",
            }
        ]

    def test_empty_input(self):
        assert insights_to_dicts([]) == []


# ── MemoryStage registry wiring ─────────────────────────────────────────


class TestMemoryStageRegistry:
    def test_structured_reflective_in_strategy_registry(self):
        stage = MemoryStage()
        registry = stage.get_strategy_slots()["strategy"].registry
        assert "structured_reflective" in registry
        assert registry["structured_reflective"] is StructuredReflectiveStrategy

    @pytest.mark.asyncio
    async def test_stage_with_structured_strategy_runs_clean(self):
        stage = MemoryStage(strategy=StructuredReflectiveStrategy())
        state = PipelineState()
        record_insight(state, title="t", content="c")
        await stage.execute(input=None, state=state)
        recorded = list_recorded_insights(state)
        assert len(recorded) == 1
