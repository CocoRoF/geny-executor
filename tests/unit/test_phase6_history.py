"""Phase 6 — History, Replay, Performance, Cost, and A/B Test tests.

Tests:
  - HistoryService: lifecycle, queries, detail, delete, stats
  - PerformanceMonitor: waterfall, stage_stats, bottlenecks
  - CostAnalyzer: estimate_cost, session_cost, cost_trend
  - ABTestRunner: create_test, complete_side, get_comparison
  - ExecutionReplayer: event replay, summary, snapshots
  - DebugExecutor: breakpoints, continue, step
"""

import sys
import os
import tempfile
import shutil
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
from geny_executor.history.service import HistoryService
from geny_executor.history.monitor import PerformanceMonitor
from geny_executor.history.cost import CostAnalyzer
from geny_executor.history.ab_test import ABTestRunner
from geny_executor.history.replay import ExecutionReplayer, DebugExecutor
from geny_executor.history.models import (
    StageTimingRecord,
    ToolCallRecord,
    WaterfallData,
    CostSummary,
    CostTrendPoint,
    ReplayEvent,
)


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def svc(tmp_dir):
    db = os.path.join(tmp_dir, "test_history.db")
    blobs = os.path.join(tmp_dir, "blobs")
    s = HistoryService(db_path=db, blob_path=blobs)
    yield s
    s.close()


@pytest.fixture
def populated_svc(svc):
    """Service with two completed executions and stage timings."""
    # Exec 1 — completed
    e1 = svc.start_execution("sess-1", "claude-sonnet-4-20250514", "hello world")
    svc.record_stage_timing(
        e1,
        StageTimingRecord(
            iteration=0,
            stage_order=1,
            stage_name="Input",
            started_at="2025-01-01T00:00:00Z",
            finished_at="2025-01-01T00:00:00.050Z",
            duration_ms=50,
            input_tokens=100,
            output_tokens=0,
        ),
    )
    svc.record_stage_timing(
        e1,
        StageTimingRecord(
            iteration=0,
            stage_order=2,
            stage_name="Think",
            started_at="2025-01-01T00:00:00.050Z",
            finished_at="2025-01-01T00:00:00.300Z",
            duration_ms=250,
            input_tokens=100,
            output_tokens=200,
            was_cached=True,
        ),
    )
    svc.record_stage_timing(
        e1,
        StageTimingRecord(
            iteration=0,
            stage_order=3,
            stage_name="Emit",
            started_at="2025-01-01T00:00:00.300Z",
            finished_at="2025-01-01T00:00:00.500Z",
            duration_ms=200,
            input_tokens=0,
            output_tokens=300,
        ),
    )
    svc.record_tool_call(
        e1,
        ToolCallRecord(
            iteration=0,
            tool_name="web_search",
            called_at="2025-01-01T00:00:00.100Z",
            input_json='{"query": "test"}',
            output_text="result",
            duration_ms=120,
        ),
    )
    svc.record_tool_call(
        e1,
        ToolCallRecord(
            iteration=0,
            tool_name="calculator",
            called_at="2025-01-01T00:00:00.200Z",
            input_json='{"expr": "1+1"}',
            output_text="2",
            duration_ms=10,
        ),
    )
    svc.add_tags(e1, ["test", "v1"])
    svc.finish_execution(
        e1,
        "completed",
        result_text="Hello!",
        usage={
            "total_tokens": 600,
            "input_tokens": 200,
            "output_tokens": 400,
            "cache_read_tokens": 50,
            "cache_write_tokens": 10,
            "cost_usd": 0.0123,
            "iterations": 1,
            "tool_calls": 2,
            "thinking_tokens": 30,
        },
    )

    # Exec 2 — error
    e2 = svc.start_execution("sess-1", "claude-opus-4-20250514", "fail test")
    svc.record_stage_timing(
        e2,
        StageTimingRecord(
            iteration=0,
            stage_order=1,
            stage_name="Input",
            started_at="2025-01-01T01:00:00Z",
            finished_at="2025-01-01T01:00:00.030Z",
            duration_ms=30,
        ),
    )
    svc.finish_execution(
        e2,
        "error",
        error={
            "type": "ToolError",
            "message": "not found",
            "stage": 5,
        },
    )

    return svc, e1, e2


# ═══════════════════════════════════════════════════════════
# HistoryService
# ═══════════════════════════════════════════════════════════


class TestHistoryServiceLifecycle:
    def test_start_execution_returns_id(self, svc):
        eid = svc.start_execution("s1", "model-a", "test input")
        assert isinstance(eid, str) and len(eid) > 0

    def test_finish_execution_updates_status(self, svc):
        eid = svc.start_execution("s1", "model-a", "test input")
        svc.finish_execution(eid, "completed", result_text="done")
        detail = svc.get_execution_detail(eid)
        assert detail["status"] == "completed"
        assert detail["result_text"] == "done"
        assert detail["duration_ms"] >= 0

    def test_finish_with_usage(self, svc):
        eid = svc.start_execution("s1", "model-a", "test")
        svc.finish_execution(
            eid,
            "completed",
            usage={
                "total_tokens": 1000,
                "input_tokens": 300,
                "output_tokens": 700,
                "cost_usd": 0.05,
                "iterations": 2,
                "tool_calls": 3,
            },
        )
        detail = svc.get_execution_detail(eid)
        assert detail["total_tokens"] == 1000
        assert detail["cost_usd"] == 0.05
        assert detail["iterations"] == 2

    def test_finish_with_error(self, svc):
        eid = svc.start_execution("s1", "model-a", "test")
        svc.finish_execution(
            eid,
            "error",
            error={
                "type": "RuntimeError",
                "message": "boom",
                "stage": 3,
            },
        )
        detail = svc.get_execution_detail(eid)
        assert detail["status"] == "error"
        assert detail["error_type"] == "RuntimeError"
        assert detail["error_message"] == "boom"
        assert detail["error_stage"] == 3


class TestHistoryServiceRecording:
    def test_record_stage_timing(self, svc):
        eid = svc.start_execution("s1", "m1", "inp")
        svc.record_stage_timing(
            eid,
            StageTimingRecord(
                iteration=0,
                stage_order=1,
                stage_name="Input",
                started_at="2025-01-01T00:00:00Z",
                finished_at="2025-01-01T00:00:00.100Z",
                duration_ms=100,
                input_tokens=50,
                output_tokens=0,
                was_cached=True,
            ),
        )
        detail = svc.get_execution_detail(eid)
        assert len(detail["stage_timings"]) == 1
        t = detail["stage_timings"][0]
        assert t["stage_name"] == "Input"
        assert t["duration_ms"] == 100
        assert t["was_cached"]

    def test_record_tool_call(self, svc):
        eid = svc.start_execution("s1", "m1", "inp")
        svc.record_tool_call(
            eid,
            ToolCallRecord(
                iteration=0,
                tool_name="web_search",
                called_at="2025-01-01T00:00:00Z",
                input_json='{"q":"test"}',
                output_text="result",
                is_error=False,
                duration_ms=50,
            ),
        )
        detail = svc.get_execution_detail(eid)
        assert len(detail["tool_call_records"]) == 1
        tc = detail["tool_call_records"][0]
        assert tc["tool_name"] == "web_search"
        assert tc["duration_ms"] == 50

    def test_add_tags(self, svc):
        eid = svc.start_execution("s1", "m1", "inp")
        svc.add_tags(eid, ["alpha", "beta"])
        detail = svc.get_execution_detail(eid)
        assert set(detail["tags"]) == {"alpha", "beta"}

    def test_add_tags_idempotent(self, svc):
        eid = svc.start_execution("s1", "m1", "inp")
        svc.add_tags(eid, ["alpha"])
        svc.add_tags(eid, ["alpha", "beta"])
        detail = svc.get_execution_detail(eid)
        assert set(detail["tags"]) == {"alpha", "beta"}


class TestHistoryServiceQueries:
    def test_list_executions_all(self, populated_svc):
        svc, e1, e2 = populated_svc
        rows, total = svc.list_executions()
        assert total == 2
        assert len(rows) == 2

    def test_list_filter_by_model(self, populated_svc):
        svc, e1, e2 = populated_svc
        rows, total = svc.list_executions(model="claude-sonnet-4-20250514")
        assert total == 1
        assert rows[0]["model"] == "claude-sonnet-4-20250514"

    def test_list_filter_by_status(self, populated_svc):
        svc, e1, e2 = populated_svc
        rows, total = svc.list_executions(status="error")
        assert total == 1
        assert rows[0]["status"] == "error"

    def test_list_filter_by_session(self, populated_svc):
        svc, e1, e2 = populated_svc
        rows, total = svc.list_executions(session_id="sess-1")
        assert total == 2

    def test_list_pagination(self, populated_svc):
        svc, e1, e2 = populated_svc
        rows, total = svc.list_executions(limit=1, offset=0)
        assert total == 2
        assert len(rows) == 1

    def test_list_order_by_whitelist(self, populated_svc):
        svc, e1, e2 = populated_svc
        # Valid order
        rows, _ = svc.list_executions(order_by="cost_usd DESC")
        assert len(rows) == 2
        # Invalid order falls back to default
        rows, _ = svc.list_executions(order_by="DROP TABLE executions; --")
        assert len(rows) == 2

    def test_get_execution_detail(self, populated_svc):
        svc, e1, e2 = populated_svc
        detail = svc.get_execution_detail(e1)
        assert detail is not None
        assert detail["model"] == "claude-sonnet-4-20250514"
        assert len(detail["stage_timings"]) == 3
        assert len(detail["tool_call_records"]) == 2
        assert "test" in detail["tags"]

    def test_get_execution_detail_not_found(self, svc):
        assert svc.get_execution_detail("nonexistent") is None

    def test_delete_execution(self, populated_svc):
        svc, e1, e2 = populated_svc
        assert svc.delete_execution(e1) is True
        assert svc.get_execution_detail(e1) is None
        _, total = svc.list_executions()
        assert total == 1

    def test_delete_nonexistent(self, svc):
        assert svc.delete_execution("nonexistent") is False

    def test_get_stats(self, populated_svc):
        svc, e1, e2 = populated_svc
        stats = svc.get_stats()
        assert stats["total"] == 2
        assert stats["completed"] == 1
        assert stats["errors"] == 1
        assert stats["total_cost"] > 0

    def test_get_stats_by_session(self, populated_svc):
        svc, e1, e2 = populated_svc
        stats = svc.get_stats(session_id="sess-1")
        assert stats["total"] == 2
        # Nonexistent session
        stats = svc.get_stats(session_id="nonexistent")
        assert stats["total"] == 0


class TestHistoryServiceEventStream:
    def test_save_and_load_events(self, svc):
        eid = svc.start_execution("s1", "m1", "test")
        events = [
            {"type": "stage_start", "stage": "Input", "data": {"stage_order": 1}},
            {"type": "stage_complete", "stage": "Input", "data": {"stage_order": 1}},
        ]
        svc.save_event_stream(eid, events)
        loaded = svc.load_event_stream(eid)
        assert len(loaded) == 2
        assert loaded[0]["type"] == "stage_start"

    def test_load_events_empty(self, svc):
        assert svc.load_event_stream("nonexistent") == []


# ═══════════════════════════════════════════════════════════
# PerformanceMonitor
# ═══════════════════════════════════════════════════════════


class TestPerformanceMonitor:
    def test_get_waterfall(self, populated_svc):
        svc, e1, _ = populated_svc
        monitor = PerformanceMonitor(svc)
        wf = monitor.get_waterfall(e1)
        assert isinstance(wf, WaterfallData)
        assert wf.execution_id == e1
        assert len(wf.iterations) == 1
        assert len(wf.iterations[0].stages) == 3
        # Verify order
        assert wf.iterations[0].stages[0].name == "Input"
        assert wf.iterations[0].stages[1].was_cached is True

    def test_get_waterfall_not_found(self, svc):
        monitor = PerformanceMonitor(svc)
        with pytest.raises(ValueError, match="not found"):
            monitor.get_waterfall("nonexistent")

    def test_get_stage_stats(self, populated_svc):
        svc, e1, e2 = populated_svc
        monitor = PerformanceMonitor(svc)
        stats = monitor.get_stage_stats()
        assert 1 in stats
        assert stats[1].name == "Input"
        assert stats[1].count >= 2  # Both executions have stage 1
        assert stats[1].avg_ms > 0

    def test_get_stage_stats_by_session(self, populated_svc):
        svc, e1, e2 = populated_svc
        monitor = PerformanceMonitor(svc)
        stats = monitor.get_stage_stats(session_id="sess-1")
        assert len(stats) > 0

    def test_get_bottlenecks(self, populated_svc):
        svc, e1, _ = populated_svc
        # Set realistic duration_ms (sum of stages = 500ms)
        svc._conn.execute("UPDATE executions SET duration_ms = 500 WHERE id = ?", (e1,))
        svc._conn.commit()
        monitor = PerformanceMonitor(svc)
        bottlenecks = monitor.get_bottlenecks(e1, threshold_pct=0.3)
        # Think stage (250ms) is 50% of 500ms total
        assert len(bottlenecks) >= 1
        names = [b["stage_name"] for b in bottlenecks]
        assert "Think" in names

    def test_get_bottlenecks_high_threshold(self, populated_svc):
        svc, e1, _ = populated_svc
        # Set realistic duration_ms (sum of stages = 500ms)
        svc._conn.execute("UPDATE executions SET duration_ms = 500 WHERE id = ?", (e1,))
        svc._conn.commit()
        monitor = PerformanceMonitor(svc)
        bottlenecks = monitor.get_bottlenecks(e1, threshold_pct=0.99)
        assert len(bottlenecks) == 0


# ═══════════════════════════════════════════════════════════
# CostAnalyzer
# ═══════════════════════════════════════════════════════════


class TestCostAnalyzerEstimate:
    def test_estimate_sonnet(self):
        cost = CostAnalyzer.estimate_cost(
            "claude-sonnet-4-20250514",
            input_tokens=1_000_000,
            output_tokens=0,
        )
        assert cost == 3.0  # $3 per 1M input tokens

    def test_estimate_with_output(self):
        cost = CostAnalyzer.estimate_cost(
            "claude-sonnet-4-20250514",
            input_tokens=0,
            output_tokens=1_000_000,
        )
        assert cost == 15.0  # $15 per 1M output tokens

    def test_estimate_with_cache(self):
        cost = CostAnalyzer.estimate_cost(
            "claude-sonnet-4-20250514",
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=1_000_000,
        )
        assert cost == 0.3  # $0.30 per 1M cache read

    def test_estimate_unknown_model(self):
        cost = CostAnalyzer.estimate_cost("unknown-model", input_tokens=1000)
        assert cost == 0.0

    def test_estimate_opus(self):
        cost = CostAnalyzer.estimate_cost(
            "claude-opus-4-20250514",
            input_tokens=1_000_000,
        )
        assert cost == 15.0


class TestCostAnalyzerSession:
    def test_get_session_cost_summary(self, populated_svc):
        svc, e1, e2 = populated_svc
        analyzer = CostAnalyzer(svc)
        summary = analyzer.get_session_cost_summary("sess-1")
        assert isinstance(summary, CostSummary)
        assert summary.session_id == "sess-1"
        assert summary.total_executions == 2
        assert summary.total_cost > 0
        assert len(summary.by_model) >= 1

    def test_get_cost_trend(self, populated_svc):
        svc, e1, e2 = populated_svc
        analyzer = CostAnalyzer(svc)
        trend = analyzer.get_cost_trend(session_id="sess-1", granularity="hour")
        assert isinstance(trend, list)
        assert all(isinstance(p, CostTrendPoint) for p in trend)
        assert len(trend) >= 1

    def test_get_cost_trend_daily(self, populated_svc):
        svc, e1, e2 = populated_svc
        analyzer = CostAnalyzer(svc)
        trend = analyzer.get_cost_trend(granularity="day")
        assert len(trend) >= 1

    def test_get_cost_trend_empty(self, svc):
        analyzer = CostAnalyzer(svc)
        trend = analyzer.get_cost_trend(session_id="nonexistent")
        assert trend == []


# ═══════════════════════════════════════════════════════════
# ABTestRunner
# ═══════════════════════════════════════════════════════════


class TestABTestRunner:
    def test_create_test(self, svc):
        runner = ABTestRunner(svc)
        result = runner.create_test("env-a", "env-b", "test input")
        assert result.env_a.execution_id
        assert result.env_b.execution_id
        assert result.env_a.environment_id == "env-a"
        assert result.env_b.environment_id == "env-b"
        assert result.user_input == "test input"
        # Both executions should exist
        assert svc.get_execution_detail(result.env_a.execution_id) is not None
        assert svc.get_execution_detail(result.env_b.execution_id) is not None

    def test_complete_side(self, svc):
        runner = ABTestRunner(svc)
        result = runner.create_test("env-a", "env-b", "test")
        runner.complete_side(
            result.env_a.execution_id,
            result_text="Answer A",
            usage={
                "total_tokens": 500,
                "input_tokens": 200,
                "output_tokens": 300,
                "cost_usd": 0.01,
            },
            duration_ms=1000,
            iterations=1,
            tool_calls_count=2,
        )
        detail = svc.get_execution_detail(result.env_a.execution_id)
        assert detail["status"] == "completed"
        assert detail["result_text"] == "Answer A"

    def test_get_comparison(self, svc):
        runner = ABTestRunner(svc)
        result = runner.create_test("env-a", "env-b", "compare me")
        # Complete both sides
        runner.complete_side(
            result.env_a.execution_id,
            result_text="A result",
            usage={"total_tokens": 500, "cost_usd": 0.01, "iterations": 1, "tool_calls": 2},
            duration_ms=1000,
            iterations=1,
            tool_calls_count=2,
        )
        runner.complete_side(
            result.env_b.execution_id,
            result_text="B result",
            usage={"total_tokens": 800, "cost_usd": 0.02, "iterations": 2, "tool_calls": 3},
            duration_ms=2000,
            iterations=2,
            tool_calls_count=3,
        )
        comp = runner.get_comparison(result.env_a.execution_id, result.env_b.execution_id)
        assert comp is not None
        assert comp["env_a"]["result_text"] == "A result"
        assert comp["env_b"]["result_text"] == "B result"
        assert "diff" in comp

    def test_get_comparison_not_found(self, svc):
        runner = ABTestRunner(svc)
        assert runner.get_comparison("x", "y") is None


# ═══════════════════════════════════════════════════════════
# ExecutionReplayer
# ═══════════════════════════════════════════════════════════


class TestExecutionReplayer:
    def test_get_events_summary(self, svc):
        eid = svc.start_execution("s1", "m1", "test")
        events = [
            {"type": "stage_start", "stage": "Input", "data": {"stage_order": 1}, "iteration": 0},
            {
                "type": "stage_complete",
                "stage": "Input",
                "data": {"stage_order": 1},
                "iteration": 0,
            },
            {"type": "stage_start", "stage": "Think", "data": {"stage_order": 2}, "iteration": 0},
            {
                "type": "stage_complete",
                "stage": "Think",
                "data": {"stage_order": 2},
                "iteration": 0,
            },
            {"type": "stage_start", "stage": "Input", "data": {"stage_order": 1}, "iteration": 1},
            {
                "type": "stage_complete",
                "stage": "Input",
                "data": {"stage_order": 1},
                "iteration": 1,
            },
        ]
        svc.save_event_stream(eid, events)

        replayer = ExecutionReplayer(svc)
        summary = replayer.get_events_summary(eid)
        assert summary["total_events"] == 6
        assert summary["iterations"] == 1
        assert len(summary["stages"]) == 3  # 3 stage_start events

    def test_get_events_summary_empty(self, svc):
        replayer = ExecutionReplayer(svc)
        summary = replayer.get_events_summary("nonexistent")
        assert summary["total_events"] == 0

    def test_get_stage_snapshot(self, svc):
        eid = svc.start_execution("s1", "m1", "test")
        events = [
            {"type": "stage_start", "data": {"stage_order": 1}, "iteration": 0},
            {"type": "stage_complete", "data": {"stage_order": 1}, "iteration": 0},
            {"type": "stage_start", "data": {"stage_order": 2}, "iteration": 0},
            {"type": "stage_complete", "data": {"stage_order": 2}, "iteration": 0},
        ]
        svc.save_event_stream(eid, events)

        replayer = ExecutionReplayer(svc)
        snap = replayer.get_stage_snapshot(eid, stage_order=1, iteration=0)
        assert snap is not None
        assert snap["stage_order"] == 1
        assert snap["events_count"] == 2  # up to stage_complete of stage 1

    def test_get_stage_snapshot_not_found(self, svc):
        replayer = ExecutionReplayer(svc)
        assert replayer.get_stage_snapshot("nonexistent", 1) is None

    def test_replay_basic(self, svc):
        """Test replay yields all events (speed=0 for instant)."""
        eid = svc.start_execution("s1", "m1", "test")
        events = [
            {
                "type": "stage_start",
                "data": {"stage_order": 1},
                "timestamp": "2025-01-01T00:00:00Z",
            },
            {
                "type": "stage_complete",
                "data": {"stage_order": 1},
                "timestamp": "2025-01-01T00:00:01Z",
            },
        ]
        svc.save_event_stream(eid, events)

        replayer = ExecutionReplayer(svc)

        async def _run():
            results = []
            gen = replayer.replay(eid, speed=0)
            async for evt in gen:
                results.append(evt)
            return results

        results = asyncio.run(_run())
        assert len(results) == 2
        assert all(isinstance(r, ReplayEvent) for r in results)
        assert results[0].type == "event"

    def test_replay_empty_raises(self, svc):
        replayer = ExecutionReplayer(svc)

        async def _run():
            gen = replayer.replay("nonexistent", speed=0)
            async for _ in gen:
                pass

        with pytest.raises(ValueError, match="No events"):
            asyncio.run(_run())


# ═══════════════════════════════════════════════════════════
# DebugExecutor
# ═══════════════════════════════════════════════════════════


class TestDebugExecutor:
    def test_set_breakpoints(self):
        de = DebugExecutor()
        de.set_breakpoints({1, 3, 5})
        assert de._breakpoints == {1, 3, 5}

    def test_clear_breakpoints(self):
        de = DebugExecutor()
        de.set_breakpoints({1, 2})
        de.clear_breakpoints()
        assert len(de._breakpoints) == 0

    def test_initial_state_not_paused(self):
        de = DebugExecutor()
        assert de.is_paused is False

    def test_continue_sets_event(self):
        de = DebugExecutor()
        de.continue_execution()
        assert de._continue_event.is_set()
        assert de._step_mode is False

    def test_step_next_sets_step_mode(self):
        de = DebugExecutor()
        de.step_next()
        assert de._continue_event.is_set()
        assert de._step_mode is True
