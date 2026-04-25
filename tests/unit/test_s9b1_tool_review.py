"""Unit tests for Stage 11 Tool Review chain (S9b.1)."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from geny_executor.core.state import PipelineState
from geny_executor.stages.s11_tool_review import (
    DestructiveResultReviewer,
    NetworkAuditReviewer,
    Reviewer,
    SchemaReviewer,
    SensitivePatternReviewer,
    SizeReviewer,
    ToolReviewFlag,
    ToolReviewStage,
    collect_flags,
    has_error_flag,
)


def _state_with(
    tool_calls: List[Dict[str, Any]] | None = None,
    tool_results: List[Dict[str, Any]] | None = None,
) -> PipelineState:
    s = PipelineState(session_id="s")
    s.pending_tool_calls = list(tool_calls or [])
    s.tool_results = list(tool_results or [])
    return s


# ── ToolReviewFlag ─────────────────────────────────────────────────────


class TestToolReviewFlag:
    def test_to_dict(self):
        f = ToolReviewFlag(
            tool_call_id="t1",
            reviewer="schema",
            severity="error",
            reason="missing field",
            details={"missing": ["x"]},
        )
        d = f.to_dict()
        assert d["tool_call_id"] == "t1"
        assert d["reviewer"] == "schema"
        assert d["severity"] == "error"
        assert d["reason"] == "missing field"
        assert d["details"] == {"missing": ["x"]}

    def test_invalid_severity_rejected(self):
        with pytest.raises(ValueError):
            ToolReviewFlag(tool_call_id="x", reviewer="r", severity="bogus", reason="r")


# ── SchemaReviewer ─────────────────────────────────────────────────────


class TestSchemaReviewer:
    @pytest.mark.asyncio
    async def test_no_required_fields_means_no_flags(self):
        r = SchemaReviewer()
        flags = await r.review(
            [{"id": "t1", "name": "Anything", "input": {}}], [], _state_with()
        )
        assert flags == []

    @pytest.mark.asyncio
    async def test_missing_required_field_raises_error(self):
        r = SchemaReviewer(required_fields={"Read": ["file_path"]})
        flags = await r.review(
            [{"id": "t1", "name": "Read", "input": {}}], [], _state_with()
        )
        assert len(flags) == 1
        assert flags[0].severity == "error"
        assert "file_path" in flags[0].reason

    @pytest.mark.asyncio
    async def test_present_required_field_passes(self):
        r = SchemaReviewer(required_fields={"Read": ["file_path"]})
        flags = await r.review(
            [{"id": "t1", "name": "Read", "input": {"file_path": "/x"}}],
            [],
            _state_with(),
        )
        assert flags == []


# ── SensitivePatternReviewer ───────────────────────────────────────────


class TestSensitiveReviewer:
    @pytest.mark.asyncio
    async def test_detects_api_key_assignment(self):
        r = SensitivePatternReviewer()
        flags = await r.review(
            [{"id": "t1", "name": "Bash", "input": {"cmd": "API_KEY=abcdef123"}}],
            [],
            _state_with(),
        )
        assert len(flags) == 1
        assert flags[0].severity == "warn"
        assert "sensitive" in flags[0].reason.lower()

    @pytest.mark.asyncio
    async def test_clean_input_passes(self):
        r = SensitivePatternReviewer()
        flags = await r.review(
            [{"id": "t1", "name": "Read", "input": {"file_path": "/etc/hosts"}}],
            [],
            _state_with(),
        )
        assert flags == []

    @pytest.mark.asyncio
    async def test_detects_aws_key(self):
        r = SensitivePatternReviewer()
        flags = await r.review(
            [{"id": "t1", "name": "Write", "input": {"text": "AKIAABCDEFGHIJKLMNOP"}}],
            [],
            _state_with(),
        )
        assert len(flags) == 1
        assert flags[0].details["pattern"] == "aws_access_key"

    @pytest.mark.asyncio
    async def test_does_not_double_flag_one_call(self):
        r = SensitivePatternReviewer()
        flags = await r.review(
            [
                {
                    "id": "t1",
                    "name": "Bash",
                    "input": {"cmd": "API_KEY=AKIAABCDEFGHIJKLMNOP"},
                }
            ],
            [],
            _state_with(),
        )
        assert len(flags) == 1


# ── DestructiveResultReviewer ──────────────────────────────────────────


class TestDestructiveReviewer:
    @pytest.mark.asyncio
    async def test_flags_known_destructive_tool(self):
        r = DestructiveResultReviewer()
        calls = [{"id": "t1", "name": "Bash", "input": {"cmd": "rm -rf /tmp/x"}}]
        results = [{"tool_use_id": "t1", "content": "ok"}]
        flags = await r.review(calls, results, _state_with())
        assert len(flags) == 1
        assert flags[0].severity == "info"
        assert flags[0].details["tool"] == "Bash"

    @pytest.mark.asyncio
    async def test_ignores_unknown_tool(self):
        r = DestructiveResultReviewer()
        calls = [{"id": "t1", "name": "Read", "input": {}}]
        results = [{"tool_use_id": "t1", "content": "ok"}]
        flags = await r.review(calls, results, _state_with())
        assert flags == []

    @pytest.mark.asyncio
    async def test_custom_destructive_list_and_severity(self):
        r = DestructiveResultReviewer(destructive_tools=["DropTable"], severity="error")
        calls = [{"id": "t1", "name": "DropTable", "input": {}}]
        results = [{"tool_use_id": "t1", "content": "ok"}]
        flags = await r.review(calls, results, _state_with())
        assert flags[0].severity == "error"


# ── NetworkAuditReviewer ───────────────────────────────────────────────


class TestNetworkReviewer:
    @pytest.mark.asyncio
    async def test_flags_default_network_tool_as_info(self):
        r = NetworkAuditReviewer()
        calls = [{"id": "t1", "name": "WebFetch", "input": {"url": "https://x.com/a"}}]
        flags = await r.review(calls, [], _state_with())
        assert len(flags) == 1
        assert flags[0].severity == "info"
        assert flags[0].details["host"] == "x.com"

    @pytest.mark.asyncio
    async def test_allowlist_blocks_unknown_host(self):
        r = NetworkAuditReviewer(allowed_hosts=["api.x.com"])
        calls = [{"id": "t1", "name": "WebFetch", "input": {"url": "https://evil.com/a"}}]
        flags = await r.review(calls, [], _state_with())
        assert flags[0].severity == "error"
        assert "evil.com" in flags[0].reason

    @pytest.mark.asyncio
    async def test_allowlist_passes_known_host(self):
        r = NetworkAuditReviewer(allowed_hosts=["api.x.com"])
        calls = [{"id": "t1", "name": "WebFetch", "input": {"url": "https://api.x.com/a"}}]
        flags = await r.review(calls, [], _state_with())
        # Still emits info entry but not error.
        assert flags[0].severity == "info"

    @pytest.mark.asyncio
    async def test_ignores_non_network_tool(self):
        r = NetworkAuditReviewer()
        flags = await r.review([{"id": "t1", "name": "Read", "input": {}}], [], _state_with())
        assert flags == []


# ── SizeReviewer ───────────────────────────────────────────────────────


class TestSizeReviewer:
    @pytest.mark.asyncio
    async def test_below_warn_threshold_passes(self):
        r = SizeReviewer(warn_threshold_bytes=100, error_threshold_bytes=200)
        flags = await r.review([], [{"tool_use_id": "t1", "content": "x"}], _state_with())
        assert flags == []

    @pytest.mark.asyncio
    async def test_warn_band(self):
        r = SizeReviewer(warn_threshold_bytes=10, error_threshold_bytes=100)
        flags = await r.review(
            [],
            [{"tool_use_id": "t1", "content": "x" * 50}],
            _state_with(),
        )
        assert len(flags) == 1
        assert flags[0].severity == "warn"

    @pytest.mark.asyncio
    async def test_error_band(self):
        r = SizeReviewer(warn_threshold_bytes=10, error_threshold_bytes=100)
        flags = await r.review(
            [],
            [{"tool_use_id": "t1", "content": "x" * 200}],
            _state_with(),
        )
        assert flags[0].severity == "error"
        assert flags[0].details["bytes"] == 200

    def test_validation(self):
        with pytest.raises(ValueError):
            SizeReviewer(warn_threshold_bytes=-1)
        with pytest.raises(ValueError):
            SizeReviewer(warn_threshold_bytes=100, error_threshold_bytes=50)


# ── ToolReviewStage integration ────────────────────────────────────────


class _AlwaysFlags(Reviewer):
    def __init__(self, name: str, severity: str = "warn"):
        self._name = name
        self._severity = severity

    @property
    def name(self) -> str:
        return self._name

    async def review(self, tool_calls, tool_results, state):
        return [
            ToolReviewFlag(
                tool_call_id="t1",
                reviewer=self._name,
                severity=self._severity,
                reason=f"{self._name} fired",
            )
        ]


class _RaisingReviewer(Reviewer):
    @property
    def name(self) -> str:
        return "raising"

    async def review(self, tool_calls, tool_results, state):
        raise RuntimeError("boom")


class TestToolReviewStage:
    @pytest.mark.asyncio
    async def test_default_chain_runs_with_no_calls_bypasses(self):
        stage = ToolReviewStage()
        state = _state_with()
        assert stage.should_bypass(state) is True

    @pytest.mark.asyncio
    async def test_flags_accumulate_across_chain(self):
        stage = ToolReviewStage(reviewers=[_AlwaysFlags("a"), _AlwaysFlags("b")])
        state = _state_with(tool_calls=[{"id": "t1", "name": "X", "input": {}}])
        await stage.execute(input=None, state=state)
        flags = collect_flags(state)
        assert [f.reviewer for f in flags] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_flag_emits_event(self):
        stage = ToolReviewStage(reviewers=[_AlwaysFlags("a")])
        state = _state_with(tool_calls=[{"id": "t1", "name": "X", "input": {}}])
        await stage.execute(input=None, state=state)
        evts = [e for e in state.events if e["type"] == "tool_review.flag"]
        assert len(evts) == 1
        assert evts[0]["data"]["reviewer"] == "a"

    @pytest.mark.asyncio
    async def test_summary_event_emitted(self):
        stage = ToolReviewStage(reviewers=[_AlwaysFlags("a"), _AlwaysFlags("b")])
        state = _state_with(tool_calls=[{"id": "t1", "name": "X", "input": {}}])
        await stage.execute(input=None, state=state)
        summary = [e for e in state.events if e["type"] == "tool_review.completed"]
        assert len(summary) == 1
        assert summary[0]["data"]["flags"] == 2
        assert summary[0]["data"]["reviewers"] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_reviewer_exception_isolated(self):
        stage = ToolReviewStage(reviewers=[_RaisingReviewer(), _AlwaysFlags("ok")])
        state = _state_with(tool_calls=[{"id": "t1", "name": "X", "input": {}}])
        await stage.execute(input=None, state=state)
        flags = collect_flags(state)
        # Raising reviewer is sidelined; the other still produced its flag.
        assert [f.reviewer for f in flags] == ["ok"]
        errs = [e for e in state.events if e["type"] == "tool_review.reviewer_error"]
        assert len(errs) == 1

    @pytest.mark.asyncio
    async def test_flags_reset_each_execute(self):
        stage = ToolReviewStage(reviewers=[_AlwaysFlags("a")])
        state = _state_with(tool_calls=[{"id": "t1", "name": "X", "input": {}}])
        await stage.execute(input=None, state=state)
        await stage.execute(input=None, state=state)
        # Second run should NOT have flags from the first run.
        flags = collect_flags(state)
        assert len(flags) == 1

    @pytest.mark.asyncio
    async def test_has_error_flag_helper(self):
        stage = ToolReviewStage(reviewers=[_AlwaysFlags("a", severity="error")])
        state = _state_with(tool_calls=[{"id": "t1", "name": "X", "input": {}}])
        await stage.execute(input=None, state=state)
        assert has_error_flag(state) is True

    def test_chain_registry_exposes_all_default_reviewer_names(self):
        stage = ToolReviewStage()
        chain = stage.get_strategy_chains()["reviewers"]
        assert set(chain.registry.keys()) == {
            "schema",
            "sensitive",
            "destructive",
            "network",
            "size",
        }

    def test_default_chain_order(self):
        stage = ToolReviewStage()
        names = [r.name for r in stage.get_strategy_chains()["reviewers"].items]
        assert names == ["schema", "sensitive", "destructive", "network", "size"]
