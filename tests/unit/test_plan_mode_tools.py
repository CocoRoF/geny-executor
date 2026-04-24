"""Phase 3 Week 6 — EnterPlanMode / ExitPlanMode tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

import pytest

from geny_executor.core.state import PipelineState
from geny_executor.stages.s10_tool import SequentialExecutor
from geny_executor.stages.s10_tool.artifact.default.stage import ToolStage
from geny_executor.tools.base import ToolContext
from geny_executor.tools.built_in.plan_mode_tools import (
    PLAN_MODE_KEY,
    EnterPlanModeTool,
    ExitPlanModeTool,
)
from geny_executor.tools.registry import ToolRegistry


def _ctx_with_mode(mode_value: Any) -> ToolContext:
    view = SimpleNamespace(shared={PLAN_MODE_KEY: mode_value} if mode_value is not None else {})
    return ToolContext(session_id="s", working_dir="", state_view=view)


class TestEnterPlanMode:
    @pytest.mark.asyncio
    async def test_off_to_on_sets_mutation(self):
        ctx = _ctx_with_mode(False)
        result = await EnterPlanModeTool().execute({"reason": "planning refactor"}, ctx)
        assert not result.is_error
        assert result.state_mutations == {PLAN_MODE_KEY: True}
        assert result.metadata["was"] is False
        assert result.metadata["plan_mode"] is True
        assert result.metadata["changed"] is True
        assert result.metadata["reason"] == "planning refactor"
        assert "entered" in result.content

    @pytest.mark.asyncio
    async def test_on_to_on_still_emits_mutation(self):
        """Idempotent — calling Enter while already on leaves the mutation
        intact (the apply step deduplicates), but the summary says 'no change'."""
        ctx = _ctx_with_mode(True)
        result = await EnterPlanModeTool().execute({}, ctx)
        assert not result.is_error
        assert result.state_mutations == {PLAN_MODE_KEY: True}
        assert result.metadata["changed"] is False
        assert "already on" in result.content

    @pytest.mark.asyncio
    async def test_reason_optional(self):
        ctx = _ctx_with_mode(False)
        result = await EnterPlanModeTool().execute({}, ctx)
        assert not result.is_error
        assert "reason" not in result.content

    @pytest.mark.asyncio
    async def test_non_string_reason_rejected(self):
        ctx = _ctx_with_mode(False)
        result = await EnterPlanModeTool().execute({"reason": 42}, ctx)
        assert result.is_error


class TestExitPlanMode:
    @pytest.mark.asyncio
    async def test_on_to_off(self):
        ctx = _ctx_with_mode(True)
        result = await ExitPlanModeTool().execute({"reason": "plan approved"}, ctx)
        assert not result.is_error
        assert result.state_mutations == {PLAN_MODE_KEY: False}
        assert result.metadata["was"] is True
        assert result.metadata["plan_mode"] is False
        assert result.metadata["changed"] is True
        assert "exited" in result.content

    @pytest.mark.asyncio
    async def test_off_to_off_is_no_op(self):
        ctx = _ctx_with_mode(False)
        result = await ExitPlanModeTool().execute({}, ctx)
        assert not result.is_error
        assert result.metadata["changed"] is False
        assert "already off" in result.content

    @pytest.mark.asyncio
    async def test_no_state_view_treats_current_as_off(self):
        ctx = ToolContext(session_id="s", working_dir="")
        result = await ExitPlanModeTool().execute({}, ctx)
        assert not result.is_error
        # With no view, the current mode reads as False — exiting is a no-op
        assert result.metadata["changed"] is False


class TestEndToEndStateWiring:
    """Confirms the state_mutations flow lands on state.shared via Stage 10."""

    @pytest.mark.asyncio
    async def test_enter_mutates_shared(self):
        reg = ToolRegistry()
        reg.register(EnterPlanModeTool())
        stage = ToolStage(registry=reg, executor=SequentialExecutor())
        state = PipelineState(session_id="s")
        state.pending_tool_calls = [
            {"tool_use_id": "u1", "tool_name": "EnterPlanMode", "tool_input": {}}
        ]
        await stage.execute(None, state)
        assert state.shared.get(PLAN_MODE_KEY) is True

    @pytest.mark.asyncio
    async def test_enter_then_exit_round_trip(self):
        reg = ToolRegistry()
        reg.register(EnterPlanModeTool())
        reg.register(ExitPlanModeTool())
        stage = ToolStage(registry=reg, executor=SequentialExecutor())
        state = PipelineState(session_id="s")

        state.pending_tool_calls = [
            {"tool_use_id": "u1", "tool_name": "EnterPlanMode", "tool_input": {}}
        ]
        await stage.execute(None, state)
        assert state.shared.get(PLAN_MODE_KEY) is True

        state.pending_tool_calls = [
            {"tool_use_id": "u2", "tool_name": "ExitPlanMode", "tool_input": {}}
        ]
        await stage.execute(None, state)
        assert state.shared.get(PLAN_MODE_KEY) is False


class TestCapabilitiesAndRegistry:
    def test_capabilities(self):
        caps = EnterPlanModeTool().capabilities({})
        assert caps.concurrency_safe is False
        assert caps.idempotent is True
        # Same shape for Exit
        assert ExitPlanModeTool().capabilities({}).idempotent is True

    def test_registered_in_meta_family(self):
        from geny_executor.tools.built_in import (
            BUILT_IN_TOOL_CLASSES,
            BUILT_IN_TOOL_FEATURES,
        )

        assert BUILT_IN_TOOL_CLASSES["EnterPlanMode"] is EnterPlanModeTool
        assert BUILT_IN_TOOL_CLASSES["ExitPlanMode"] is ExitPlanModeTool
        assert "EnterPlanMode" in BUILT_IN_TOOL_FEATURES["meta"]
        assert "ExitPlanMode" in BUILT_IN_TOOL_FEATURES["meta"]
