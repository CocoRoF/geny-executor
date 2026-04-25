"""Tests for Sub-phase 9a scaffolding stages.

These five new stages reserve the slots that Sub-phase 9b will fill
with real behaviour. For now each one is a pass-through (or always-
bypass for HITL). The pipeline registry / introspection / capability
matrix are NOT yet aware of them — those updates land in S9a.3.
"""

from __future__ import annotations

import pytest

from geny_executor.core.state import PipelineState
from geny_executor.stages.s11_tool_review import ToolReviewStage
from geny_executor.stages.s13_task_registry import TaskRegistryStage
from geny_executor.stages.s15_hitl import HITLStage
from geny_executor.stages.s19_summarize import SummarizeStage
from geny_executor.stages.s20_persist import PersistStage


SCAFFOLDS = [
    (ToolReviewStage, "tool_review", 11, "review"),
    (TaskRegistryStage, "task_registry", 13, "orchestration"),
    (HITLStage, "hitl", 15, "gate"),
    (SummarizeStage, "summarize", 19, "finalize"),
    (PersistStage, "persist", 20, "finalize"),
]


@pytest.mark.parametrize("cls,name,order,category", SCAFFOLDS)
class TestScaffoldMetadata:
    def test_name(self, cls, name, order, category):
        assert cls().name == name

    def test_order(self, cls, name, order, category):
        assert cls().order == order

    def test_category(self, cls, name, order, category):
        assert cls().category == category

    def test_no_strategy_slots(self, cls, name, order, category):
        assert cls().get_strategy_slots() == {}


@pytest.mark.parametrize("cls", [c for c, *_ in SCAFFOLDS])
class TestScaffoldExecution:
    @pytest.mark.asyncio
    async def test_returns_input_unchanged(self, cls):
        state = PipelineState(session_id="s")
        result = await cls().execute(input="passthrough-marker", state=state)
        assert result == "passthrough-marker"

    @pytest.mark.asyncio
    async def test_does_not_mutate_state(self, cls):
        state = PipelineState(session_id="s")
        before_msgs = list(state.messages)
        before_meta = dict(state.metadata)
        await cls().execute(input=None, state=state)
        assert state.messages == before_msgs
        assert state.metadata == before_meta


class TestHITLBypass:
    def test_hitl_always_bypasses(self):
        # Sub-phase 9a scaffolding: HITL never blocks the pipeline.
        assert HITLStage().should_bypass(PipelineState()) is True


class TestRegistration:
    """Sanity: S9a.3 wired the scaffolds into STAGE_MODULES."""

    def test_stage_modules_now_21_entries(self):
        from geny_executor.core.artifact import STAGE_MODULES

        assert len(STAGE_MODULES) == 21
        # Each new order points at its scaffolding module.
        assert STAGE_MODULES[11] == "s11_tool_review"
        assert STAGE_MODULES[13] == "s13_task_registry"
        assert STAGE_MODULES[15] == "s15_hitl"
        assert STAGE_MODULES[19] == "s19_summarize"
        assert STAGE_MODULES[20] == "s20_persist"

    def test_scaffolds_importable_via_artifact(self):
        from geny_executor.stages.s11_tool_review.artifact.default import Stage

        assert Stage is ToolReviewStage

    def test_create_stage_resolves_scaffolds(self):
        from geny_executor.core.artifact import create_stage

        s = create_stage("s11_tool_review")
        assert isinstance(s, ToolReviewStage)
        assert s.order == 11
