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


class TestNotYetRegistered:
    """Sanity: scaffolds exist as packages but aren't in STAGE_MODULES yet."""

    def test_stage_modules_unchanged(self):
        from geny_executor.core.artifact import STAGE_MODULES

        # Still 16 entries; new orders 11/13/15/19/20 not yet wired.
        assert len(STAGE_MODULES) == 16
        # The new order numbers map to legacy modules, not the
        # scaffolding ones — that mapping moves in S9a.3.
        assert STAGE_MODULES[11] == "s12_agent"
        assert STAGE_MODULES[13] == "s16_loop"
        assert STAGE_MODULES[15] == "s18_memory"

    def test_scaffolds_importable_via_artifact(self):
        # create_stage uses STAGE_MODULES, so it can't reach the
        # scaffolds yet by short name. Direct import still works:
        from geny_executor.stages.s11_tool_review.artifact.default import Stage

        assert Stage is ToolReviewStage
