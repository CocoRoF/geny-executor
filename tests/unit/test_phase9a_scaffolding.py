"""Tests for Sub-phase 9a scaffolding stages.

Originally all five stages were pure pass-throughs. Sub-phase 9b
promotes them one at a time:

  * S9b.1 — tool_review (chain of reviewers)
  * S9b.2 — task_registry (registry + policy slots)
  * S9b.3+ — hitl / summarize / persist still pass-through-ish

The metadata + ``execute`` round-trip checks below skip the stages
that have been promoted (they have their own dedicated tests).
"""

from __future__ import annotations

import pytest

from geny_executor.core.state import PipelineState
from geny_executor.stages.s11_tool_review import ToolReviewStage
from geny_executor.stages.s13_task_registry import TaskRegistryStage
from geny_executor.stages.s15_hitl import HITLStage
from geny_executor.stages.s19_summarize import SummarizeStage
from geny_executor.stages.s20_persist import PersistStage


# All five still report the same identity; only the bodies have grown.
IDENTITY_CASES = [
    (ToolReviewStage, "tool_review", 11, "review"),
    (TaskRegistryStage, "task_registry", 13, "orchestration"),
    (HITLStage, "hitl", 15, "gate"),
    (SummarizeStage, "summarize", 19, "finalize"),
    (PersistStage, "persist", 20, "finalize"),
]

# Sub-phase 9b is complete — every former scaffold has real
# behaviour now. The pass-through tests below were the original
# zero-body sanity checks; the per-stage `test_s9b*_*.py` files
# now own those guarantees.


@pytest.mark.parametrize("cls,name,order,category", IDENTITY_CASES)
class TestScaffoldIdentity:
    def test_name(self, cls, name, order, category):
        assert cls().name == name

    def test_order(self, cls, name, order, category):
        assert cls().order == order

    def test_category(self, cls, name, order, category):
        assert cls().category == category


class TestHITLBypass:
    def test_hitl_bypasses_when_no_request(self):
        # S9b.3: HITL still bypasses when there's no pending request,
        # so pipelines that don't opt in see no behaviour change.
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
