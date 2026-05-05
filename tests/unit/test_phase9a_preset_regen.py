"""Tests for Sub-phase 9a preset regen (S9a.5)."""

from __future__ import annotations

import pytest

from geny_executor.core.builder import PipelineBuilder
from geny_executor.core.presets import PipelinePresets
from geny_executor.stages.s11_tool_review import ToolReviewStage
from geny_executor.stages.s13_task_registry import TaskRegistryStage
from geny_executor.stages.s15_hitl import HITLStage
from geny_executor.stages.s19_summarize import SummarizeStage
from geny_executor.stages.s20_persist import PersistStage


SCAFFOLD_ORDERS = (11, 13, 15, 19, 20)


# ── Builder method round-trip ──────────────────────────────────────────


class TestBuilderScaffoldMethods:
    def test_with_tool_review_registers_stage(self):
        p = PipelineBuilder("t", api_key="k").with_tool_review().build()
        assert isinstance(p.get_stage(11), ToolReviewStage)

    def test_with_task_registry_registers_stage(self):
        p = PipelineBuilder("t", api_key="k").with_task_registry().build()
        assert isinstance(p.get_stage(13), TaskRegistryStage)

    def test_with_hitl_registers_stage(self):
        p = PipelineBuilder("t", api_key="k").with_hitl().build()
        assert isinstance(p.get_stage(15), HITLStage)

    def test_with_summarize_registers_stage(self):
        p = PipelineBuilder("t", api_key="k").with_summarize().build()
        assert isinstance(p.get_stage(19), SummarizeStage)

    def test_with_persist_registers_stage(self):
        p = PipelineBuilder("t", api_key="k").with_persist().build()
        assert isinstance(p.get_stage(20), PersistStage)

    def test_no_scaffold_method_no_registration(self):
        """Sanity: not calling the scaffold builders leaves slots unregistered."""
        p = PipelineBuilder("t", api_key="k").build()
        for order in SCAFFOLD_ORDERS:
            assert p.get_stage(order) is None


# ── Built-in PipelinePresets include scaffolds ─────────────────────────


class TestPipelinePresetsScaffoldOptIn:
    def test_agent_includes_all_scaffolds(self):
        p = PipelinePresets.agent(api_key="k")
        for order in SCAFFOLD_ORDERS:
            assert p.get_stage(order) is not None, f"agent preset missing scaffold {order}"

    def test_geny_vtuber_includes_all_scaffolds(self):
        p = PipelinePresets.geny_vtuber(api_key="k")
        for order in SCAFFOLD_ORDERS:
            assert p.get_stage(order) is not None, f"geny_vtuber missing scaffold {order}"

    def test_minimal_preset_does_not_register_scaffolds(self):
        """minimal stays minimal — opt-in matters."""
        p = PipelinePresets.minimal(api_key="k")
        for order in SCAFFOLD_ORDERS:
            assert p.get_stage(order) is None


# ── GenyPresets (memory/presets.py) include scaffolds ─────────────────


def _make_provider():
    """Bare-minimum MemoryProvider for preset construction smoke."""
    from geny_executor.memory.providers.ephemeral import EphemeralMemoryProvider

    return EphemeralMemoryProvider()


class TestGenyPresetsScaffoldOptIn:
    def test_worker_easy_includes_all_scaffolds(self):
        from geny_executor.memory.presets import GenyPresets

        p = GenyPresets.worker_easy(api_key="k", provider=_make_provider())
        for order in SCAFFOLD_ORDERS:
            assert p.get_stage(order) is not None, f"worker_easy missing scaffold {order}"

    def test_worker_adaptive_includes_all_scaffolds(self):
        from geny_executor.memory.presets import GenyPresets

        p = GenyPresets.worker_adaptive(api_key="k", provider=_make_provider())
        for order in SCAFFOLD_ORDERS:
            assert p.get_stage(order) is not None, f"worker_adaptive missing scaffold {order}"

    def test_vtuber_includes_all_scaffolds(self):
        from geny_executor.memory.presets import GenyPresets

        p = GenyPresets.vtuber(api_key="k", provider=_make_provider())
        for order in SCAFFOLD_ORDERS:
            assert p.get_stage(order) is not None, f"vtuber missing scaffold {order}"


# ── Describe shows all 21 slots after a preset build ───────────────────


class TestDescribeReports21:
    @pytest.mark.parametrize(
        "factory",
        [
            lambda: PipelinePresets.agent(api_key="k"),
            lambda: PipelinePresets.geny_vtuber(api_key="k"),
        ],
    )
    def test_describe_returns_21(self, factory):
        p = factory()
        descs = p.describe()
        assert len(descs) == 21
        active_orders = {d.order for d in descs if d.is_active}
        # Every scaffold should be active (registered) in the
        # full-blown presets.
        for order in SCAFFOLD_ORDERS:
            assert order in active_orders
