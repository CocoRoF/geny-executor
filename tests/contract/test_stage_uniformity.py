"""Stage Uniformity Contract tests (E1.8).

Every stage in geny-executor's 16-stage pipeline must satisfy the same
introspection and configuration contract, regardless of slot- vs. chain-
based architecture. These tests pin that contract so regressions fail fast.

Contract surface:
  1. name / order / category properties
  2. get_strategy_slots() → Dict[str, StrategySlot]
  3. get_strategy_chains() → Dict[str, SlotChain]
  4. describe() → StageDescription
  5. tool_binding → StageToolBinding (per-stage view of the registry)
  6. model_override → defaults to None (opt-in override)
  7. get_config_schema / get_config / update_config triad
  8. Chain stages (s04 guard, s14 emit) expose the expected chain name
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor import (
    ConfigSchema,
    SlotChain,
    Stage,
    StageDescription,
    StrategySlot,
)
from geny_executor.tools.stage_binding import StageToolBinding

from geny_executor.stages.s01_input.artifact.default.stage import InputStage
from geny_executor.stages.s02_context.artifact.default.stage import ContextStage
from geny_executor.stages.s03_system.artifact.default.stage import SystemStage
from geny_executor.stages.s04_guard.artifact.default.stage import GuardStage
from geny_executor.stages.s05_cache.artifact.default.stage import CacheStage
from geny_executor.stages.s06_api.artifact.default.stage import APIStage
from geny_executor.stages.s06_api.artifact.default.providers import MockProvider
from geny_executor.stages.s07_token.artifact.default.stage import TokenStage
from geny_executor.stages.s08_think.artifact.default.stage import ThinkStage
from geny_executor.stages.s09_parse.artifact.default.stage import ParseStage
from geny_executor.stages.s10_tool.artifact.default.stage import ToolStage
from geny_executor.stages.s11_agent.artifact.default.stage import AgentStage
from geny_executor.stages.s12_evaluate.artifact.default.stage import EvaluateStage
from geny_executor.stages.s13_loop.artifact.default.stage import LoopStage
from geny_executor.stages.s14_emit.artifact.default.stage import EmitStage
from geny_executor.stages.s15_memory.artifact.default.stage import MemoryStage
from geny_executor.stages.s16_yield.artifact.default.stage import YieldStage


# ── Stage Factory ────────────────────────────────────────────────


def _build_api_stage() -> APIStage:
    """APIStage needs a provider — use MockProvider to avoid network."""
    return APIStage(provider=MockProvider())


STAGE_FACTORIES = [
    (1, "input", InputStage),
    (2, "context", ContextStage),
    (3, "system", SystemStage),
    (4, "guard", GuardStage),
    (5, "cache", CacheStage),
    (6, "api", _build_api_stage),
    (7, "token", TokenStage),
    (8, "think", ThinkStage),
    (9, "parse", ParseStage),
    (10, "tool", ToolStage),
    (11, "agent", AgentStage),
    (12, "evaluate", EvaluateStage),
    (13, "loop", LoopStage),
    (14, "emit", EmitStage),
    (15, "memory", MemoryStage),
    (16, "yield", YieldStage),
]

VALID_CATEGORIES = {"ingress", "pre_flight", "execution", "decision", "egress"}


@pytest.fixture(params=STAGE_FACTORIES, ids=[f"s{o:02d}_{n}" for o, n, _ in STAGE_FACTORIES])
def stage(request) -> Stage:
    order, name, factory = request.param
    instance = factory()
    # attach the expected name/order for assertions
    instance.__expected_order__ = order  # type: ignore[attr-defined]
    instance.__expected_name__ = name  # type: ignore[attr-defined]
    return instance


# ── Core contract ────────────────────────────────────────────────


class TestStageUniformityContract:
    """Every stage must expose the full uniformity surface."""

    def test_identity_properties(self, stage: Stage) -> None:
        assert stage.name == stage.__expected_name__  # type: ignore[attr-defined]
        assert stage.order == stage.__expected_order__  # type: ignore[attr-defined]
        assert stage.category in VALID_CATEGORIES

    def test_strategy_slots_is_dict(self, stage: Stage) -> None:
        slots = stage.get_strategy_slots()
        assert isinstance(slots, dict)
        for key, value in slots.items():
            assert isinstance(key, str)
            assert isinstance(value, StrategySlot)

    def test_strategy_chains_is_dict(self, stage: Stage) -> None:
        chains = stage.get_strategy_chains()
        assert isinstance(chains, dict)
        for key, value in chains.items():
            assert isinstance(key, str)
            assert isinstance(value, SlotChain)

    def test_at_least_one_strategy_surface(self, stage: Stage) -> None:
        """A stage must expose at least one slot or one chain."""
        slots = stage.get_strategy_slots()
        chains = stage.get_strategy_chains()
        assert slots or chains, (
            f"Stage {stage.name} exposes no configurable strategies (neither slots nor chains)."
        )

    def test_describe_matches_identity(self, stage: Stage) -> None:
        desc = stage.describe()
        assert isinstance(desc, StageDescription)
        assert desc.name == stage.name
        assert desc.order == stage.order
        assert desc.category == stage.category

    def test_tool_binding_is_stage_scoped(self, stage: Stage) -> None:
        binding = stage.tool_binding
        assert isinstance(binding, StageToolBinding)
        assert binding.stage_order == stage.order
        # default = inherit everything
        assert binding.allowed is None
        assert binding.blocked is None
        assert binding.is_allowed("any_tool_name") is True

    def test_model_override_default_none(self, stage: Stage) -> None:
        assert stage.model_override is None

    def test_model_override_roundtrip(self, stage: Stage) -> None:
        sentinel = object()
        stage.model_override = sentinel
        assert stage.model_override is sentinel
        stage.model_override = None
        assert stage.model_override is None

    def test_config_schema_type(self, stage: Stage) -> None:
        schema = stage.get_config_schema()
        assert schema is None or isinstance(schema, ConfigSchema)

    def test_get_config_is_dict(self, stage: Stage) -> None:
        cfg = stage.get_config()
        assert isinstance(cfg, dict)

    def test_update_config_empty_is_noop(self, stage: Stage) -> None:
        # Empty update must not raise for any stage.
        stage.update_config({})


# ── Chain stage contract ─────────────────────────────────────────


class TestChainStageContract:
    """s04 Guard and s14 Emit must expose the canonical chain names."""

    def test_guard_stage_has_guards_chain(self) -> None:
        stage = GuardStage()
        chains = stage.get_strategy_chains()
        assert "guards" in chains, "GuardStage must expose a 'guards' chain"
        assert isinstance(chains["guards"], SlotChain)

    def test_emit_stage_has_emitters_chain(self) -> None:
        stage = EmitStage()
        chains = stage.get_strategy_chains()
        assert "emitters" in chains, "EmitStage must expose an 'emitters' chain"
        assert isinstance(chains["emitters"], SlotChain)


# ── Ordering contract ────────────────────────────────────────────


def test_stage_orders_are_unique_and_dense() -> None:
    """Each stage declares a distinct order; the set covers 1..16 exactly."""
    orders = []
    for order, _name, factory in STAGE_FACTORIES:
        stage = factory()
        orders.append(stage.order)
        assert stage.order == order
    assert sorted(orders) == list(range(1, 17))


def test_stage_names_are_unique() -> None:
    """No two stages share a name."""
    names = []
    for _order, _name, factory in STAGE_FACTORIES:
        stage = factory()
        names.append(stage.name)
    assert len(set(names)) == len(names)
