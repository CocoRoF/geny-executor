"""L1.B — Session-less stage introspection.

Verifies the public surface of :mod:`geny_executor.core.introspection`:

    * ``introspect_stage`` returns a populated :class:`StageIntrospection` for
      every default artifact in the 16-stage pipeline, without touching any
      external system.
    * Per-slot / per-chain registries surface their impl schemas.
    * ``introspect_all`` yields exactly 16 introspections ordered 1-16.
    * ``introspect_all`` with overrides routes to the correct artifact.
    * Strategy-only artifacts (``Stage = None``) raise
      :class:`IntrospectionUnsupported` for ``introspect_stage`` but fall back
      to default when called through ``introspect_all``.
    * Non-default artifacts that need ctor secrets (``s06_api/openai``) are
      introspected via library-owned mocks.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor import (
    ChainIntrospection,
    ConfigSchema,
    IntrospectionUnsupported,
    StageIntrospection,
    introspect_all,
    introspect_stage,
)
from geny_executor.core.artifact import STAGE_MODULES


# ── introspect_stage on every default artifact ─────────────────


@pytest.mark.parametrize("order,module_name", sorted(STAGE_MODULES.items()))
def test_introspect_stage_default_every_stage(order: int, module_name: str):
    """Every default artifact must introspect without raising."""
    insp = introspect_stage(module_name, "default")
    assert isinstance(insp, StageIntrospection)
    assert insp.stage == module_name
    assert insp.artifact == "default"
    assert insp.order == order
    assert insp.artifact_info.provides_stage is True
    # Every stage exposes *something* configurable (either schema, a slot, or a chain)
    assert insp.config_schema is not None or insp.strategy_slots or insp.strategy_chains, (
        f"{module_name} exposes no introspectable surface"
    )


# ── Per-stage capability flags (honest plumbing) ─────────────────
#
# A UI that offers model/tool-binding inputs on every stage is misleading —
# the runtime only honours them on the API and tool stages respectively.
# These tests pin the contract so regressions can't quietly return.


_MODEL_OVERRIDE_STAGES = {"s02_context", "s06_api", "s15_memory"}
_TOOL_BINDING_STAGES = {"s10_tool"}


def test_capability_flags_api_stage_only_allows_model_override():
    insp = introspect_stage("s06_api", "default")
    assert insp.model_override_supported is True
    assert insp.tool_binding_supported is False


def test_capability_flags_tool_stage_only_allows_tool_binding():
    insp = introspect_stage("s10_tool", "default")
    assert insp.tool_binding_supported is True
    assert insp.model_override_supported is False


def test_capability_flags_context_stage_allows_model_override():
    """s02 consumes the override via LLMSummaryCompactor."""
    insp = introspect_stage("s02_context", "default")
    assert insp.model_override_supported is True
    assert insp.tool_binding_supported is False


def test_capability_flags_memory_stage_allows_model_override():
    """s15 consumes the override via GenyMemoryStrategy native reflection."""
    insp = introspect_stage("s15_memory", "default")
    assert insp.model_override_supported is True
    assert insp.tool_binding_supported is False


@pytest.mark.parametrize(
    "order,module_name",
    [
        (o, m)
        for o, m in sorted(STAGE_MODULES.items())
        if m not in _MODEL_OVERRIDE_STAGES and m not in _TOOL_BINDING_STAGES
    ],
)
def test_capability_flags_default_false_elsewhere(order: int, module_name: str):
    insp = introspect_stage(module_name, "default")
    assert insp.tool_binding_supported is False, (
        f"{module_name} claims tool_binding support but its runtime ignores the binding"
    )
    assert insp.model_override_supported is False, (
        f"{module_name} claims model_override support but its runtime ignores the override"
    )


# ── Structurally required stages ─────────────────────────────────
#
# Input → API → Parse → Yield is the ``minimal`` preset. These 4 stages must
# always stay active — UIs read ``required`` to force the Active toggle on.

_REQUIRED_STAGES = {"s01_input", "s06_api", "s09_parse", "s16_yield"}


@pytest.mark.parametrize("module_name", sorted(_REQUIRED_STAGES))
def test_required_flag_true_for_structurally_required_stages(module_name: str):
    insp = introspect_stage(module_name, "default")
    assert insp.required is True, (
        f"{module_name} belongs to the minimum Input/API/Parse/Yield set but reports required=False"
    )


@pytest.mark.parametrize(
    "module_name",
    sorted({m for m in STAGE_MODULES.values() if m not in _REQUIRED_STAGES}),
)
def test_required_flag_false_for_optional_stages(module_name: str):
    insp = introspect_stage(module_name, "default")
    assert insp.required is False, (
        f"{module_name} is not structurally required but reports required=True — "
        "UIs will refuse to let users deactivate it"
    )


def test_required_flag_serializes_in_to_dict():
    insp = introspect_stage("s01_input", "default")
    payload = insp.to_dict()
    assert payload["required"] is True


def test_introspect_stage_accepts_alias_and_order():
    by_module = introspect_stage("s06_api", "default")
    by_alias = introspect_stage("api", "default")
    by_order = introspect_stage("6", "default")
    assert by_module.stage == by_alias.stage == by_order.stage == "s06_api"


def test_introspect_stage_slot_schema_keys_match_available_impls():
    """Every ``impl_schemas`` key mirrors ``available_impls`` exactly."""
    insp = introspect_stage("s05_cache", "default")
    for slot in insp.strategy_slots.values():
        assert set(slot.impl_schemas.keys()) == set(slot.available_impls), (
            f"slot '{slot.slot_name}' schema/impls mismatch"
        )
        assert set(slot.impl_descriptions.keys()) == set(slot.available_impls)
        for schema in slot.impl_schemas.values():
            assert schema is None or isinstance(schema, ConfigSchema)


def test_introspect_stage_chain_schemas_match_available_impls():
    """Chain stages (s04/s14) expose impl schemas for every registered strategy."""
    for module_name in ("s04_guard", "s14_emit"):
        insp = introspect_stage(module_name, "default")
        assert insp.strategy_chains, f"{module_name} must expose at least one chain"
        for chain in insp.strategy_chains.values():
            assert isinstance(chain, ChainIntrospection)
            assert set(chain.impl_schemas.keys()) == set(chain.available_impls)
            for schema in chain.impl_schemas.values():
                assert schema is None or isinstance(schema, ConfigSchema)


# ── Non-default artifact: OpenAI must introspect without network ───


def test_introspect_stage_openai_uses_dummy_key():
    """s06_api/openai ctor requires api_key — library injects a dummy."""
    insp = introspect_stage("s06_api", "openai")
    assert insp.artifact == "openai"
    assert insp.stage == "s06_api"
    assert insp.artifact_info.provides_stage is True
    # slot registry still surfaces
    assert insp.strategy_slots


# ── Strategy-only artifact handling ─────────────────────────────


def test_introspect_stage_raises_for_strategy_only_artifact():
    """Adaptive evaluate is strategy-only (Stage = None)."""
    with pytest.raises(IntrospectionUnsupported):
        introspect_stage("s12_evaluate", "adaptive")


def test_introspect_all_falls_back_to_default_on_strategy_only():
    """introspect_all must never raise on a well-formed override map."""
    results = introspect_all({"s12_evaluate": "adaptive"})
    # Find s12 entry and verify it fell back to default
    s12 = next(r for r in results if r.stage == "s12_evaluate")
    assert s12.artifact == "default"


# ── introspect_all contract ─────────────────────────────────────


def test_introspect_all_returns_16_in_order():
    results = introspect_all()
    assert len(results) == 16
    assert [r.order for r in results] == list(range(1, 17))
    assert all(r.artifact == "default" for r in results)


def test_introspect_all_with_overrides():
    results = introspect_all({"s06_api": "openai"})
    s06 = next(r for r in results if r.stage == "s06_api")
    assert s06.artifact == "openai"
    # other stages untouched
    for r in results:
        if r.stage != "s06_api":
            assert r.artifact == "default"


# ── Serialization: to_dict is JSON-safe ────────────────────────


def test_stage_introspection_to_dict_is_json_safe():
    """to_dict output must round-trip through json.dumps without custom encoders."""
    import json

    insp = introspect_stage("s05_cache", "default")
    payload = insp.to_dict()
    serialized = json.dumps(payload)
    restored = json.loads(serialized)
    assert restored["stage"] == "s05_cache"
    assert restored["artifact"] == "default"
    assert "strategy_slots" in restored
    assert "strategy_chains" in restored
