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
    assert insp.tool_binding_supported is True
    assert insp.model_override_supported is True
    assert insp.artifact_info.provides_stage is True
    # Every stage exposes *something* configurable (either schema, a slot, or a chain)
    assert insp.config_schema is not None or insp.strategy_slots or insp.strategy_chains, (
        f"{module_name} exposes no introspectable surface"
    )


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
