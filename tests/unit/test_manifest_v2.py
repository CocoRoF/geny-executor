"""L1.C — EnvironmentManifest v2 + PipelineSnapshot v2 + v1 migration.

Covers:
    * PipelineSnapshot gains ``artifact``, ``tool_binding``,
      ``model_override``, and ``chain_order`` fields per stage.
    * Missing v2 fields in a v1 payload load with safe defaults.
    * EnvironmentManifest.from_dict silently upgrades v1 → v2.
    * StageManifestEntry round-trips.
    * PipelineMutator.snapshot captures the new fields; restore applies them
      without wiping pre-existing overrides on v1 payloads.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor import (
    EnvironmentManifest,
    ModelConfig,
    Pipeline,
    PipelineConfig,
    PipelineMutator,
    StageManifestEntry,
    create_stage,
)
from geny_executor.core.environment import MANIFEST_VERSION
from geny_executor.core.snapshot import PipelineSnapshot, StageSnapshot
from geny_executor.stages.s06_api.artifact.default.providers import MockProvider


# ── StageSnapshot v2 ───────────────────────────────────────────


def test_stage_snapshot_defaults_new_fields():
    s = StageSnapshot(order=5, name="cache", is_active=True)
    assert s.artifact == "default"
    assert s.tool_binding is None
    assert s.model_override is None
    assert s.chain_order == {}


def test_pipeline_snapshot_version_defaults_to_v2():
    assert PipelineSnapshot(pipeline_name="x").version == "2.0"


def test_pipeline_snapshot_roundtrip_preserves_v2_fields():
    snap = PipelineSnapshot(
        pipeline_name="t",
        stages=[
            StageSnapshot(
                order=5,
                name="cache",
                is_active=True,
                artifact="redis",
                tool_binding={
                    "stage_order": 5,
                    "allowed": ["a"],
                    "blocked": None,
                    "extra_context": {},
                },
                chain_order={"guards": ["safety", "policy"]},
                model_override={"model": "claude-opus-4-7"},
            )
        ],
    )
    restored = PipelineSnapshot.from_dict(snap.to_dict())
    assert restored.stages[0].artifact == "redis"
    assert restored.stages[0].tool_binding["allowed"] == ["a"]
    assert restored.stages[0].chain_order == {"guards": ["safety", "policy"]}
    assert restored.stages[0].model_override == {"model": "claude-opus-4-7"}


def test_pipeline_snapshot_loads_v1_payload_silently():
    """Missing v2 fields fall back to safe defaults."""
    v1 = {
        "version": "1.0",
        "pipeline_name": "legacy",
        "stages": [
            {
                "order": 1,
                "name": "input",
                "is_active": True,
                "strategies": {"validator": "default"},
            },
        ],
        "pipeline_config": {},
        "model_config": {},
    }
    snap = PipelineSnapshot.from_dict(v1)
    assert snap.version == "1.0"  # field preserved as declared
    assert snap.stages[0].artifact == "default"
    assert snap.stages[0].tool_binding is None
    assert snap.stages[0].chain_order == {}


# ── StageManifestEntry ─────────────────────────────────────────


def test_stage_manifest_entry_roundtrip():
    e = StageManifestEntry(
        order=6,
        name="api",
        artifact="openai",
        strategies={"provider": "openai"},
        config={"stream": True},
        tool_binding={
            "stage_order": 6,
            "allowed": ["search"],
            "blocked": None,
            "extra_context": {},
        },
        model_override={"model": "claude-opus-4-7"},
        chain_order={"guards": ["a", "b"]},
    )
    restored = StageManifestEntry.from_dict(e.to_dict())
    assert restored == e


# ── EnvironmentManifest v1 → v2 migration ─────────────────────


def test_environment_manifest_default_version_is_v2():
    assert EnvironmentManifest().version == MANIFEST_VERSION


def test_environment_manifest_migrates_v1_silently():
    v1 = {
        "version": "1.0",
        "metadata": {
            "id": "env_x",
            "name": "Legacy",
            "description": "",
            "tags": [],
            "created_at": "",
            "updated_at": "",
            "base_preset": "agent",
        },
        "model": {"model": "claude-opus-4-7"},
        "pipeline": {},
        "stages": [
            {
                "order": 5,
                "name": "cache",
                "active": True,
                "strategies": {"strategy": "default"},
                "strategy_configs": {},
                "config": {"cache_prefix": "old_"},
            }
        ],
        "tools": {"built_in": ["search"]},
    }
    m = EnvironmentManifest.from_dict(v1)
    assert m.version == MANIFEST_VERSION
    entry = m.stages[0]
    assert entry["artifact"] == "default"
    assert entry["tool_binding"] is None
    assert entry["model_override"] is None
    assert entry["chain_order"] == {}
    # Preserves original payload
    assert entry["config"] == {"cache_prefix": "old_"}
    assert entry["strategies"] == {"strategy": "default"}


def test_environment_manifest_v2_roundtrip_idempotent():
    v1 = {
        "version": "1.0",
        "metadata": {"id": "env_x", "name": "Legacy"},
        "model": {},
        "pipeline": {},
        "stages": [{"order": 1, "name": "input", "active": True}],
        "tools": {},
    }
    v2_once = EnvironmentManifest.from_dict(v1)
    v2_twice = EnvironmentManifest.from_dict(v2_once.to_dict())
    assert v2_once.to_dict() == v2_twice.to_dict()


def test_environment_manifest_stage_entries_helper():
    v1 = {
        "version": "1.0",
        "metadata": {"id": "env_x"},
        "stages": [{"order": 1, "name": "input", "active": True}],
    }
    m = EnvironmentManifest.from_dict(v1)
    entries = m.stage_entries()
    assert len(entries) == 1
    assert entries[0].order == 1
    entries[0].artifact = "custom"
    m.set_stage_entries(entries)
    assert m.stages[0]["artifact"] == "custom"


# ── PipelineMutator snapshot / restore round-trip ─────────────


def _mini_pipeline() -> Pipeline:
    pipeline = Pipeline(PipelineConfig(name="v2-test", api_key="dummy"))
    pipeline.register_stage(create_stage("s01_input"))
    pipeline.register_stage(create_stage("s05_cache"))
    pipeline.register_stage(create_stage("s06_api", "default", provider=MockProvider()))
    return pipeline


def test_mutator_snapshot_captures_v2_fields():
    pipeline = _mini_pipeline()
    stage = pipeline.get_stage(6)
    stage.tool_binding.allow("search")
    stage.tool_binding.block("delete")
    stage.model_override = ModelConfig(model="claude-opus-4-7", temperature=0.3)

    mut = PipelineMutator(pipeline)
    snap = mut.snapshot()
    s06 = next(s for s in snap.stages if s.order == 6)
    assert s06.artifact == "default"
    assert s06.tool_binding == {
        "stage_order": 6,
        "allowed": ["search"],
        "blocked": ["delete"],
        "extra_context": {},
    }
    assert s06.model_override is not None
    assert s06.model_override["model"] == "claude-opus-4-7"
    assert s06.model_override["temperature"] == pytest.approx(0.3)


def test_mutator_restore_applies_v2_fields():
    pipeline = _mini_pipeline()
    stage = pipeline.get_stage(6)
    stage.tool_binding.allow("search")
    stage.model_override = ModelConfig(model="claude-opus-4-7", temperature=0.3)

    snap = PipelineMutator(pipeline).snapshot()

    # Fresh pipeline — should acquire the v2 state from the snapshot.
    fresh = _mini_pipeline()
    PipelineMutator(fresh).restore(snap)
    restored = fresh.get_stage(6)
    assert restored.tool_binding.allowed == {"search"}
    assert restored.model_override is not None
    assert restored.model_override.model == "claude-opus-4-7"
    assert restored.model_override.temperature == pytest.approx(0.3)


def test_mutator_restore_v1_snapshot_preserves_live_overrides():
    """v1 snapshots lack model_override; restore must not wipe a live override."""
    snap = PipelineMutator(_mini_pipeline()).snapshot()
    # Simulate downgrading the payload back to v1 (as if loaded from ancient disk)
    data = snap.to_dict()
    data["version"] = "1.0"
    for s in data["stages"]:
        for k in ("artifact", "tool_binding", "model_override", "chain_order"):
            s.pop(k, None)
    v1_snap = PipelineSnapshot.from_dict(data)

    fresh = _mini_pipeline()
    fresh.get_stage(6).model_override = ModelConfig(model="seed-model")
    PipelineMutator(fresh).restore(v1_snap)
    assert fresh.get_stage(6).model_override.model == "seed-model"


def test_snapshot_json_is_valid_utf8():
    snap = PipelineMutator(_mini_pipeline()).snapshot()
    text = snap.to_json()
    # Round-trips cleanly without custom encoders
    restored = PipelineSnapshot.from_json(text)
    assert restored.version == "2.0"
    # Also safe for json.loads
    json.loads(text)


# ── EnvironmentManifest.blank_manifest ────────────────────────


REQUIRED_STAGE_ORDERS = {1, 6, 9, 16}


def test_blank_manifest_returns_16_stages_with_required_ones_active():
    """Required stages (s01_input, s06_api, s09_parse, s16_yield) default
    active=True so a blank env is runnable without forcing the user to flip
    the load-bearing four. The other twelve remain active=False."""
    m = EnvironmentManifest.blank_manifest("Blank Env")
    entries = m.stage_entries()
    assert [e.order for e in entries] == list(range(1, 17))
    active_orders = {e.order for e in entries if e.active}
    assert active_orders == REQUIRED_STAGE_ORDERS


def test_blank_manifest_optional_stages_default_inactive():
    """Every non-required stage must default active=False — users opt in."""
    m = EnvironmentManifest.blank_manifest("Blank Env")
    inactive_orders = {e.order for e in m.stage_entries() if not e.active}
    assert inactive_orders == set(range(1, 17)) - REQUIRED_STAGE_ORDERS


def test_blank_manifest_uses_default_artifact_per_stage():
    m = EnvironmentManifest.blank_manifest("Blank Env")
    for entry in m.stage_entries():
        # introspect_all() uses DEFAULT_ARTIFACT when no overrides given
        assert entry.artifact == "default"


def test_blank_manifest_seeds_strategy_current_impls():
    """Each stage captures its artifact's default strategy picks so toggling
    active doesn't produce a manifest that fails to rehydrate.

    The provider slot on s06_api must resolve to ``"anthropic"`` — the real
    runtime default. Introspection historically injected a ``MockProvider``
    to stay session-less, which bled into manifests and made new envs hit
    the test-only mock at run time.
    """
    m = EnvironmentManifest.blank_manifest("Blank Env")
    s06 = next(e for e in m.stage_entries() if e.order == 6)
    assert s06.strategies.get("provider") == "anthropic"


def test_blank_manifest_metadata_has_no_base_preset():
    m = EnvironmentManifest.blank_manifest("Blank Env", description="seed")
    assert m.metadata.name == "Blank Env"
    assert m.metadata.description == "seed"
    assert m.metadata.base_preset == ""
    assert m.metadata.id.startswith("env_")
    assert m.metadata.created_at
    assert m.metadata.updated_at == m.metadata.created_at


def test_blank_manifest_accepts_optional_tags_model_pipeline():
    m = EnvironmentManifest.blank_manifest(
        "Tagged",
        tags=["scratch", "experiment"],
        model={"model": "claude-opus-4-7"},
        pipeline={"single_turn": True},
    )
    assert m.metadata.tags == ["scratch", "experiment"]
    assert m.model == {"model": "claude-opus-4-7"}
    assert m.pipeline == {"single_turn": True}


def test_blank_manifest_roundtrips_through_json():
    original = EnvironmentManifest.blank_manifest("RT")
    restored = EnvironmentManifest.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored.version == MANIFEST_VERSION
    assert len(restored.stages) == 16
    assert restored.metadata.base_preset == ""


def test_blank_manifest_builds_minimal_pipeline_via_from_manifest():
    """blank_manifest turns on the required four stages (Input/API/Parse/
    Yield), so Pipeline.from_manifest on a fresh blank env already
    registers exactly those four — matching the ``minimal`` preset."""
    m = EnvironmentManifest.blank_manifest("Blank")
    rebuilt = Pipeline.from_manifest(m, api_key="dummy", strict=False)
    assert {s.order for s in rebuilt.stages} == REQUIRED_STAGE_ORDERS


def test_blank_manifest_rebuild_uses_real_provider_with_supplied_api_key():
    """Regression: a blank manifest rehydrated via ``Pipeline.from_manifest``
    must land on AnthropicProvider carrying the caller's api_key. Earlier
    snapshots recorded ``provider: mock`` (from introspection) and the
    restore pass swapped in MockProvider at run time, so sessions silently
    answered with ``"Mock response"`` instead of calling the real API.
    """
    from geny_executor.stages.s06_api.artifact.default.providers import (
        AnthropicProvider,
    )

    m = EnvironmentManifest.blank_manifest("Blank")
    rebuilt = Pipeline.from_manifest(m, api_key="sk-test-key")
    api_stage = rebuilt.get_stage(6)
    provider = api_stage.get_strategy_slots()["provider"].strategy
    assert isinstance(provider, AnthropicProvider)
    assert provider._api_key == "sk-test-key"


def test_blank_manifest_extra_activation_then_rebuild_succeeds():
    """Flipping an optional stage on top of the required defaults adds it
    to the rebuilt pipeline alongside the required four."""
    m = EnvironmentManifest.blank_manifest("Blank")
    entries = m.stage_entries()
    for e in entries:
        if e.order == 5:  # cache — optional
            e.active = True
    m.set_stage_entries(entries)

    rebuilt = Pipeline.from_manifest(m, api_key="dummy", strict=True)
    assert {s.order for s in rebuilt.stages} == REQUIRED_STAGE_ORDERS | {5}
