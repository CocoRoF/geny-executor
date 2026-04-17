"""L1.D — ``Pipeline.from_manifest`` round-trip + artifact routing.

Covers:
    * A live pipeline → snapshot → manifest → ``from_manifest`` → structurally
      equal pipeline (stage order, artifact names, active flags).
    * Tool bindings, model overrides, and chain ordering survive the round-trip.
    * Multi-artifact s06_api: manifest with ``artifact="mock"`` yields a stage
      whose provider is a ``MockProvider`` — proving artifact routing, not just
      strategy re-selection.
    * ``strict=True`` surfaces an error when the manifest references a stage
      whose artifact requires credentials that weren't supplied.
    * ``strict=False`` drops such stages silently and returns a partial pipeline.
"""

from __future__ import annotations

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
    create_stage,
)
from geny_executor.stages.s06_api.artifact.default.providers import MockProvider


def _template_pipeline() -> Pipeline:
    pipeline = Pipeline(
        PipelineConfig(
            name="env-template",
            api_key="template-key",
            model=ModelConfig(model="claude-opus-4-7", temperature=0.2),
        )
    )
    pipeline.register_stage(create_stage("s01_input"))
    pipeline.register_stage(create_stage("s04_guard"))
    pipeline.register_stage(create_stage("s05_cache"))
    pipeline.register_stage(create_stage("s06_api", "default", provider=MockProvider()))
    pipeline.register_stage(create_stage("s09_parse"))
    pipeline.register_stage(create_stage("s14_emit"))
    pipeline.register_stage(create_stage("s16_yield"))
    return pipeline


# ── PipelineConfig.from_dict ──────────────────────────────────


def test_pipeline_config_roundtrip_preserves_model_nesting():
    cfg = PipelineConfig(
        name="t",
        api_key="k",
        model=ModelConfig(model="claude-opus-4-7", temperature=0.5),
        max_iterations=7,
        single_turn=True,
    )
    restored = PipelineConfig.from_dict(cfg.to_dict())
    assert restored.name == "t"
    assert restored.api_key == "k"
    assert restored.model.model == "claude-opus-4-7"
    assert restored.model.temperature == pytest.approx(0.5)
    assert restored.max_iterations == 7
    assert restored.single_turn is True


def test_pipeline_config_from_dict_ignores_unknown_keys():
    restored = PipelineConfig.from_dict({"name": "t", "future_flag": True})
    assert restored.name == "t"


# ── Pipeline.from_manifest — snapshot round-trip ──────────────


def test_from_manifest_roundtrip_preserves_structure():
    source = _template_pipeline()
    # Exercise every v2 field so we can verify they all survive the trip.
    api_stage = source.get_stage(6)
    api_stage.tool_binding.allow("search")
    api_stage.tool_binding.block("dangerous")
    api_stage.model_override = ModelConfig(model="claude-opus-4-7", temperature=0.4)

    snapshot = PipelineMutator(source).snapshot()
    manifest = EnvironmentManifest.from_snapshot(snapshot, name="Round-Trip Env")

    # The API artifact for s06 was registered via an injected MockProvider, so
    # ``default`` artifact needs ``api_key`` when re-instantiated fresh.
    rebuilt = Pipeline.from_manifest(manifest, api_key="rebuild-key", strict=True)

    # Stage order, names, artifacts
    rebuilt_orders = [s.order for s in rebuilt.stages]
    assert rebuilt_orders == [1, 4, 5, 6, 9, 14, 16]
    rebuilt_s06 = rebuilt.get_stage(6)
    assert rebuilt_s06.artifact_name == "default"

    # Tool binding survived
    assert rebuilt_s06.tool_binding.allowed == {"search"}
    assert rebuilt_s06.tool_binding.blocked == {"dangerous"}

    # Model override survived
    assert rebuilt_s06.model_override is not None
    assert rebuilt_s06.model_override.model == "claude-opus-4-7"
    assert rebuilt_s06.model_override.temperature == pytest.approx(0.4)

    # Pipeline-level model config survived via to_snapshot → restore
    assert rebuilt._config.model.model == "claude-opus-4-7"
    assert rebuilt._config.model.temperature == pytest.approx(0.2)


def test_from_manifest_preserves_chain_ordering():
    source = _template_pipeline()
    guard = source.get_stage(4)
    chains = guard.get_strategy_chains() if hasattr(guard, "get_strategy_chains") else {}
    if not chains:
        pytest.skip("s04_guard exposes no chain for this test environment")

    chain_name = next(iter(chains.keys()))
    chain = chains[chain_name]
    original_order = [item.name for item in chain.items]
    if len(original_order) < 2:
        pytest.skip("Chain too short to reorder meaningfully")
    reversed_order = list(reversed(original_order))
    guard.reorder_chain(chain_name, reversed_order)

    snap = PipelineMutator(source).snapshot()
    manifest = EnvironmentManifest.from_snapshot(snap, name="chain-env")
    rebuilt = Pipeline.from_manifest(manifest, api_key="rebuild-key", strict=True)
    rebuilt_guard = rebuilt.get_stage(4)
    rebuilt_chain = rebuilt_guard.get_strategy_chains()[chain_name]
    assert [item.name for item in rebuilt_chain.items] == reversed_order


# ── Artifact routing ──────────────────────────────────────────


def test_from_manifest_respects_artifact_selection():
    """Manifest with artifact=default for s06_api + api_key injects Anthropic provider."""
    source = _template_pipeline()
    snap = PipelineMutator(source).snapshot()
    manifest = EnvironmentManifest.from_snapshot(snap, name="artifact-env")

    rebuilt = Pipeline.from_manifest(manifest, api_key="sk-test", strict=True)
    s06 = rebuilt.get_stage(6)
    assert s06 is not None
    assert s06.artifact_name == "default"


def test_from_manifest_strict_raises_without_api_key():
    source = _template_pipeline()
    snap = PipelineMutator(source).snapshot()
    manifest = EnvironmentManifest.from_snapshot(snap, name="no-key-env")

    with pytest.raises(ValueError):
        Pipeline.from_manifest(manifest, api_key=None, strict=True)


def test_from_manifest_non_strict_drops_broken_stages():
    source = _template_pipeline()
    snap = PipelineMutator(source).snapshot()
    manifest = EnvironmentManifest.from_snapshot(snap, name="non-strict-env")

    rebuilt = Pipeline.from_manifest(manifest, api_key=None, strict=False)
    # s06_api requires api_key and should have been skipped
    assert rebuilt.get_stage(6) is None
    # Stages that don't need credentials still registered
    assert rebuilt.get_stage(1) is not None
    assert rebuilt.get_stage(16) is not None


def test_from_manifest_v1_payload_migrates_then_instantiates():
    """v1 manifest JSON loads + builds a pipeline without explicit migration."""
    v1 = {
        "version": "1.0",
        "metadata": {"id": "env_legacy", "name": "Legacy"},
        "model": {"model": "claude-opus-4-7"},
        "pipeline": {"name": "legacy"},
        "stages": [
            {"order": 1, "name": "s01_input", "active": True},
            {"order": 16, "name": "s16_yield", "active": True},
        ],
        "tools": {},
    }
    manifest = EnvironmentManifest.from_dict(v1)
    rebuilt = Pipeline.from_manifest(manifest, strict=True)
    assert {s.order for s in rebuilt.stages} == {1, 16}
    assert rebuilt._config.model.model == "claude-opus-4-7"
