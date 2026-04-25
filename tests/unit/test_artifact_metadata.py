"""L1.A — Artifact metadata + Stage provenance + serialization helpers.

Covers the public surface introduced in v0.13.0 L1.A:

    * ``describe_artifact()`` reads the optional ``ARTIFACT_META`` dict and
      falls back to sensible defaults.
    * ``list_artifacts_with_meta()`` enumerates artifacts with metadata.
    * ``create_stage()`` stamps ``_artifact_name`` / ``_stage_module`` on the
      instance and ``Stage.artifact_name`` / ``Stage.stage_module`` surface them.
    * ``StageToolBinding.to_dict``/``from_dict`` round-trip.
    * ``ModelConfig.to_dict``/``from_dict`` round-trip.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor import (
    ArtifactInfo,
    ModelConfig,
    create_stage,
    describe_artifact,
    list_artifacts,
    list_artifacts_with_meta,
)
from geny_executor.core.artifact import STAGE_MODULES, DEFAULT_ARTIFACT
from geny_executor.tools.stage_binding import StageToolBinding


# ── describe_artifact / list_artifacts_with_meta ────────────────


def test_describe_artifact_defaults_for_missing_meta():
    """Artifacts without ``ARTIFACT_META`` still produce a usable ArtifactInfo."""
    info = describe_artifact("s01_input", "default")
    assert isinstance(info, ArtifactInfo)
    assert info.stage == "s01_input"
    assert info.name == "default"
    assert info.is_default is True
    assert info.stability == "stable"
    assert info.version == "1.0"
    assert info.requires == ()
    assert info.provides_stage is True
    assert info.extra == {}


def test_describe_artifact_flags_strategy_only_artifacts():
    """``Stage = None`` artifacts (e.g. evaluate/adaptive) report provides_stage=False."""
    info = describe_artifact("s14_evaluate", "adaptive")
    assert info.provides_stage is False
    # default artifact remains fully instantiable
    assert describe_artifact("s14_evaluate", "default").provides_stage is True


def test_describe_artifact_accepts_all_stage_identifier_forms():
    """module name / alias / int / digit string all resolve the same way."""
    by_module = describe_artifact("s01_input", "default")
    by_alias = describe_artifact("input", "default")
    by_order = describe_artifact("1", "default")
    assert by_module == by_alias == by_order


def test_describe_artifact_raises_on_bad_meta_type(monkeypatch):
    """Non-dict ``ARTIFACT_META`` is a programmer error, not silent."""
    from geny_executor.stages.s01_input.artifact import default as default_mod

    monkeypatch.setattr(default_mod, "ARTIFACT_META", ["not", "a", "dict"], raising=False)
    with pytest.raises(TypeError, match="must be a dict"):
        describe_artifact("s01_input", "default")


def test_describe_artifact_captures_unknown_keys_as_extra(monkeypatch):
    """Unknown meta keys land in ``extra`` instead of being dropped."""
    from geny_executor.stages.s01_input.artifact import default as default_mod

    fake_meta = {
        "description": "Test input",
        "version": "2.0",
        "stability": "beta",
        "requires": ["python>=3.11"],
        "author": "qa@example.com",
        "ui_hint": "avoid-in-prod",
    }
    monkeypatch.setattr(default_mod, "ARTIFACT_META", fake_meta, raising=False)

    info = describe_artifact("s01_input", "default")
    assert info.description == "Test input"
    assert info.version == "2.0"
    assert info.stability == "beta"
    assert info.requires == ("python>=3.11",)
    assert info.extra == {"author": "qa@example.com", "ui_hint": "avoid-in-prod"}


def test_list_artifacts_with_meta_matches_list_artifacts():
    """Every artifact on disk shows up in the metadata-enriched listing."""
    for module_name in STAGE_MODULES.values():
        names = list_artifacts(module_name)
        infos = list_artifacts_with_meta(module_name)
        assert [i.name for i in infos] == names, f"mismatch for {module_name}"
        for info in infos:
            assert info.stage == module_name
            assert isinstance(info.is_default, bool)
            assert info.is_default == (info.name == DEFAULT_ARTIFACT)


def test_artifact_info_to_dict_is_json_ready():
    """Every field in ArtifactInfo.to_dict is a JSON-safe type."""
    info = ArtifactInfo(
        stage="s06_api",
        name="default",
        description="mock",
        version="1.2",
        stability="stable",
        requires=("anthropic>=0.40",),
        is_default=True,
        provides_stage=True,
        extra={"cost_tier": "free"},
    )
    payload = info.to_dict()
    assert payload == {
        "stage": "s06_api",
        "name": "default",
        "description": "mock",
        "version": "1.2",
        "stability": "stable",
        "requires": ["anthropic>=0.40"],
        "is_default": True,
        "provides_stage": True,
        "extra": {"cost_tier": "free"},
    }


# ── Stage provenance via create_stage ────────────────────────────


def test_create_stage_stamps_artifact_and_module():
    """Stage instances remember which artifact+module produced them."""
    stage = create_stage("s01_input")
    assert stage.artifact_name == "default"
    assert stage.stage_module == "s01_input"


def test_create_stage_accepts_aliases_and_order():
    """``create_stage`` accepts the same identifier forms as describe_artifact."""
    by_alias = create_stage("input")
    by_order = create_stage("1")
    assert by_alias.stage_module == "s01_input"
    assert by_order.stage_module == "s01_input"


def test_stage_artifact_name_defaults_to_default_when_unset():
    """Directly-instantiated stages report ``default`` instead of blowing up."""
    from geny_executor.stages.s01_input.artifact.default.stage import InputStage

    bare = InputStage()
    assert bare.artifact_name == "default"
    # stage_module falls back to f"s{order:02d}_{name}"
    assert bare.stage_module == f"s{bare.order:02d}_{bare.name}"


# ── StageToolBinding round-trip ──────────────────────────────────


def test_stage_tool_binding_roundtrip_none_values():
    """``allowed``/``blocked`` None must round-trip as None, not empty set."""
    original = StageToolBinding(stage_order=6)
    payload = original.to_dict()
    assert payload == {
        "stage_order": 6,
        "allowed": None,
        "blocked": None,
        "extra_context": {},
    }
    restored = StageToolBinding.from_dict(payload)
    assert restored == original


def test_stage_tool_binding_roundtrip_with_values():
    """Populated allow/block lists survive serialization and are sorted."""
    original = StageToolBinding(
        stage_order=10,
        allowed={"b_tool", "a_tool"},
        blocked={"c_tool"},
        extra_context={"mode": "strict"},
    )
    payload = original.to_dict()
    # sorted for deterministic output
    assert payload["allowed"] == ["a_tool", "b_tool"]
    assert payload["blocked"] == ["c_tool"]
    restored = StageToolBinding.from_dict(payload)
    assert restored == original


# ── ModelConfig round-trip ───────────────────────────────────────


def test_model_config_roundtrip_defaults():
    """Default ModelConfig survives to_dict/from_dict unchanged."""
    original = ModelConfig()
    restored = ModelConfig.from_dict(original.to_dict())
    assert restored == original


def test_model_config_roundtrip_full():
    """All ModelConfig fields survive to_dict/from_dict."""
    original = ModelConfig(
        model="claude-opus-4-7",
        max_tokens=4096,
        temperature=0.3,
        top_p=0.9,
        top_k=40,
        stop_sequences=["STOP"],
        thinking_enabled=True,
        thinking_budget_tokens=5000,
        thinking_type="adaptive",
        thinking_display="summarized",
    )
    restored = ModelConfig.from_dict(original.to_dict())
    assert restored == original


def test_model_config_from_dict_ignores_unknown_keys():
    """Forward-compat: unknown keys from future versions must not crash."""
    payload = ModelConfig(model="claude-opus-4-7").to_dict()
    payload["future_new_knob"] = 42
    restored = ModelConfig.from_dict(payload)
    assert restored.model == "claude-opus-4-7"
