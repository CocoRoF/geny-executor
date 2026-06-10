"""2.2.0 Wave 2 — ``build_manifest`` preset→manifest factory.

The factory absorbs Geny's host-side manifest builder
(``Geny/backend/service/executor/default_manifest.py``, ~728 lines of
hand-mirrored layout — audit §1-3 host-compensation table). Contract
under test:

  - presets ``worker_adaptive`` / ``vtuber`` / ``default`` (alias);
  - loud validation of preset / provider / mcp_servers inputs;
  - provider lands at ``stages[6].config['provider']`` (single home);
  - model lands in the top-level ``model`` block (single home);
  - output round-trips ``to_dict``/``from_dict`` and builds under
    ``Pipeline.from_manifest(strict=True)``;
  - layout is byte-compatible with what Geny's builder produced when
    the library absorbed it (vendored snapshot fixture:
    ``tests/_fixtures/geny_manifest_layout.json``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geny_executor import (
    EnvironmentManifest,
    Pipeline,
    build_manifest,
    known_manifest_presets,
    validate_manifest,
)
from geny_executor.llm_client.credentials import CredentialBundle

_SNAPSHOT_PATH = Path(__file__).resolve().parents[1] / "_fixtures" / "geny_manifest_layout.json"


def _stage(manifest: EnvironmentManifest, order: int) -> dict:
    for entry in manifest.stages:
        if entry["order"] == order:
            return entry
    raise AssertionError(f"no stage entry with order {order}")


# ── Input validation ─────────────────────────────────────────


class TestInputValidation:
    def test_unknown_preset_raises_listing_known(self):
        with pytest.raises(ValueError, match="unknown preset"):
            build_manifest("workr_adaptive", provider="anthropic")

    def test_known_presets_accessor(self):
        assert known_manifest_presets() == ["default", "vtuber", "worker_adaptive"]

    def test_unknown_provider_raises_listing_registered(self):
        with pytest.raises(ValueError, match="unknown provider 'antrhopic'"):
            build_manifest("worker_adaptive", provider="antrhopic")

    def test_empty_provider_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            build_manifest("worker_adaptive", provider="")

    def test_nameless_mcp_server_raises(self):
        with pytest.raises(ValueError, match="mcp_servers"):
            build_manifest(
                "worker_adaptive",
                provider="anthropic",
                mcp_servers=[{"command": "npx", "args": ["server"]}],
            )

    def test_non_dict_mcp_server_raises(self):
        with pytest.raises(ValueError, match="mcp_servers"):
            build_manifest("worker_adaptive", provider="anthropic", mcp_servers=["fs"])


# ── Declarations land where promised ─────────────────────────


class TestDeclarationHomes:
    def test_provider_lands_in_stage6_config(self):
        m = build_manifest("worker_adaptive", provider="claude_code_cli")
        s6 = _stage(m, 6)
        assert s6["config"]["provider"] == "claude_code_cli"
        # Never in the legacy strategies slot.
        assert "provider" not in s6["strategies"]

    def test_model_lands_in_top_level_block_only(self):
        m = build_manifest("vtuber", provider="anthropic", model="claude-sonnet-4-5")
        assert m.model == {"model": "claude-sonnet-4-5"}
        assert "model" not in _stage(m, 6)["config"]

    def test_no_model_means_empty_block(self):
        m = build_manifest("vtuber", provider="anthropic")
        assert m.model == {}

    def test_tools_land_in_tools_snapshot(self):
        m = build_manifest(
            "worker_adaptive",
            provider="anthropic",
            built_in_tools=["*"],
            external_tools=["alpha", {"name": "beta", "required": True}],
            mcp_servers=[{"name": "fs", "command": "npx", "args": ["server-fs"]}],
        )
        assert m.tools.built_in == ["*"]
        assert m.tools.external == ["alpha", {"name": "beta", "required": True}]
        assert m.tools.mcp_servers == [{"name": "fs", "command": "npx", "args": ["server-fs"]}]

    def test_default_alias_collapses_to_worker_adaptive(self):
        alias = build_manifest("default", provider="anthropic")
        canonical = build_manifest("worker_adaptive", provider="anthropic")
        assert alias.metadata.base_preset == "worker_adaptive"
        assert alias.stages == canonical.stages

    def test_metadata_defaults_and_overrides(self):
        m = build_manifest("vtuber", provider="anthropic")
        assert m.metadata.name == "preset:vtuber"
        assert m.metadata.base_preset == "vtuber"
        assert m.metadata.id.startswith("env_")
        named = build_manifest(
            "vtuber", provider="anthropic", name="Mio", description="persona env"
        )
        assert named.metadata.name == "Mio"
        assert named.metadata.description == "persona env"


# ── Round-trip + strict build ─────────────────────────────────


class TestRoundTrip:
    @pytest.mark.parametrize("preset", ["worker_adaptive", "vtuber", "default"])
    def test_factory_to_dict_from_dict_builds_strict(self, preset):
        m = build_manifest(preset, provider="anthropic", model="claude-sonnet-4-5")
        reloaded = EnvironmentManifest.from_dict(m.to_dict())
        assert reloaded.to_dict() == m.to_dict()
        pipeline = Pipeline.from_manifest(
            reloaded, credentials=CredentialBundle(), strict=True
        )
        # The full 21-slot layout, with each preset's active set built.
        assert pipeline.get_stage(6) is not None
        assert pipeline.get_stage(1) is not None
        assert pipeline.get_stage(21) is not None

    @pytest.mark.parametrize("preset", ["worker_adaptive", "vtuber"])
    def test_factory_output_validates_clean(self, preset):
        issues = validate_manifest(build_manifest(preset, provider="anthropic"))
        errors = [i for i in issues if i.severity == "error"]
        assert errors == []
        # The only acceptable warnings are the documented chain-population
        # ones (restore can only reorder, never populate — module docstring).
        assert {i.code for i in issues} <= {"chain.order_unappliable"}

    def test_vtuber_thinking_stage_declared_inactive(self):
        """Stage 8 is declared with active=False rather than omitted, so
        env editors render the slot like every other inactive stage."""
        m = build_manifest("vtuber", provider="anthropic")
        s8 = _stage(m, 8)
        assert s8["active"] is False
        assert s8["strategies"]["processor"] == "extract_and_store"

    def test_worker_strategy_configs_actually_land(self):
        """The audit §2.1 regression at factory level: the s14/s16
        strategy_configs must configure the live strategies after a
        strict build (Wave 1 made configure real; the factory must
        declare shapes that exercise it)."""
        m = build_manifest("worker_adaptive", provider="anthropic")
        p = Pipeline.from_manifest(m, credentials=CredentialBundle(), strict=True)
        chain = p.get_stage(14).get_strategy_slots()["strategy"].strategy
        assert [type(ev).__name__ for ev in chain.evaluators] == [
            "BinaryClassifyEvaluation",
            "SignalBasedEvaluation",
        ]
        controller = p.get_stage(16).get_strategy_slots()["controller"].strategy
        assert [type(d).__name__ for d in controller.dimensions] == ["IterationBudget"]


# ── Layout compatibility with the absorbed Geny builder ──────


class TestGenyLayoutCompatibility:
    """Compare against the vendored snapshot of Geny's builder output.

    The snapshot was generated from
    ``Geny/backend/service/executor/default_manifest.py`` (the host
    layer this factory deletes) at absorption time — see the fixture's
    ``_rationale`` field. Regenerating it is a deliberate layout-change
    act, not a test-fixing one.
    """

    @pytest.fixture(scope="class")
    def snapshot(self):
        return json.loads(_SNAPSHOT_PATH.read_text())

    @pytest.mark.parametrize("preset", ["worker_adaptive", "vtuber"])
    def test_stage_layout_identical_to_geny_builder(self, snapshot, preset):
        ours = build_manifest(preset, provider=snapshot["provider"]).to_dict()["stages"]
        theirs = snapshot["presets"][preset]
        assert ours == theirs

    @pytest.mark.parametrize("preset", ["worker_adaptive", "vtuber"])
    def test_full_21_slot_layout(self, snapshot, preset):
        ours = build_manifest(preset, provider="anthropic").to_dict()["stages"]
        assert [e["order"] for e in ours] == list(range(1, 22))
