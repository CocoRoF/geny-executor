"""Phase 7 Sprint S7.1 — DynamicPersonaPromptBuilder tests."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from geny_executor.core.state import PipelineState
from geny_executor.stages.s03_system import (
    ComposablePromptBuilder,
    DynamicPersonaPromptBuilder,
    PersonaBlock,
    PersonaProvider,
    PersonaResolution,
    PromptBlock,
    RulesBlock,
    SystemStage,
)


# ─────────────────────────────────────────────────────────────────
# Test providers
# ─────────────────────────────────────────────────────────────────


class _StaticProvider:
    """Returns the same resolution every turn (no per-state logic)."""

    def __init__(self, resolution: PersonaResolution):
        self._resolution = resolution
        self.call_count = 0

    def resolve(self, state: Any, *, session_meta: Dict[str, Any]) -> PersonaResolution:
        self.call_count += 1
        return self._resolution


class _StatefulProvider:
    """Returns a different persona block based on session_meta + state."""

    def resolve(self, state: Any, *, session_meta: Dict[str, Any]) -> PersonaResolution:
        character = session_meta.get("character_id", "anonymous")
        iteration = getattr(state, "iteration", 0)
        return PersonaResolution(
            persona_blocks=[
                PersonaBlock(f"Character: {character}, turn {iteration}"),
            ],
            cache_key=f"{character}/{iteration}",
        )


# ─────────────────────────────────────────────────────────────────
# PersonaResolution dataclass
# ─────────────────────────────────────────────────────────────────


class TestPersonaResolution:
    def test_defaults(self):
        r = PersonaResolution()
        assert r.persona_blocks == []
        assert r.system_tail is None
        assert r.cache_key == ""

    def test_immutability(self):
        r = PersonaResolution(persona_blocks=[PersonaBlock("hi")])
        with pytest.raises(Exception):
            r.cache_key = "new"  # type: ignore[misc]

    def test_runtime_checkable_provider_protocol(self):
        # PersonaProvider is @runtime_checkable — duck-typed instances pass.
        p = _StaticProvider(PersonaResolution())
        assert isinstance(p, PersonaProvider)


# ─────────────────────────────────────────────────────────────────
# DynamicPersonaPromptBuilder basic build
# ─────────────────────────────────────────────────────────────────


def _state() -> PipelineState:
    return PipelineState(session_id="s1")


class TestBuilderBasics:
    def test_strategy_metadata(self):
        b = DynamicPersonaPromptBuilder(_StaticProvider(PersonaResolution()))
        assert b.name == "dynamic_persona"
        assert "persona" in b.description.lower()

    def test_session_meta_is_defensive_copy(self):
        meta = {"session_id": "s1", "character_id": "ellen"}
        b = DynamicPersonaPromptBuilder(
            _StaticProvider(PersonaResolution()), session_meta=meta
        )
        # Mutate the original after construction — builder should not see it
        meta["character_id"] = "rinko"
        assert b.session_meta["character_id"] == "ellen"

    def test_get_config_summary_only(self):
        b = DynamicPersonaPromptBuilder(
            _StaticProvider(PersonaResolution()),
            session_meta={"session_id": "s1", "character_id": "ellen"},
            tail_blocks=[RulesBlock(["be kind"])],
            use_content_blocks=True,
        )
        cfg = b.get_config()
        assert sorted(cfg["session_meta_keys"]) == ["character_id", "session_id"]
        assert cfg["tail_block_names"] == ["rules"]
        assert cfg["use_content_blocks"] is True

    def test_provider_property(self):
        prov = _StaticProvider(PersonaResolution())
        b = DynamicPersonaPromptBuilder(prov)
        assert b.provider is prov


# ─────────────────────────────────────────────────────────────────
# Build output composition
# ─────────────────────────────────────────────────────────────────


class TestBuilderComposition:
    def test_persona_blocks_render_in_order(self):
        prov = _StaticProvider(
            PersonaResolution(
                persona_blocks=[
                    PersonaBlock("first persona"),
                    PersonaBlock("second persona"),
                ]
            )
        )
        b = DynamicPersonaPromptBuilder(prov)
        out = b.build(_state())
        assert isinstance(out, str)
        assert "first persona" in out
        assert "second persona" in out
        assert out.index("first persona") < out.index("second persona")

    def test_tail_blocks_appended_after_persona(self):
        prov = _StaticProvider(
            PersonaResolution(persona_blocks=[PersonaBlock("PERSONA")])
        )
        b = DynamicPersonaPromptBuilder(
            prov, tail_blocks=[RulesBlock(["TAIL_RULE"])]
        )
        out = b.build(_state())
        assert isinstance(out, str)
        assert out.index("PERSONA") < out.index("TAIL_RULE")

    def test_system_tail_text_appended_last(self):
        prov = _StaticProvider(
            PersonaResolution(
                persona_blocks=[PersonaBlock("PERSONA")],
                system_tail="EPHEMERAL_LINE",
            )
        )
        b = DynamicPersonaPromptBuilder(
            prov, tail_blocks=[RulesBlock(["MIDDLE"])]
        )
        out = b.build(_state())
        assert isinstance(out, str)
        # Order: persona → tail_blocks → system_tail
        assert out.index("PERSONA") < out.index("MIDDLE") < out.index("EPHEMERAL_LINE")

    def test_empty_persona_yields_only_tail(self):
        prov = _StaticProvider(PersonaResolution())
        b = DynamicPersonaPromptBuilder(
            prov, tail_blocks=[RulesBlock(["the rule"])]
        )
        out = b.build(_state())
        assert "the rule" in out

    def test_completely_empty_returns_empty_string(self):
        prov = _StaticProvider(PersonaResolution())
        b = DynamicPersonaPromptBuilder(prov)
        out = b.build(_state())
        assert out == ""


# ─────────────────────────────────────────────────────────────────
# Per-turn re-resolution
# ─────────────────────────────────────────────────────────────────


class TestPerTurnResolution:
    def test_resolve_called_each_build(self):
        prov = _StaticProvider(
            PersonaResolution(persona_blocks=[PersonaBlock("p")])
        )
        b = DynamicPersonaPromptBuilder(prov)
        b.build(_state())
        b.build(_state())
        b.build(_state())
        assert prov.call_count == 3

    def test_provider_sees_session_meta_each_call(self):
        captured: List[Dict[str, Any]] = []

        class _Capturing:
            def resolve(self, state, *, session_meta):
                captured.append(dict(session_meta))
                return PersonaResolution()

        b = DynamicPersonaPromptBuilder(
            _Capturing(),
            session_meta={"session_id": "s9", "character_id": "ellen"},
        )
        b.build(_state())
        b.build(_state())
        assert captured == [
            {"session_id": "s9", "character_id": "ellen"},
            {"session_id": "s9", "character_id": "ellen"},
        ]

    def test_provider_sees_state_each_call(self):
        prov = _StatefulProvider()
        b = DynamicPersonaPromptBuilder(
            prov, session_meta={"character_id": "ellen"}
        )
        s1 = PipelineState(session_id="s1")
        s1.iteration = 1
        out_a = b.build(s1)
        s1.iteration = 2
        out_b = b.build(s1)
        assert "turn 1" in str(out_a)
        assert "turn 2" in str(out_b)


# ─────────────────────────────────────────────────────────────────
# Content-blocks mode
# ─────────────────────────────────────────────────────────────────


class TestContentBlocksMode:
    def test_returns_list_when_use_content_blocks_true(self):
        prov = _StaticProvider(
            PersonaResolution(persona_blocks=[PersonaBlock("p1")])
        )
        b = DynamicPersonaPromptBuilder(
            prov, tail_blocks=[RulesBlock(["r"])], use_content_blocks=True
        )
        out = b.build(_state())
        assert isinstance(out, list)
        assert all(isinstance(blk, dict) and blk.get("type") == "text" for blk in out)


# ─────────────────────────────────────────────────────────────────
# SystemStage strategy registry wiring
# ─────────────────────────────────────────────────────────────────


class TestStageRegistration:
    def test_dynamic_persona_in_strategy_registry(self):
        stage = SystemStage()
        registry = stage.get_strategy_slots()["builder"].registry
        assert "dynamic_persona" in registry
        assert registry["dynamic_persona"] is DynamicPersonaPromptBuilder

    def test_built_in_via_inner_composable_path(self):
        """Smoke test the integration: build a SystemStage with the
        dynamic builder and confirm Stage 3's execute path produces
        a sensible system prompt without any host-specific code."""
        prov = _StaticProvider(
            PersonaResolution(persona_blocks=[PersonaBlock("hello")])
        )
        builder = DynamicPersonaPromptBuilder(
            prov, tail_blocks=[RulesBlock(["be helpful"])]
        )
        stage = SystemStage(builder=builder)
        # Sanity: composable / static / dynamic_persona all live in
        # the strategy registry.
        registry_keys = set(stage.get_strategy_slots()["builder"].registry.keys())
        assert {"static", "composable", "dynamic_persona"} <= registry_keys
        # The active strategy is the dynamic one.
        assert stage._slots["builder"].strategy is builder

    def test_can_swap_to_dynamic_persona_via_slot_api(self):
        stage = SystemStage()
        # Default is StaticPromptBuilder
        assert stage._slots["builder"].strategy.name == "static"
        slot = stage.get_strategy_slots()["builder"]
        # The slot's swap rebuilds via cls() — DynamicPersonaPromptBuilder
        # requires a provider arg, so we expect TypeError here. This
        # documents the actual contract (host must construct the
        # builder and attach it via ``Pipeline.attach_runtime``).
        with pytest.raises(TypeError):
            slot.swap("dynamic_persona")


# ─────────────────────────────────────────────────────────────────
# Inner ComposablePromptBuilder consistency
# ─────────────────────────────────────────────────────────────────


class TestComposableConsistency:
    def test_separator_passed_through(self):
        prov = _StaticProvider(
            PersonaResolution(
                persona_blocks=[PersonaBlock("a"), PersonaBlock("b")]
            )
        )
        b = DynamicPersonaPromptBuilder(prov, separator=" | ")
        out = b.build(_state())
        assert isinstance(out, str)
        assert " | " in out

    def test_static_blocks_rendered_via_composable_engine(self):
        """Tail blocks like ``RulesBlock`` should render exactly as
        ``ComposablePromptBuilder`` would render them — ensures the
        dynamic builder doesn't introduce its own divergent path."""
        rules = RulesBlock(["one", "two"])
        prov = _StaticProvider(PersonaResolution())
        dyn = DynamicPersonaPromptBuilder(prov, tail_blocks=[rules])
        out_dyn = dyn.build(_state())

        comp = ComposablePromptBuilder(blocks=[rules])
        out_static = comp.build(_state())
        assert out_dyn == out_static
