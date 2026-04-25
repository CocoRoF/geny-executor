"""Unit tests for the Stage 6 adaptive model router (S7.8)."""

from __future__ import annotations

import pytest

from geny_executor.core.config import ModelConfig, PipelineConfig
from geny_executor.core.state import PipelineState
from geny_executor.stages.s06_api import (
    APIStage,
    AdaptiveModelRouter,
    MockProvider,
    ModelRouter,
    PassthroughRouter,
)
from geny_executor.stages.s06_api.artifact.default.router import (
    AdaptiveModelRouter as DefaultAdaptive,
)


# ── helpers ───────────────────────────────────────────────────────────────


def _state(
    *,
    system: str = "",
    user_text: str = "hi",
    tools: list | None = None,
    thinking: bool = False,
    model: str = "claude-sonnet-4-6",
) -> PipelineState:
    cfg = PipelineConfig(
        model=ModelConfig(
            model=model,
            thinking_enabled=thinking,
            thinking_budget_tokens=2048,
        )
    )
    state = PipelineState()
    cfg.apply_to_state(state)
    state.system = system
    state.messages = [{"role": "user", "content": user_text}]
    state.tools = list(tools or [])
    return state


def _cfg(**overrides) -> ModelConfig:
    base = dict(model="claude-sonnet-4-6")
    base.update(overrides)
    return ModelConfig(**base)


# ── PassthroughRouter ────────────────────────────────────────────────────


class TestPassthroughRouter:
    def test_returns_none_for_any_state(self):
        r = PassthroughRouter()
        assert r.route(_cfg(), _state()) is None

    def test_name_is_passthrough(self):
        assert PassthroughRouter().name == "passthrough"

    def test_is_modelrouter_subclass(self):
        assert isinstance(PassthroughRouter(), ModelRouter)


# ── AdaptiveModelRouter — tier selection ─────────────────────────────────


class TestAdaptiveTierSelection:
    def test_default_class_alias_matches_module_export(self):
        assert AdaptiveModelRouter is DefaultAdaptive

    def test_thinking_promotes_heavy(self):
        r = AdaptiveModelRouter(
            light_model="L", balanced_model="B", heavy_model="H"
        )
        result = r.route(_cfg(thinking_enabled=True), _state(thinking=True))
        assert result is not None
        assert result.model == "H"

    def test_short_query_picks_light(self):
        r = AdaptiveModelRouter(
            light_model="L", balanced_model="B", heavy_model="H"
        )
        result = r.route(_cfg(), _state(user_text="hi"))
        assert result is not None and result.model == "L"

    def test_huge_query_picks_heavy(self):
        r = AdaptiveModelRouter(
            light_model="L",
            balanced_model="B",
            heavy_model="H",
            light_threshold_chars=10,
            heavy_threshold_chars=100,
        )
        big = "x" * 5000
        result = r.route(_cfg(), _state(user_text=big))
        assert result is not None and result.model == "H"

    def test_tools_promote_balanced_when_tools_set(self):
        r = AdaptiveModelRouter(
            light_model="L", balanced_model="B", heavy_model="H"
        )
        # short query but with tools → balanced, not light
        result = r.route(
            _cfg(),
            _state(user_text="hi", tools=[{"name": "t", "description": "x"}]),
        )
        assert result is not None and result.model == "B"

    def test_tools_promote_disabled(self):
        r = AdaptiveModelRouter(
            light_model="L",
            balanced_model="B",
            heavy_model="H",
            tools_promote_balanced=False,
        )
        result = r.route(
            _cfg(),
            _state(user_text="hi", tools=[{"name": "t"}]),
        )
        assert result is not None and result.model == "L"

    def test_thinking_promote_disabled(self):
        r = AdaptiveModelRouter(
            light_model="L",
            balanced_model="B",
            heavy_model="H",
            thinking_promotes_heavy=False,
        )
        result = r.route(
            _cfg(thinking_enabled=True), _state(thinking=True, user_text="hi")
        )
        # short query, no tools → light
        assert result is not None and result.model == "L"

    def test_medium_query_picks_balanced(self):
        r = AdaptiveModelRouter(
            light_model="L",
            balanced_model="B",
            heavy_model="H",
            light_threshold_chars=10,
            heavy_threshold_chars=10_000,
        )
        result = r.route(_cfg(), _state(user_text="x" * 500))
        assert result is not None and result.model == "B"

    def test_returns_none_when_target_equals_current(self):
        r = AdaptiveModelRouter(
            light_model="claude-sonnet-4-6",
            balanced_model="claude-sonnet-4-6",
            heavy_model="claude-sonnet-4-6",
        )
        assert r.route(_cfg(model="claude-sonnet-4-6"), _state()) is None

    def test_swap_preserves_other_config_fields(self):
        r = AdaptiveModelRouter(light_model="L", balanced_model="B", heavy_model="H")
        cfg = _cfg(
            max_tokens=4096,
            temperature=0.7,
            thinking_enabled=False,
        )
        result = r.route(cfg, _state(user_text="hi"))
        assert result is not None
        assert result.model == "L"
        # only model should change
        assert result.max_tokens == 4096
        assert result.temperature == 0.7
        assert result.thinking_enabled is False


# ── AdaptiveModelRouter — character estimation ──────────────────────────


class TestCharEstimation:
    def test_counts_system_prompt(self):
        # tiny user msg, big system → still goes heavy
        r = AdaptiveModelRouter(
            light_model="L",
            balanced_model="B",
            heavy_model="H",
            light_threshold_chars=10,
            heavy_threshold_chars=200,
        )
        result = r.route(_cfg(), _state(system="z" * 1000, user_text="hi"))
        assert result is not None and result.model == "H"

    def test_counts_block_text(self):
        r = AdaptiveModelRouter(
            light_model="L",
            balanced_model="B",
            heavy_model="H",
            light_threshold_chars=10,
            heavy_threshold_chars=300,
        )
        state = _state(user_text="hi")
        state.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "y" * 500},
                ],
            }
        ]
        result = r.route(_cfg(), state)
        assert result is not None and result.model == "H"

    def test_counts_tool_use_input(self):
        r = AdaptiveModelRouter(
            light_model="L",
            balanced_model="B",
            heavy_model="H",
            light_threshold_chars=10,
            heavy_threshold_chars=200,
        )
        state = _state(user_text="hi")
        state.messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "writer",
                        "input": {"data": "p" * 500},
                    }
                ],
            }
        ]
        result = r.route(_cfg(), state)
        assert result is not None and result.model == "H"

    def test_handles_empty_state(self):
        r = AdaptiveModelRouter(
            light_model="L", balanced_model="B", heavy_model="H"
        )
        state = PipelineState()
        # no system, no messages → 0 chars → light tier
        result = r.route(_cfg(), state)
        assert result is not None and result.model == "L"


# ── AdaptiveModelRouter — validation ────────────────────────────────────


class TestAdaptiveValidation:
    def test_negative_thresholds_rejected(self):
        with pytest.raises(ValueError):
            AdaptiveModelRouter(light_threshold_chars=-1)
        with pytest.raises(ValueError):
            AdaptiveModelRouter(heavy_threshold_chars=-1)

    def test_inverted_thresholds_rejected(self):
        with pytest.raises(ValueError):
            AdaptiveModelRouter(
                light_threshold_chars=100, heavy_threshold_chars=50
            )

    def test_equal_thresholds_allowed(self):
        AdaptiveModelRouter(light_threshold_chars=100, heavy_threshold_chars=100)


# ── APIStage integration ─────────────────────────────────────────────────


class TestAPIStageRouterSlot:
    def test_default_router_is_passthrough(self):
        stage = APIStage(provider=MockProvider())
        slot = stage.get_strategy_slots()["router"]
        assert isinstance(slot.strategy, PassthroughRouter)

    def test_router_slot_registry_exposes_both_routers(self):
        stage = APIStage(provider=MockProvider())
        registry = stage.get_strategy_slots()["router"].registry
        assert "passthrough" in registry
        assert "adaptive" in registry

    def test_custom_router_via_constructor(self):
        custom = AdaptiveModelRouter(
            light_model="L", balanced_model="B", heavy_model="H"
        )
        stage = APIStage(provider=MockProvider(), router=custom)
        assert stage.get_strategy_slots()["router"].strategy is custom

    def test_route_model_passes_through_by_default(self):
        stage = APIStage(provider=MockProvider())
        state = _state(model="claude-sonnet-4-6", user_text="hi")
        cfg = stage._route_model(state)
        assert cfg.model == "claude-sonnet-4-6"
        # no event emitted
        events = [e for e in state.events if e["type"] == "api.model_routed"]
        assert events == []

    def test_route_model_swaps_and_emits_event(self):
        router = AdaptiveModelRouter(
            light_model="LIGHT-X",
            balanced_model="BAL-X",
            heavy_model="HEAVY-X",
        )
        stage = APIStage(provider=MockProvider(), router=router)
        # short query, no tools, no thinking → LIGHT-X
        state = _state(model="claude-sonnet-4-6", user_text="hi")
        cfg = stage._route_model(state)
        assert cfg.model == "LIGHT-X"
        events = [e for e in state.events if e["type"] == "api.model_routed"]
        assert len(events) == 1
        data = events[0]["data"]
        assert data["from"] == "claude-sonnet-4-6"
        assert data["to"] == "LIGHT-X"
        assert data["router"] == "adaptive"

    def test_route_model_no_event_when_target_matches(self):
        router = AdaptiveModelRouter(
            light_model="claude-sonnet-4-6",
            balanced_model="claude-sonnet-4-6",
            heavy_model="claude-sonnet-4-6",
        )
        stage = APIStage(provider=MockProvider(), router=router)
        state = _state(model="claude-sonnet-4-6", user_text="hi")
        cfg = stage._route_model(state)
        assert cfg.model == "claude-sonnet-4-6"
        assert [e for e in state.events if e["type"] == "api.model_routed"] == []

    def test_router_exception_falls_back_to_baseline(self):
        class BoomRouter(ModelRouter):
            @property
            def name(self) -> str:
                return "boom"

            def route(self, cfg, state):
                raise RuntimeError("kaboom")

        stage = APIStage(provider=MockProvider(), router=BoomRouter())
        state = _state(model="claude-sonnet-4-6", user_text="hi")
        cfg = stage._route_model(state)
        assert cfg.model == "claude-sonnet-4-6"
        errs = [e for e in state.events if e["type"] == "api.router.error"]
        assert len(errs) == 1
        assert errs[0]["data"]["router"] == "boom"
        assert "kaboom" in errs[0]["data"]["error"]

    def test_state_not_mutated_by_routing(self):
        """Router override must not bleed into state.model."""
        router = AdaptiveModelRouter(
            light_model="LIGHT-X", balanced_model="B", heavy_model="H"
        )
        stage = APIStage(provider=MockProvider(), router=router)
        state = _state(model="claude-sonnet-4-6", user_text="hi")
        before = state.model
        stage._route_model(state)
        assert state.model == before
