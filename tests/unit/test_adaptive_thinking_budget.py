"""Unit tests for the Stage 8 adaptive thinking budget (S7.10)."""

from __future__ import annotations

import pytest

from geny_executor.core.state import PipelineState
from geny_executor.stages.s08_think import (
    AdaptiveThinkingBudget,
    StaticThinkingBudget,
    ThinkingBudgetPlanner,
    ThinkStage,
    apply_thinking_budget,
    make_planner,
)


# ── helpers ───────────────────────────────────────────────────────────────


def _state(
    *,
    user_text: str = "hi",
    system: str = "",
    tools: list | None = None,
    needs_reflection: bool = False,
) -> PipelineState:
    state = PipelineState()
    state.system = system
    state.messages = [{"role": "user", "content": user_text}]
    state.tools = list(tools or [])
    if needs_reflection:
        state.metadata["needs_reflection"] = True
    return state


# ── StaticThinkingBudget ─────────────────────────────────────────────────


class TestStaticThinkingBudget:
    def test_default_budget(self):
        p = StaticThinkingBudget()
        assert p.plan(_state()) == 10_000
        assert p.budget_tokens == 10_000

    def test_custom_budget(self):
        p = StaticThinkingBudget(budget_tokens=5_000)
        assert p.plan(_state()) == 5_000

    def test_negative_budget_rejected(self):
        with pytest.raises(ValueError):
            StaticThinkingBudget(budget_tokens=-1)

    def test_is_planner(self):
        assert isinstance(StaticThinkingBudget(), ThinkingBudgetPlanner)

    def test_state_independent(self):
        p = StaticThinkingBudget(budget_tokens=7_777)
        a = p.plan(_state(user_text="hi"))
        b = p.plan(_state(user_text="x" * 10_000, tools=[{"name": "t"}]))
        assert a == b == 7_777


# ── AdaptiveThinkingBudget ───────────────────────────────────────────────


class TestAdaptiveTierLogic:
    def test_base_only_for_small_state(self):
        p = AdaptiveThinkingBudget(
            base_budget=4_000,
            min_budget=2_000,
            max_budget=24_000,
            tools_bonus=4_000,
            reflection_bonus=4_000,
            size_step_chars=4_000,
            size_step_bonus=2_000,
        )
        assert p.plan(_state(user_text="hi")) == 4_000

    def test_tools_bonus_applied(self):
        p = AdaptiveThinkingBudget(
            base_budget=4_000,
            min_budget=2_000,
            max_budget=24_000,
            tools_bonus=4_000,
            reflection_bonus=0,
            size_step_chars=10_000,
            size_step_bonus=0,
        )
        out = p.plan(_state(user_text="hi", tools=[{"name": "t"}]))
        assert out == 8_000

    def test_reflection_bonus_applied(self):
        p = AdaptiveThinkingBudget(
            base_budget=4_000,
            min_budget=2_000,
            max_budget=24_000,
            tools_bonus=0,
            reflection_bonus=4_000,
            size_step_chars=10_000,
            size_step_bonus=0,
        )
        out = p.plan(_state(user_text="hi", needs_reflection=True))
        assert out == 8_000

    def test_size_step_bonus_applied(self):
        p = AdaptiveThinkingBudget(
            base_budget=4_000,
            min_budget=2_000,
            max_budget=100_000,
            tools_bonus=0,
            reflection_bonus=0,
            size_step_chars=1_000,
            size_step_bonus=1_000,
        )
        # 5000-char prompt → 5 size steps → +5000
        out = p.plan(_state(user_text="x" * 5_000))
        assert out == 9_000

    def test_clamped_to_max(self):
        p = AdaptiveThinkingBudget(
            base_budget=10_000,
            min_budget=2_000,
            max_budget=12_000,
            tools_bonus=10_000,
            reflection_bonus=10_000,
            size_step_chars=1_000,
            size_step_bonus=10_000,
        )
        # all bonuses fire and exceed max — clamped to 12_000
        out = p.plan(
            _state(
                user_text="x" * 5_000,
                tools=[{"name": "t"}],
                needs_reflection=True,
            )
        )
        assert out == 12_000

    def test_clamped_to_min(self):
        p = AdaptiveThinkingBudget(
            base_budget=500,
            min_budget=2_000,
            max_budget=24_000,
            tools_bonus=0,
            reflection_bonus=0,
            size_step_chars=10_000,
            size_step_bonus=0,
        )
        assert p.plan(_state(user_text="hi")) == 2_000

    def test_bounds_property(self):
        p = AdaptiveThinkingBudget(min_budget=1_000, max_budget=20_000)
        assert p.bounds == (1_000, 20_000)


class TestAdaptiveValidation:
    def test_negative_min_rejected(self):
        with pytest.raises(ValueError):
            AdaptiveThinkingBudget(min_budget=-1)

    def test_negative_max_rejected(self):
        with pytest.raises(ValueError):
            AdaptiveThinkingBudget(max_budget=-1)

    def test_inverted_bounds_rejected(self):
        with pytest.raises(ValueError):
            AdaptiveThinkingBudget(min_budget=10_000, max_budget=5_000)

    def test_zero_size_step_rejected(self):
        with pytest.raises(ValueError):
            AdaptiveThinkingBudget(size_step_chars=0)


class TestAdaptiveSystemAndBlocks:
    def test_counts_system_prompt(self):
        p = AdaptiveThinkingBudget(
            base_budget=2_000,
            min_budget=2_000,
            max_budget=100_000,
            tools_bonus=0,
            reflection_bonus=0,
            size_step_chars=1_000,
            size_step_bonus=1_000,
        )
        out = p.plan(_state(system="z" * 3_000, user_text="hi"))
        # 3000 chars + ~2 chars user → 3 steps → +3000
        assert out == 5_000

    def test_counts_block_text(self):
        p = AdaptiveThinkingBudget(
            base_budget=2_000,
            min_budget=2_000,
            max_budget=100_000,
            tools_bonus=0,
            reflection_bonus=0,
            size_step_chars=500,
            size_step_bonus=1_000,
        )
        state = _state()
        state.messages = [
            {"role": "user", "content": [{"type": "text", "text": "y" * 1_500}]}
        ]
        out = p.plan(state)
        # 1500 // 500 = 3 steps → +3000
        assert out == 5_000


# ── apply_thinking_budget helper ─────────────────────────────────────────


class TestApplyThinkingBudget:
    def test_writes_to_state(self):
        state = _state()
        state.thinking_budget_tokens = 1_000
        new = apply_thinking_budget(state, StaticThinkingBudget(5_555))
        assert new == 5_555
        assert state.thinking_budget_tokens == 5_555

    def test_emits_event_by_default(self):
        state = _state()
        state.thinking_budget_tokens = 1_000
        apply_thinking_budget(state, StaticThinkingBudget(5_555))
        evts = [e for e in state.events if e["type"] == "think.budget_applied"]
        assert len(evts) == 1
        data = evts[0]["data"]
        assert data["from"] == 1_000
        assert data["to"] == 5_555
        assert data["planner"] == "static"

    def test_emit_event_disabled(self):
        state = _state()
        apply_thinking_budget(
            state, StaticThinkingBudget(5_555), emit_event=False
        )
        assert [e for e in state.events if e["type"] == "think.budget_applied"] == []

    def test_returns_int(self):
        state = _state()
        result = apply_thinking_budget(state, StaticThinkingBudget(7_000))
        assert isinstance(result, int)


# ── make_planner factory ─────────────────────────────────────────────────


class TestMakePlanner:
    def test_static_default(self):
        p = make_planner()
        assert isinstance(p, StaticThinkingBudget)
        assert p.budget_tokens == 10_000

    def test_static_with_base(self):
        p = make_planner(base_budget=6_000)
        assert isinstance(p, StaticThinkingBudget)
        assert p.budget_tokens == 6_000

    def test_adaptive_with_bounds(self):
        p = make_planner(adaptive_budget=True, min_budget=1_000, max_budget=15_000)
        assert isinstance(p, AdaptiveThinkingBudget)
        assert p.bounds == (1_000, 15_000)

    def test_adaptive_with_base(self):
        p = make_planner(adaptive_budget=True, base_budget=3_000)
        assert isinstance(p, AdaptiveThinkingBudget)


# ── ThinkStage integration ───────────────────────────────────────────────


class TestThinkStageBudgetSlot:
    def test_default_planner_is_static(self):
        stage = ThinkStage()
        slot = stage.get_strategy_slots()["budget_planner"]
        assert isinstance(slot.strategy, StaticThinkingBudget)

    def test_registry_lists_both_planners(self):
        stage = ThinkStage()
        registry = stage.get_strategy_slots()["budget_planner"].registry
        assert "static" in registry
        assert "adaptive" in registry

    def test_custom_planner_via_constructor(self):
        custom = AdaptiveThinkingBudget()
        stage = ThinkStage(budget_planner=custom)
        assert stage.get_strategy_slots()["budget_planner"].strategy is custom

    def test_apply_planned_budget_writes_state(self):
        stage = ThinkStage(budget_planner=StaticThinkingBudget(7_777))
        state = _state()
        state.thinking_budget_tokens = 1_000
        new = stage.apply_planned_budget(state)
        assert new == 7_777
        assert state.thinking_budget_tokens == 7_777

    def test_apply_planned_budget_emits_event(self):
        stage = ThinkStage(budget_planner=StaticThinkingBudget(3_333))
        state = _state()
        stage.apply_planned_budget(state)
        evts = [e for e in state.events if e["type"] == "think.budget_applied"]
        assert len(evts) == 1
        assert evts[0]["data"]["to"] == 3_333

    def test_execute_does_not_auto_apply(self):
        """Sanity: execute() must not invoke the planner — Stage 8 runs after the API call."""
        stage = ThinkStage(budget_planner=StaticThinkingBudget(99_999))
        state = _state()
        state.thinking_budget_tokens = 1_000
        # No last_api_response → execute returns input quickly without
        # touching the budget.
        import asyncio

        result = asyncio.run(stage.execute(input="x", state=state))
        assert result == "x"
        assert state.thinking_budget_tokens == 1_000  # untouched
