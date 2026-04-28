"""Stage 8 thinking-budget planners (S7.10).

Two flavours:

* :class:`StaticThinkingBudget` — returns a single fixed value. Used as
  the default slot strategy so the new ``budget_planner`` slot is a
  zero-behaviour-change addition: existing pipelines that never call
  :meth:`ThinkStage.apply_planned_budget` see no difference, and even
  if they do, a static planner reproduces today's behaviour.
* :class:`AdaptiveThinkingBudget` — sizes the per-turn budget from
  cheap heuristics (message size in characters, tools-on-state, the
  ``needs_reflection`` flag from Stage 15). Bounded by ``min_budget``
  and ``max_budget`` so the strategy can never blow past the model's
  ``max_tokens``-tied budget cap.

Apply via :func:`apply_thinking_budget` or via
:meth:`ThinkStage.apply_planned_budget` — both write the planned value
back onto ``state.thinking_budget_tokens`` so Stage 6's
``resolve_model_config`` picks it up on the next API call.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.core.state import PipelineState
from geny_executor.stages.s08_think.interface import ThinkingBudgetPlanner


def _estimate_chars(state: PipelineState) -> int:
    """Same shape as the Stage 6 router heuristic — counts system + messages."""
    total = 0
    if state.system:
        total += len(state.system) if isinstance(state.system, str) else 0
    for msg in state.messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += len(content)
            continue
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        total += len(text)
                        continue
                    for key in ("input", "content"):
                        if key in block:
                            total += len(str(block[key]))
                            break
    return total


class StaticThinkingBudget(ThinkingBudgetPlanner):
    """Always returns a single fixed budget — current behaviour."""

    def __init__(self, budget_tokens: int = 10_000) -> None:
        if budget_tokens < 0:
            raise ValueError("budget_tokens must be non-negative")
        self._budget = int(budget_tokens)

    @property
    def name(self) -> str:
        return "static"

    @property
    def budget_tokens(self) -> int:
        return self._budget

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name="static",
            fields=[
                ConfigField(
                    name="budget_tokens",
                    type="integer",
                    label="Budget tokens",
                    description="Fixed thinking budget per turn.",
                    default=10_000,
                    min_value=0,
                ),
            ],
        )

    def configure(self, config: Dict[str, Any]) -> None:
        n = config.get("budget_tokens")
        if isinstance(n, int) and n >= 0:
            self._budget = n

    def get_config(self) -> Dict[str, Any]:
        return {"budget_tokens": self._budget}

    def plan(self, state: PipelineState) -> int:
        return self._budget


class AdaptiveThinkingBudget(ThinkingBudgetPlanner):
    """Heuristic budget sizing.

    Decision (clamped to ``[min_budget, max_budget]``):

    * Start from ``base_budget``.
    * Add ``tools_bonus`` when ``state.tools`` is non-empty (tool-use
      turns benefit from more reasoning).
    * Add ``reflection_bonus`` when
      ``state.metadata["needs_reflection"]`` is truthy (the previous
      turn flagged a reflection request).
    * Scale by message size — for every full ``size_step_chars`` of
      prompt characters, add ``size_step_bonus``.
    * Clamp the final value into the configured bounds.
    """

    def __init__(
        self,
        *,
        base_budget: int = 4_000,
        min_budget: int = 2_000,
        max_budget: int = 24_000,
        tools_bonus: int = 4_000,
        reflection_bonus: int = 4_000,
        size_step_chars: int = 4_000,
        size_step_bonus: int = 2_000,
    ) -> None:
        if min_budget < 0 or max_budget < 0 or base_budget < 0:
            raise ValueError("budget values must be non-negative")
        if max_budget < min_budget:
            raise ValueError("max_budget must be >= min_budget")
        if size_step_chars <= 0:
            raise ValueError("size_step_chars must be positive")
        self._base = int(base_budget)
        self._min = int(min_budget)
        self._max = int(max_budget)
        self._tools_bonus = int(tools_bonus)
        self._reflection_bonus = int(reflection_bonus)
        self._size_step_chars = int(size_step_chars)
        self._size_step_bonus = int(size_step_bonus)

    @property
    def name(self) -> str:
        return "adaptive"

    @property
    def bounds(self) -> tuple[int, int]:
        return (self._min, self._max)

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name="adaptive",
            fields=[
                ConfigField(
                    name="base_budget",
                    type="integer",
                    label="Base budget",
                    description="Starting budget before bonuses.",
                    default=4_000,
                    min_value=0,
                ),
                ConfigField(
                    name="min_budget",
                    type="integer",
                    label="Min budget",
                    description="Lower clamp on the final per-turn budget.",
                    default=2_000,
                    min_value=0,
                ),
                ConfigField(
                    name="max_budget",
                    type="integer",
                    label="Max budget",
                    description="Upper clamp on the final per-turn budget.",
                    default=24_000,
                    min_value=0,
                ),
                ConfigField(
                    name="tools_bonus",
                    type="integer",
                    label="Tools bonus",
                    description="Extra budget when state.tools is non-empty.",
                    default=4_000,
                    min_value=0,
                ),
                ConfigField(
                    name="reflection_bonus",
                    type="integer",
                    label="Reflection bonus",
                    description="Extra budget when state.metadata.needs_reflection is set.",
                    default=4_000,
                    min_value=0,
                ),
            ],
        )

    def configure(self, config: Dict[str, Any]) -> None:
        for key in ("base_budget", "min_budget", "max_budget", "tools_bonus", "reflection_bonus"):
            v = config.get(key)
            if isinstance(v, int) and v >= 0:
                setattr(
                    self,
                    {
                        "base_budget": "_base",
                        "min_budget": "_min",
                        "max_budget": "_max",
                        "tools_bonus": "_tools_bonus",
                        "reflection_bonus": "_reflection_bonus",
                    }[key],
                    v,
                )
        if self._max < self._min:
            self._max = self._min

    def get_config(self) -> Dict[str, Any]:
        return {
            "base_budget": self._base,
            "min_budget": self._min,
            "max_budget": self._max,
            "tools_bonus": self._tools_bonus,
            "reflection_bonus": self._reflection_bonus,
        }

    def _raw(self, state: PipelineState) -> int:
        budget = self._base
        if state.tools:
            budget += self._tools_bonus
        if state.metadata.get("needs_reflection"):
            budget += self._reflection_bonus
        chars = _estimate_chars(state)
        if chars > 0 and self._size_step_chars > 0:
            steps = chars // self._size_step_chars
            budget += steps * self._size_step_bonus
        return budget

    def plan(self, state: PipelineState) -> int:
        raw = self._raw(state)
        return max(self._min, min(self._max, raw))


def apply_thinking_budget(
    state: PipelineState,
    planner: ThinkingBudgetPlanner,
    *,
    emit_event: bool = True,
) -> int:
    """Run ``planner`` and write the result onto ``state.thinking_budget_tokens``.

    Returns the new budget. Emits ``think.budget_applied`` with the
    before/after values when ``emit_event`` is True (the default). The
    state's other fields are not touched.
    """
    new_budget = int(planner.plan(state))
    old_budget = int(state.thinking_budget_tokens)
    state.thinking_budget_tokens = new_budget
    if emit_event:
        data: Dict[str, Any] = {
            "planner": getattr(planner, "name", ""),
            "from": old_budget,
            "to": new_budget,
        }
        state.add_event("think.budget_applied", data)
    return new_budget


def make_planner(
    *,
    adaptive_budget: bool = False,
    min_budget: Optional[int] = None,
    max_budget: Optional[int] = None,
    base_budget: Optional[int] = None,
) -> ThinkingBudgetPlanner:
    """Construct a planner from ``ConfigSchema``-style flags.

    ``adaptive_budget=False`` returns a :class:`StaticThinkingBudget`
    using ``base_budget`` (or the default). ``adaptive_budget=True``
    returns an :class:`AdaptiveThinkingBudget` with the supplied
    bounds; ``None`` falls back to that planner's defaults.
    """
    if adaptive_budget:
        kwargs: Dict[str, Any] = {}
        if base_budget is not None:
            kwargs["base_budget"] = base_budget
        if min_budget is not None:
            kwargs["min_budget"] = min_budget
        if max_budget is not None:
            kwargs["max_budget"] = max_budget
        return AdaptiveThinkingBudget(**kwargs)
    if base_budget is not None:
        return StaticThinkingBudget(budget_tokens=base_budget)
    return StaticThinkingBudget()


__all__ = [
    "StaticThinkingBudget",
    "AdaptiveThinkingBudget",
    "apply_thinking_budget",
    "make_planner",
]
