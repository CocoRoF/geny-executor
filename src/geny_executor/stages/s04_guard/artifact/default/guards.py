"""Guard implementations — Level 2 strategies for pre-flight checks."""

from __future__ import annotations

from typing import Any, Dict, Optional, Set

from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.core.state import PipelineState
from geny_executor.stages.s04_guard.interface import Guard
from geny_executor.stages.s04_guard.types import GuardResult


class TokenBudgetGuard(Guard):
    """Check if token budget allows another API call."""

    def __init__(self, min_remaining_tokens: int = 10_000):
        self._min_remaining = min_remaining_tokens

    @property
    def name(self) -> str:
        return "token_budget"

    @property
    def description(self) -> str:
        return f"Ensures at least {self._min_remaining} tokens remaining"

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name="token_budget",
            fields=[
                ConfigField(
                    name="min_remaining_tokens",
                    type="integer",
                    label="Min remaining tokens",
                    description="Reject the turn if this many tokens aren't free in the context window.",
                    default=10_000,
                    min_value=0,
                ),
            ],
        )

    def configure(self, config: Dict[str, Any]) -> None:
        n = config.get("min_remaining_tokens")
        if isinstance(n, int) and n >= 0:
            self._min_remaining = n

    def get_config(self) -> Dict[str, Any]:
        return {"min_remaining_tokens": self._min_remaining}

    def check(self, state: PipelineState) -> GuardResult:
        used = state.token_usage.input_tokens + state.token_usage.output_tokens
        remaining = state.context_window_budget - used
        if remaining < self._min_remaining:
            return GuardResult(
                passed=False,
                guard_name=self.name,
                message=f"Token budget low: {remaining} remaining (min {self._min_remaining})",
                action="reject",
            )
        return GuardResult(passed=True, guard_name=self.name)


class CostBudgetGuard(Guard):
    """Check if cost budget is not exceeded."""

    def __init__(self, max_cost_usd: Optional[float] = None):
        self._max_cost = max_cost_usd

    @property
    def name(self) -> str:
        return "cost_budget"

    @property
    def description(self) -> str:
        limit = f"${self._max_cost}" if self._max_cost else "session budget"
        return f"Ensures cost doesn't exceed {limit}"

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name="cost_budget",
            fields=[
                ConfigField(
                    name="max_cost_usd",
                    type="number",
                    label="Max cost (USD)",
                    description="Per-stage hard cap. Empty = fall back to session-level state.cost_budget_usd.",
                    min_value=0,
                ),
            ],
        )

    def configure(self, config: Dict[str, Any]) -> None:
        v = config.get("max_cost_usd")
        if v is None:
            self._max_cost = None
        elif isinstance(v, (int, float)) and v >= 0:
            self._max_cost = float(v)

    def get_config(self) -> Dict[str, Any]:
        return {"max_cost_usd": self._max_cost}

    def check(self, state: PipelineState) -> GuardResult:
        limit = self._max_cost or state.cost_budget_usd
        if limit is None:
            return GuardResult(passed=True, guard_name=self.name)
        if state.total_cost_usd >= limit:
            return GuardResult(
                passed=False,
                guard_name=self.name,
                message=f"Cost budget exceeded: ${state.total_cost_usd:.4f} >= ${limit:.4f}",
                action="reject",
            )
        return GuardResult(passed=True, guard_name=self.name)


class IterationGuard(Guard):
    """Check if iteration limit is not reached."""

    def __init__(self, max_iterations: Optional[int] = None):
        self._max_iterations = max_iterations

    @property
    def name(self) -> str:
        return "iteration"

    @property
    def description(self) -> str:
        return "Prevents infinite loops"

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name="iteration",
            fields=[
                ConfigField(
                    name="max_iterations",
                    type="integer",
                    label="Max iterations",
                    description="Per-stage hard cap. Empty = fall back to session-level state.max_iterations.",
                    min_value=1,
                ),
            ],
        )

    def configure(self, config: Dict[str, Any]) -> None:
        v = config.get("max_iterations")
        if v is None:
            self._max_iterations = None
        elif isinstance(v, int) and v >= 1:
            self._max_iterations = v

    def get_config(self) -> Dict[str, Any]:
        return {"max_iterations": self._max_iterations}

    def check(self, state: PipelineState) -> GuardResult:
        limit = self._max_iterations or state.max_iterations
        if state.iteration >= limit:
            return GuardResult(
                passed=False,
                guard_name=self.name,
                message=f"Iteration limit reached: {state.iteration} >= {limit}",
                action="reject",
            )
        return GuardResult(passed=True, guard_name=self.name)


class PermissionGuard(Guard):
    """Check if pending tool calls are permitted."""

    def __init__(
        self, allowed_tools: Optional[Set[str]] = None, blocked_tools: Optional[Set[str]] = None
    ):
        self._allowed = allowed_tools
        self._blocked = blocked_tools or set()

    @property
    def name(self) -> str:
        return "permission"

    @property
    def description(self) -> str:
        return "Validates tool execution permissions"

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name="permission",
            fields=[
                ConfigField(
                    name="allowed_tools",
                    type="array",
                    item_type="string",
                    label="Allowed tools",
                    description="If non-empty, only these tools may be called. Empty = no allowlist filter.",
                    default=[],
                ),
                ConfigField(
                    name="blocked_tools",
                    type="array",
                    item_type="string",
                    label="Blocked tools",
                    description="Tool names that are always rejected regardless of the allowlist.",
                    default=[],
                ),
            ],
        )

    def configure(self, config: Dict[str, Any]) -> None:
        allowed = config.get("allowed_tools")
        if isinstance(allowed, list):
            self._allowed = {str(x) for x in allowed} if allowed else None
        elif allowed is None and "allowed_tools" in config:
            self._allowed = None
        blocked = config.get("blocked_tools")
        if isinstance(blocked, list):
            self._blocked = {str(x) for x in blocked}

    def get_config(self) -> Dict[str, Any]:
        return {
            "allowed_tools": sorted(self._allowed) if self._allowed else [],
            "blocked_tools": sorted(self._blocked),
        }

    def check(self, state: PipelineState) -> GuardResult:
        for tc in state.pending_tool_calls:
            tool_name = tc.get("tool_name", "")
            if tool_name in self._blocked:
                return GuardResult(
                    passed=False,
                    guard_name=self.name,
                    message=f"Tool '{tool_name}' is blocked",
                    action="reject",
                )
            if self._allowed is not None and tool_name not in self._allowed:
                return GuardResult(
                    passed=False,
                    guard_name=self.name,
                    message=f"Tool '{tool_name}' is not in allowed list",
                    action="reject",
                )
        return GuardResult(passed=True, guard_name=self.name)
