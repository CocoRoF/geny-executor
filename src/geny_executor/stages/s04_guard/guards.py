"""Guard implementations — Level 2 strategies for pre-flight checks."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Set

from geny_executor.core.stage import Strategy
from geny_executor.core.state import PipelineState


@dataclass
class GuardResult:
    """Result of a guard check."""

    passed: bool
    guard_name: str = ""
    message: str = ""
    action: str = "reject"  # "reject" | "modify" | "warn"


class Guard(Strategy):
    """Base interface for individual guard checks."""

    @abstractmethod
    def check(self, state: PipelineState) -> GuardResult:
        """Run guard check. Returns GuardResult."""
        ...


class GuardChain:
    """Chain of guards — all must pass for execution to proceed."""

    def __init__(self, guards: Optional[List[Guard]] = None):
        self._guards = list(guards or [])

    def add(self, guard: Guard) -> GuardChain:
        self._guards.append(guard)
        return self

    def check_all(self, state: PipelineState) -> GuardResult:
        """Run all guards. First failure stops the chain."""
        for guard in self._guards:
            result = guard.check(state)
            if not result.passed:
                return result
        return GuardResult(passed=True, guard_name="chain", message="All guards passed")

    @property
    def guards(self) -> List[Guard]:
        return list(self._guards)


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
