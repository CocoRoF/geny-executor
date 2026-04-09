"""Stage 4: Guard — pre-flight safety checks."""

from __future__ import annotations

from typing import Any, List, Optional

from geny_executor.core.errors import GuardRejectError
from geny_executor.core.stage import Stage, StrategyInfo
from geny_executor.core.state import PipelineState
from geny_executor.stages.s04_guard.guards import Guard, GuardChain


class GuardStage(Stage[Any, Any]):
    """Stage 4: Guard.

    Dual abstraction:
      - Level 2 guard_chain: ordered list of Guard checks
    """

    def __init__(self, guards: Optional[List[Guard]] = None):
        self._chain = GuardChain(guards)

    def add_guard(self, guard: Guard) -> GuardStage:
        """Add a guard to the chain."""
        self._chain.add(guard)
        return self

    @property
    def name(self) -> str:
        return "guard"

    @property
    def order(self) -> int:
        return 4

    @property
    def category(self) -> str:
        return "pre_flight"

    async def execute(self, input: Any, state: PipelineState) -> Any:
        result = self._chain.check_all(state)

        state.add_event(
            "guard.check",
            {
                "passed": result.passed,
                "guard_name": result.guard_name,
                "message": result.message,
            },
        )

        if not result.passed:
            if result.action == "warn":
                # Warn but continue
                state.add_event("guard.warn", {"message": result.message})
                return input
            raise GuardRejectError(
                result.message,
                guard_name=result.guard_name,
            )

        return input

    def list_strategies(self) -> List[StrategyInfo]:
        return [
            StrategyInfo(
                slot_name="guards",
                current_impl=", ".join(type(g).__name__ for g in self._chain.guards) or "none",
                available_impls=[
                    "TokenBudgetGuard",
                    "CostBudgetGuard",
                    "IterationGuard",
                    "PermissionGuard",
                ],
            ),
        ]
