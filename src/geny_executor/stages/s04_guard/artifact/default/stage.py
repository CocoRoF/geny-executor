"""Stage 4: Guard — pre-flight safety checks."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

from geny_executor.core.errors import GuardRejectError
from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.core.slot import SlotChain
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState
from geny_executor.stages.s04_guard.interface import Guard, GuardChain
from geny_executor.stages.s04_guard.artifact.default.guards import (
    CostBudgetGuard,
    IterationGuard,
    PermissionGuard,
    TokenBudgetGuard,
)


class GuardStage(Stage[Any, Any]):
    """Stage 4: Guard.

    Dual abstraction:
      - Level 2 guards chain: ordered list of Guard checks
    """

    def __init__(
        self,
        guards: Optional[List[Guard]] = None,
        *,
        max_chain_length: int = 32,
        fail_fast: bool = True,
    ):
        self._chains: Dict[str, SlotChain] = {
            "guards": SlotChain(
                name="guards",
                items=list(guards or []),
                registry={
                    "token_budget": TokenBudgetGuard,
                    "cost_budget": CostBudgetGuard,
                    "iteration": IterationGuard,
                    "permission": PermissionGuard,
                },
                description="Ordered chain of pre-flight guard checks",
            ),
        }
        self._max_chain_length = int(max_chain_length)
        self._fail_fast = bool(fail_fast)

    @property
    def guards(self) -> SlotChain:
        """Public handle on the guard chain (list/mutate items directly)."""
        return self._chains["guards"]

    @property
    def _guard_chain(self) -> SlotChain:
        return self._chains["guards"]

    def add_guard(self, guard: Guard) -> "GuardStage":
        """Append a guard to the chain.

        .. deprecated::
            Prefer :meth:`add_to_chain("guards", impl_name)` for hot-swappable
            configuration. Retained for backward compatibility with builders
            that pre-construct Guard instances.
        """
        warnings.warn(
            "GuardStage.add_guard() is deprecated; use "
            "stage.add_to_chain('guards', impl_name) or pre-populate via "
            "GuardStage(guards=[...]).",
            DeprecationWarning,
            stacklevel=2,
        )
        self._guard_chain.add(guard)
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

    def get_strategy_chains(self) -> Dict[str, SlotChain]:
        return self._chains

    def get_config_schema(self) -> ConfigSchema:
        return ConfigSchema(
            name="guard",
            fields=[
                ConfigField(
                    name="max_chain_length",
                    type="integer",
                    label="Max Chain Length",
                    description="Reject configurations with more than this many guards.",
                    default=32,
                    min_value=1,
                ),
                ConfigField(
                    name="fail_fast",
                    type="boolean",
                    label="Fail Fast",
                    description="Abort on the first failing guard instead of collecting all.",
                    default=True,
                    ui_widget="toggle",
                ),
            ],
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "max_chain_length": self._max_chain_length,
            "fail_fast": self._fail_fast,
        }

    def update_config(self, config: Dict[str, Any]) -> None:
        if "max_chain_length" in config:
            self._max_chain_length = int(config["max_chain_length"])
        if "fail_fast" in config:
            self._fail_fast = bool(config["fail_fast"])

    async def execute(self, input: Any, state: PipelineState) -> Any:
        chain = GuardChain(self._guard_chain.items)
        result = chain.check_all(state)

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
                state.add_event("guard.warn", {"message": result.message})
                return input
            raise GuardRejectError(
                result.message,
                guard_name=result.guard_name,
            )

        return input
