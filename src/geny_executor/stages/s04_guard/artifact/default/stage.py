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
from geny_executor.stages.s04_guard.types import GuardResult
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
        # Budget-recovery wiring (2.5.0). A guard may return
        # ``action="compact"`` (the token-budget guard does when the
        # projected request is too large). When a compactor is wired,
        # ``execute`` compacts ``state.messages`` and re-checks once
        # before rejecting. Auto-wired from the Context stage's compactor
        # by ``Pipeline._init_state`` each turn; a host may override via
        # :meth:`attach_budget_recovery`. ``None`` → "compact" degrades
        # to a hard reject (pre-2.5.0 behaviour).
        self._budget_compactor: Optional[Any] = None
        self._memory_provider: Optional[Any] = None
        # True once a host explicitly wired recovery — stops the
        # per-turn auto-wire (Pipeline._init_state) from clobbering it.
        self._budget_recovery_explicit = False

    def attach_budget_recovery(
        self, compactor: Optional[Any], provider: Optional[Any] = None
    ) -> "GuardStage":
        """Wire the compactor (and optional memory provider) used to recover
        from a ``compact`` guard signal. See class/__init__ notes."""
        self._budget_compactor = compactor
        self._memory_provider = provider
        self._budget_recovery_explicit = True
        return self

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
        guards = self._guard_chain.items

        # 2026-06-09 audit ("validated-but-inert" table): both knobs below
        # passed schema validation and then were never read here. An
        # operator who edited them saw a green check and no behaviour
        # change — the masked-degradation pattern this release closes.
        if len(guards) > self._max_chain_length:
            raise GuardRejectError(
                f"Guard chain has {len(guards)} guards, exceeding "
                f"max_chain_length={self._max_chain_length}. Raise the limit via "
                "stage config or remove guards from the chain.",
                guard_name="chain",
            )

        if self._fail_fast:
            result = GuardChain(guards).check_all(state)

            state.add_event(
                "guard.check",
                {
                    "passed": result.passed,
                    "guard_name": result.guard_name,
                    "message": result.message,
                },
            )

            if result.passed:
                return input

            if result.action == "warn":
                state.add_event("guard.warn", {"message": result.message})
                return input

            # Recoverable budget pressure: compact history and re-check
            # once before giving up. Only the first compact-actionable
            # failure triggers recovery; a recovered chain re-runs clean.
            if result.action == "compact" and await self._recover_budget(state, result):
                result = GuardChain(guards).check_all(state)
                state.add_event(
                    "guard.check",
                    {
                        "passed": result.passed,
                        "guard_name": result.guard_name,
                        "message": result.message,
                        "recheck": True,
                    },
                )
                if result.passed:
                    return input
                if result.action == "warn":
                    state.add_event("guard.warn", {"message": result.message})
                    return input

            raise GuardRejectError(
                result.message,
                guard_name=result.guard_name,
            )

        # fail_fast=False — run EVERY guard and aggregate. Operators use
        # this to see the full violation picture in one turn instead of
        # fixing guards one rejection at a time.
        failures = [r for r in (guard.check(state) for guard in guards) if not r.passed]

        # Recoverable budget pressure: if any failure asked to compact,
        # compact once and re-evaluate the whole chain before aggregating
        # rejects (a recovered token_budget guard drops out of the list).
        if any(r.action == "compact" for r in failures):
            compact_result = next(r for r in failures if r.action == "compact")
            if await self._recover_budget(state, compact_result):
                failures = [
                    r for r in (guard.check(state) for guard in guards) if not r.passed
                ]

        rejects = [r for r in failures if r.action not in ("warn",)]

        state.add_event(
            "guard.check",
            {
                "passed": not failures,
                "guard_name": ", ".join(r.guard_name for r in failures) or "chain",
                "message": "; ".join(r.message for r in failures) or "All guards passed",
                "violations": [
                    {"guard_name": r.guard_name, "message": r.message, "action": r.action}
                    for r in failures
                ],
            },
        )

        for r in failures:
            if r.action == "warn":
                state.add_event("guard.warn", {"message": r.message})

        if rejects:
            raise GuardRejectError(
                "; ".join(f"{r.guard_name}: {r.message}" for r in rejects),
                guard_name=", ".join(r.guard_name for r in rejects),
            )

        return input

    async def _recover_budget(self, state: PipelineState, result: GuardResult) -> bool:
        """Compact history in response to a ``compact`` guard signal.

        Returns True iff a compactor was wired and ran (so the caller
        should re-check the chain). With no compactor wired the signal
        degrades to a hard reject — the pre-2.5.0 behaviour.
        """
        if self._budget_compactor is None:
            return False

        state.add_event(
            "guard.compacting",
            {"guard_name": result.guard_name, "reason": result.message},
        )
        # Lazy import: keeps the guard stage import-light and avoids a
        # core→stage cycle at module load.
        from geny_executor.core.compaction import run_compaction

        outcome = await run_compaction(
            state,
            self._budget_compactor,
            trigger="guard",
            provider=self._memory_provider,
        )
        return bool(outcome.get("ok"))
