"""Pass-through scaffolding for Stage 13: Task Registry (Sub-phase 9a)."""

from __future__ import annotations

from typing import Any, Dict

from geny_executor.core.slot import StrategySlot
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState


class TaskRegistryStage(Stage[Any, Any]):
    """Stage 13: Task Registry — pass-through scaffold.

    Sub-phase 9b replaces this with the real Task Registry
    (in-memory + policy slots — eager_wait / fire_and_forget /
    timed_wait). For now it is a no-op so manifest authors and
    introspection can name the slot ahead of the real implementation.
    """

    @property
    def name(self) -> str:
        return "task_registry"

    @property
    def order(self) -> int:
        return 13

    @property
    def category(self) -> str:
        return "orchestration"

    def get_strategy_slots(self) -> Dict[str, StrategySlot]:
        return {}

    async def execute(self, input: Any, state: PipelineState) -> Any:
        return input
