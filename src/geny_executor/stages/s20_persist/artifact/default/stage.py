"""NoPersistStrategy scaffolding for Stage 20: Persist (Sub-phase 9a)."""

from __future__ import annotations

from typing import Any, Dict

from geny_executor.core.slot import StrategySlot
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState


class PersistStage(Stage[Any, Any]):
    """Stage 20: Persist — NoPersist scaffold.

    Sub-phase 9b adds the real session-checkpoint stage (crash
    recovery / time-travel snapshots — distinct from Memory's
    storage). For now it is a no-op so existing pipelines see no
    behaviour change.
    """

    @property
    def name(self) -> str:
        return "persist"

    @property
    def order(self) -> int:
        return 20

    @property
    def category(self) -> str:
        return "finalize"

    def get_strategy_slots(self) -> Dict[str, StrategySlot]:
        return {}

    async def execute(self, input: Any, state: PipelineState) -> Any:
        return input
