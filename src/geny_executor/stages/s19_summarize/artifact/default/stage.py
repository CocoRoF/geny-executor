"""No-op scaffolding for Stage 19: Summarize (Sub-phase 9a)."""

from __future__ import annotations

from typing import Any, Dict

from geny_executor.core.slot import StrategySlot
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState


class SummarizeStage(Stage[Any, Any]):
    """Stage 19: Summarize — no-op scaffold.

    Sub-phase 9b adds the real summarize stage (LTM index generation
    after Memory). For now it is a pass-through so existing
    pipelines see no behaviour change.
    """

    @property
    def name(self) -> str:
        return "summarize"

    @property
    def order(self) -> int:
        return 19

    @property
    def category(self) -> str:
        return "finalize"

    def get_strategy_slots(self) -> Dict[str, StrategySlot]:
        return {}

    async def execute(self, input: Any, state: PipelineState) -> Any:
        return input
