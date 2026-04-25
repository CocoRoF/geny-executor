"""Bypass scaffolding for Stage 15: HITL (Sub-phase 9a)."""

from __future__ import annotations

from typing import Any, Dict

from geny_executor.core.slot import StrategySlot
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState


class HITLStage(Stage[Any, Any]):
    """Stage 15: Human-in-the-loop — bypass scaffold.

    Sub-phase 9b adds the real HITL gate (UI requester + timeout
    policy + Pipeline.resume API). Until then the stage always
    bypasses so existing pipelines that don't need approvals see
    no behaviour change.
    """

    @property
    def name(self) -> str:
        return "hitl"

    @property
    def order(self) -> int:
        return 15

    @property
    def category(self) -> str:
        return "gate"

    def get_strategy_slots(self) -> Dict[str, StrategySlot]:
        return {}

    def should_bypass(self, state: PipelineState) -> bool:
        # Sub-phase 9a scaffolding: always bypass.
        return True

    async def execute(self, input: Any, state: PipelineState) -> Any:
        return input
