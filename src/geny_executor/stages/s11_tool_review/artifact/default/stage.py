"""Pass-through scaffolding for Stage 11: Tool Review (Sub-phase 9a).

A Stage 11 implementation that does nothing — just returns its input
unchanged. Sub-phase 9b replaces this with the real Tool Review
chain (Schema / Sensitive / Destructive / Network / Size reviewers).
The class is shipped now so manifest authors and introspection can
name the slot before the real behaviour lands.
"""

from __future__ import annotations

from typing import Any, Dict

from geny_executor.core.slot import StrategySlot
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState


class ToolReviewStage(Stage[Any, Any]):
    """Stage 11: Tool Review — pass-through scaffold."""

    @property
    def name(self) -> str:
        return "tool_review"

    @property
    def order(self) -> int:
        return 11

    @property
    def category(self) -> str:
        return "review"

    def get_strategy_slots(self) -> Dict[str, StrategySlot]:
        return {}

    async def execute(self, input: Any, state: PipelineState) -> Any:
        # Sub-phase 9a scaffolding: pass-through no-op.
        return input
