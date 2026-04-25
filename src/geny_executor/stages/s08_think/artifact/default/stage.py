"""Stage 8: Think — concrete stage implementation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from geny_executor.core.slot import StrategySlot
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState
from geny_executor.stages.s08_think.interface import (
    ThinkingBudgetPlanner,
    ThinkingProcessor,
)
from geny_executor.stages.s08_think.types import ThinkingBlock, ThinkingResult
from geny_executor.stages.s08_think.artifact.default.budget import (
    AdaptiveThinkingBudget,
    StaticThinkingBudget,
    apply_thinking_budget,
)
from geny_executor.stages.s08_think.artifact.default.processors import (
    ExtractAndStoreProcessor,
    PassthroughProcessor,
    ThinkingFilterProcessor,
)


class ThinkStage(Stage[Any, Any]):
    """Stage 8: Think.

    Separates thinking content blocks from response blocks.
    Processes thinking via Level 2 ThinkingProcessor strategy.

    When thinking_enabled=True, this stage:
    1. Extracts type="thinking" blocks from API response content
    2. Processes them via the configured ThinkingProcessor
    3. Passes remaining blocks (text, tool_use) to downstream stages

    Per-turn budget sizing (S7.10) lives on the ``budget_planner`` slot.
    Planners run *before* the API call (Stage 6) — invoke
    :meth:`apply_planned_budget` from a host-side hook or test
    fixture. ``execute()`` itself does not auto-invoke the planner
    because Stage 8 only runs after the API response is in hand.
    """

    def __init__(
        self,
        processor: Optional[ThinkingProcessor] = None,
        budget_planner: Optional[ThinkingBudgetPlanner] = None,
    ):
        self._slots: Dict[str, StrategySlot] = {
            "processor": StrategySlot(
                name="processor",
                strategy=processor or ExtractAndStoreProcessor(),
                registry={
                    "passthrough": PassthroughProcessor,
                    "extract_and_store": ExtractAndStoreProcessor,
                    "filter": ThinkingFilterProcessor,
                },
                description="Thinking block processing strategy",
            ),
            "budget_planner": StrategySlot(
                name="budget_planner",
                strategy=budget_planner or StaticThinkingBudget(),
                registry={
                    "static": StaticThinkingBudget,
                    "adaptive": AdaptiveThinkingBudget,
                },
                description=(
                    "Per-turn thinking_budget_tokens planner — invoke via "
                    "apply_planned_budget(state) before the API call"
                ),
            ),
        }

    @property
    def _processor(self) -> ThinkingProcessor:
        return self._slots["processor"].strategy  # type: ignore[return-value]

    @property
    def _budget_planner(self) -> ThinkingBudgetPlanner:
        return self._slots["budget_planner"].strategy  # type: ignore[return-value]

    def apply_planned_budget(self, state: PipelineState) -> int:
        """Run the configured planner and write the result onto state.

        Hosts call this from a pre-Stage-6 hook (or a test) to size
        the per-turn ``thinking_budget_tokens``. Returns the new
        budget; the state mutation and the ``think.budget_applied``
        event are handled by :func:`apply_thinking_budget`.
        """
        return apply_thinking_budget(state, self._budget_planner)

    @property
    def name(self) -> str:
        return "think"

    @property
    def order(self) -> int:
        return 8

    @property
    def category(self) -> str:
        return "execution"

    def get_strategy_slots(self) -> Dict[str, StrategySlot]:
        return self._slots

    def should_bypass(self, state: PipelineState) -> bool:
        """Bypass if extended thinking is not enabled or no thinking blocks present."""
        if not state.thinking_enabled:
            return True
        # Check if last API response has thinking blocks
        response = state.last_api_response
        if response is None:
            return True
        content = self._get_content_blocks(response)
        return not any(self._is_thinking_block(b) for b in content)

    async def execute(self, input: Any, state: PipelineState) -> Any:
        response = state.last_api_response
        if response is None:
            return input

        content = self._get_content_blocks(response)
        if not content:
            return input

        # Separate thinking blocks from response blocks
        thinking_blocks: List[ThinkingBlock] = []
        response_blocks: List[Dict[str, Any]] = []

        for block in content:
            if self._is_thinking_block(block):
                thinking_blocks.append(
                    ThinkingBlock(
                        text=block.get("thinking", block.get("text", "")),
                        budget_tokens_used=block.get("budget_tokens_used", 0),
                    )
                )
            else:
                response_blocks.append(block)

        # Sum tokens from ORIGINAL blocks before processing (filter may remove some)
        total_thinking_tokens = sum(b.budget_tokens_used for b in thinking_blocks)

        # Process thinking blocks via strategy
        processed = await self._processor.process(thinking_blocks, state)

        state.add_event(
            "think.processed",
            {
                "thinking_block_count": len(thinking_blocks),
                "total_thinking_tokens": total_thinking_tokens,
            },
        )

        return ThinkingResult(
            thinking_blocks=processed,
            response_blocks=response_blocks,
            total_thinking_tokens=total_thinking_tokens,
        )

    def _get_content_blocks(self, response: Any) -> List[Dict[str, Any]]:
        """Extract content blocks from API response."""
        if isinstance(response, dict):
            content = response.get("content", [])
            if isinstance(content, list):
                return content
        # Handle Anthropic SDK response object
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, list):
                return [self._block_to_dict(b) for b in content]
        return []

    def _block_to_dict(self, block: Any) -> Dict[str, Any]:
        """Convert an Anthropic SDK content block to dict."""
        if isinstance(block, dict):
            return block
        if hasattr(block, "model_dump"):
            return block.model_dump()
        if hasattr(block, "__dict__"):
            return {k: v for k, v in block.__dict__.items() if not k.startswith("_")}
        return {"type": "unknown", "text": str(block)}

    def _is_thinking_block(self, block: Any) -> bool:
        """Check if a content block is a thinking block."""
        if isinstance(block, dict):
            return block.get("type") == "thinking"
        return getattr(block, "type", None) == "thinking"
