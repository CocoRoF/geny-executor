"""Stage 8: Think — Extended Thinking processing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from geny_executor.core.stage import Stage, StrategyInfo
from geny_executor.core.state import PipelineState
from geny_executor.stages.s08_think.processors import (
    ThinkingProcessor,
    PassthroughProcessor,
    ExtractAndStoreProcessor,
    ThinkingBlock,
    ThinkingResult,
)


class ThinkStage(Stage[Any, Any]):
    """Stage 8: Think.

    Separates thinking content blocks from response blocks.
    Processes thinking via Level 2 ThinkingProcessor strategy.

    When thinking_enabled=True, this stage:
    1. Extracts type="thinking" blocks from API response content
    2. Processes them via the configured ThinkingProcessor
    3. Passes remaining blocks (text, tool_use) to downstream stages
    """

    def __init__(
        self,
        processor: Optional[ThinkingProcessor] = None,
    ):
        self._processor = processor or ExtractAndStoreProcessor()

    @property
    def name(self) -> str:
        return "think"

    @property
    def order(self) -> int:
        return 8

    @property
    def category(self) -> str:
        return "execution"

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
                thinking_blocks.append(ThinkingBlock(
                    text=block.get("thinking", block.get("text", "")),
                    budget_tokens_used=block.get("budget_tokens_used", 0),
                ))
            else:
                response_blocks.append(block)

        # Sum tokens from ORIGINAL blocks before processing (filter may remove some)
        total_thinking_tokens = sum(b.budget_tokens_used for b in thinking_blocks)

        # Process thinking blocks via strategy
        processed = await self._processor.process(thinking_blocks, state)

        state.add_event("think.processed", {
            "thinking_block_count": len(thinking_blocks),
            "total_thinking_tokens": total_thinking_tokens,
        })

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

    def list_strategies(self) -> List[StrategyInfo]:
        return [
            StrategyInfo(
                slot_name="processor",
                current_impl=type(self._processor).__name__,
                available_impls=[
                    "PassthroughProcessor",
                    "ExtractAndStoreProcessor",
                    "ThinkingFilterProcessor",
                ],
            ),
        ]
