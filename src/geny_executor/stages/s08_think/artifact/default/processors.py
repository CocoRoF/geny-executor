"""Think stage — concrete thinking content processors (Level 2 strategies)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.core.state import PipelineState
from geny_executor.stages.s08_think.interface import ThinkingProcessor
from geny_executor.stages.s08_think.types import ThinkingBlock


class PassthroughProcessor(ThinkingProcessor):
    """Preserve thinking blocks as-is (separation only)."""

    @property
    def name(self) -> str:
        return "passthrough"

    async def process(
        self,
        thinking_blocks: List[ThinkingBlock],
        state: PipelineState,
    ) -> List[ThinkingBlock]:
        return thinking_blocks


class ExtractAndStoreProcessor(ThinkingProcessor):
    """Extract thinking content and store in state.thinking_history."""

    @property
    def name(self) -> str:
        return "extract_and_store"

    async def process(
        self,
        thinking_blocks: List[ThinkingBlock],
        state: PipelineState,
    ) -> List[ThinkingBlock]:
        for block in thinking_blocks:
            state.thinking_history.append(
                {
                    "iteration": state.iteration,
                    "text": block.text,
                    "tokens": block.budget_tokens_used,
                }
            )
        return thinking_blocks


class ThinkingFilterProcessor(ThinkingProcessor):
    """Filter thinking blocks by pattern — e.g., remove sensitive reasoning."""

    def __init__(self, exclude_patterns: Optional[List[str]] = None):
        self._exclude_patterns = exclude_patterns or []

    @property
    def name(self) -> str:
        return "filter"

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name="filter",
            fields=[
                ConfigField(
                    name="exclude_patterns",
                    type="array",
                    item_type="string",
                    label="Exclude patterns",
                    description="Substrings; any thinking block containing one is dropped before storage.",
                    default=[],
                ),
            ],
        )

    def configure(self, config: Dict[str, Any]) -> None:
        patterns = config.get("exclude_patterns")
        if isinstance(patterns, list):
            self._exclude_patterns = [str(p) for p in patterns]

    def get_config(self) -> Dict[str, Any]:
        return {"exclude_patterns": list(self._exclude_patterns)}

    async def process(
        self,
        thinking_blocks: List[ThinkingBlock],
        state: PipelineState,
    ) -> List[ThinkingBlock]:
        if not self._exclude_patterns:
            return thinking_blocks

        filtered = []
        for block in thinking_blocks:
            should_keep = True
            for pattern in self._exclude_patterns:
                if pattern in block.text:
                    should_keep = False
                    break
            if should_keep:
                filtered.append(block)
        return filtered
