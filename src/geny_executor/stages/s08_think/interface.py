"""Stage 8: Think — interface definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from geny_executor.core.stage import Strategy
from geny_executor.core.state import PipelineState
from geny_executor.stages.s08_think.types import ThinkingBlock


class ThinkingProcessor(Strategy, ABC):
    """Level 2 strategy: how to process thinking content blocks."""

    @abstractmethod
    async def process(
        self,
        thinking_blocks: List[ThinkingBlock],
        state: PipelineState,
    ) -> List[ThinkingBlock]:
        """Process thinking blocks. Return processed blocks."""
        ...
