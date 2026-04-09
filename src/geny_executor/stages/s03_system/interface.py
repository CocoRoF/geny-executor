"""Stage 3: System — interface definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from geny_executor.core.stage import Strategy
from geny_executor.core.state import PipelineState


class PromptBuilder(Strategy):
    """Base interface for system prompt construction."""

    @abstractmethod
    def build(self, state: PipelineState) -> Union[str, List[Dict[str, Any]]]:
        """Build system prompt. Returns str or content blocks (for caching)."""


class PromptBlock(ABC):
    """A composable block of system prompt content."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this block."""

    @abstractmethod
    def render(self, state: PipelineState) -> str:
        """Render this block to text."""

    @property
    def cache_control(self) -> Optional[Dict[str, str]]:
        """Optional cache_control for this block."""
        return None
