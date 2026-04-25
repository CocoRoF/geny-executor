"""Stage 15: Memory — interface definitions."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List

from geny_executor.core.stage import Strategy
from geny_executor.core.state import PipelineState


class MemoryUpdateStrategy(Strategy):
    """Base interface for memory update logic."""

    @abstractmethod
    async def update(self, state: PipelineState) -> None:
        """Update memory based on execution results."""
        ...


class ConversationPersistence(Strategy):
    """Base interface for conversation persistence."""

    @abstractmethod
    async def save(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Save messages for a session."""
        ...

    @abstractmethod
    async def load(self, session_id: str) -> List[Dict[str, Any]]:
        """Load messages for a session."""
        ...

    @abstractmethod
    async def clear(self, session_id: str) -> None:
        """Clear messages for a session."""
        ...
