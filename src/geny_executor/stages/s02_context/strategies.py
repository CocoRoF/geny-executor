"""Context strategies — Level 2 strategies for context collection."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional

from geny_executor.core.stage import Strategy
from geny_executor.core.state import PipelineState


class ContextStrategy(Strategy):
    """Base interface for context collection."""

    @abstractmethod
    async def build_context(self, state: PipelineState) -> None:
        """Build context by modifying state (loading history, memory, etc.)."""
        ...


class SimpleLoadStrategy(ContextStrategy):
    """Simple context — uses whatever is already in state.messages."""

    @property
    def name(self) -> str:
        return "simple_load"

    @property
    def description(self) -> str:
        return "Uses existing messages as-is"

    async def build_context(self, state: PipelineState) -> None:
        # Messages are already in state from previous iterations
        pass


class HybridStrategy(ContextStrategy):
    """Hybrid — recent history + memory injection.

    Keeps the last N turns of history and injects memory refs.
    """

    def __init__(self, max_recent_turns: int = 20):
        self._max_recent_turns = max_recent_turns

    @property
    def name(self) -> str:
        return "hybrid"

    @property
    def description(self) -> str:
        return f"Recent {self._max_recent_turns} turns + memory injection"

    async def build_context(self, state: PipelineState) -> None:
        # Trim history to last N messages (each turn = user + assistant = 2 messages)
        max_messages = self._max_recent_turns * 2
        if len(state.messages) > max_messages:
            state.messages = state.messages[-max_messages:]


class ProgressiveDisclosureStrategy(ContextStrategy):
    """OpenAI-style progressive disclosure.

    Start with summaries, expand relevant parts on demand.
    """

    def __init__(self, summary_threshold: int = 10):
        self._summary_threshold = summary_threshold

    @property
    def name(self) -> str:
        return "progressive_disclosure"

    @property
    def description(self) -> str:
        return "Start with summaries, expand relevant parts"

    async def build_context(self, state: PipelineState) -> None:
        # If history is short, keep as-is
        if len(state.messages) <= self._summary_threshold * 2:
            return

        # Keep first message (original task) + recent messages
        first = state.messages[:1]
        recent = state.messages[-(self._summary_threshold * 2):]

        # Insert a summary marker between old and recent
        summary_msg = {
            "role": "user",
            "content": (
                "[Previous conversation summarized. "
                f"{len(state.messages) - len(recent) - 1} messages omitted.]"
            ),
        }
        state.messages = first + [summary_msg] + recent
