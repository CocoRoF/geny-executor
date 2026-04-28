"""Context strategies — concrete implementations for context collection."""

from __future__ import annotations

from typing import Any, Dict

from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.core.state import PipelineState
from geny_executor.stages.s02_context.interface import ContextStrategy


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

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name="hybrid",
            fields=[
                ConfigField(
                    name="max_recent_turns",
                    type="integer",
                    label="Max recent turns",
                    description="Number of most recent user+assistant turn pairs to keep.",
                    default=20,
                    min_value=1,
                ),
            ],
        )

    def configure(self, config: Dict[str, Any]) -> None:
        n = config.get("max_recent_turns")
        if isinstance(n, int) and n > 0:
            self._max_recent_turns = n

    def get_config(self) -> Dict[str, Any]:
        return {"max_recent_turns": self._max_recent_turns}

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

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name="progressive_disclosure",
            fields=[
                ConfigField(
                    name="summary_threshold",
                    type="integer",
                    label="Summary threshold (turns)",
                    description="Once history exceeds this many turn pairs, older turns are folded into a summary marker.",
                    default=10,
                    min_value=1,
                ),
            ],
        )

    def configure(self, config: Dict[str, Any]) -> None:
        n = config.get("summary_threshold")
        if isinstance(n, int) and n > 0:
            self._summary_threshold = n

    def get_config(self) -> Dict[str, Any]:
        return {"summary_threshold": self._summary_threshold}

    async def build_context(self, state: PipelineState) -> None:
        # If history is short, keep as-is
        if len(state.messages) <= self._summary_threshold * 2:
            return

        # Keep first message (original task) + recent messages
        first = state.messages[:1]
        recent = state.messages[-(self._summary_threshold * 2) :]

        # Insert a summary marker between old and recent
        summary_msg = {
            "role": "user",
            "content": (
                "[Previous conversation summarized. "
                f"{len(state.messages) - len(recent) - 1} messages omitted.]"
            ),
        }
        state.messages = first + [summary_msg] + recent
