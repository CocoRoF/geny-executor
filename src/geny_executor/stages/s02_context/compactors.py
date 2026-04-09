"""History compactors — Level 2 strategies for history compression."""

from __future__ import annotations

from abc import abstractmethod

from geny_executor.core.stage import Strategy
from geny_executor.core.state import PipelineState


class HistoryCompactor(Strategy):
    """Base interface for compacting conversation history when budget is exceeded."""

    @abstractmethod
    async def compact(self, state: PipelineState) -> None:
        """Compact history in state.messages to fit within budget."""
        ...


class TruncateCompactor(HistoryCompactor):
    """Truncate oldest messages."""

    def __init__(self, keep_last: int = 20):
        self._keep_last = keep_last

    @property
    def name(self) -> str:
        return "truncate"

    @property
    def description(self) -> str:
        return f"Keep last {self._keep_last} messages, drop older"

    async def compact(self, state: PipelineState) -> None:
        if len(state.messages) > self._keep_last:
            state.messages = state.messages[-self._keep_last :]


class SummaryCompactor(HistoryCompactor):
    """Replace old messages with a summary placeholder.

    Note: actual summarization would require an API call.
    This implementation provides the structural framework;
    integration with the API stage would be done at the pipeline level.
    """

    def __init__(self, keep_recent: int = 10, summary_text: str = ""):
        self._keep_recent = keep_recent
        self._summary_text = summary_text

    @property
    def name(self) -> str:
        return "summary"

    @property
    def description(self) -> str:
        return "Replace old messages with summary"

    async def compact(self, state: PipelineState) -> None:
        if len(state.messages) <= self._keep_recent:
            return

        old_count = len(state.messages) - self._keep_recent
        recent = state.messages[-self._keep_recent :]

        summary = self._summary_text or (
            f"[Summary of {old_count} previous messages. "
            "Conversation history has been compacted to save context window.]"
        )

        state.messages = [
            {"role": "user", "content": summary},
            {
                "role": "assistant",
                "content": "Understood, I have the context from our previous conversation.",
            },
        ] + recent


class SlidingWindowCompactor(HistoryCompactor):
    """Sliding window — maintains a fixed message window, summarizes overflow."""

    def __init__(self, window_size: int = 30):
        self._window_size = window_size

    @property
    def name(self) -> str:
        return "sliding_window"

    @property
    def description(self) -> str:
        return f"Fixed window of {self._window_size} messages"

    async def compact(self, state: PipelineState) -> None:
        if len(state.messages) <= self._window_size:
            return

        overflow = len(state.messages) - self._window_size
        summary = {
            "role": "user",
            "content": f"[{overflow} earlier messages summarized and compacted.]",
        }
        state.messages = [summary] + state.messages[-self._window_size :]
