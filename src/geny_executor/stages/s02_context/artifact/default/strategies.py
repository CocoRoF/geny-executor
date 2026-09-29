"""Context strategies — concrete implementations for context collection."""

from __future__ import annotations

from typing import Any, Dict, List

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
        # The last N turns — counted as user instructions, not as pairs of
        # messages. "Two messages per turn" cut into the middle of any turn
        # that used tools: a tool_result without its call (rejected by the
        # API), the current request dropped mid-loop, and a record
        # watermark pointing past the end.
        _keep_last_turns(state, self._max_recent_turns)


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
        # The first request (the original task) and the last N turns, with
        # an honest line between them. It used to claim a summary it never
        # wrote, as a second user message in a row, after cutting by
        # message count into the middle of a tool loop.
        _keep_last_turns(state, self._summary_threshold, keep_first=True)


def _turn_starts(messages: List[Dict[str, Any]]) -> List[int]:
    starts = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        ):
            continue
        starts.append(i)
    return starts


def _keep_last_turns(state: PipelineState, turns: int, *, keep_first: bool = False) -> None:
    """Drop whole turns from the front so at most ``turns`` remain.

    The cut lands on a user instruction, so no call loses its result and the
    current turn is never touched. The record watermark follows the cut
    (``reconcile_recorded_index``), and messages cut before they were
    recorded are kept for Stage 18.
    """
    from geny_executor.core.compaction import reconcile_recorded_index

    starts = _turn_starts(state.messages)
    if turns <= 0 or len(starts) <= turns:
        return
    cut = starts[-turns]
    before = list(state.messages)
    kept = before[cut:]
    if keep_first and starts[0] < cut:
        first = before[starts[0]]
        note = {
            "role": "assistant",
            "content": f"[{cut - starts[0] - 1} earlier messages are not shown.]",
        }
        state.messages = [first, note] + kept
    else:
        state.messages = kept
    reconcile_recorded_index(before, list(state.messages), state.metadata)
