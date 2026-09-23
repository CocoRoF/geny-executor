"""Turn replay — the previous turns, back in front of this one, as messages.

See :mod:`geny_executor.memory.short_term_window` for what the window is and
why. This module is the Stage 2 side: when to build it, where it goes, and
the bookkeeping that keeps a replayed message from being recorded twice.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.core.state import PipelineState
from geny_executor.memory.short_term_window import (
    DEFAULT_DIALOGUE_TURNS,
    DEFAULT_FULL_TURNS,
    DEFAULT_SILENT_MARKERS,
    DEFAULT_WINDOW_RATIO,
    WindowConfig,
    build_window,
    window_token_budget,
)
from geny_executor.stages.s02_context.interface import TurnReplay

logger = logging.getLogger(__name__)

#: ``state.metadata`` key the window's description lives under for the turn.
#: The retriever reads it to stand its own recent-turns layer down, and the
#: host reads it to know what the model was shown.
WINDOW_METADATA_KEY = "memory.short_term_window"

#: Stage 18's record watermark (``ProviderDrivenStrategy``). Replayed
#: messages are already in STM — that is where they came from — so the
#: watermark moves past them.
_RECORDED_KEY = "memory.provider_strategy_recorded_idx"
#: Geny's stamping cursor, which shadows the watermark when set.
_HOST_RECORDED_KEYS = ("_stm_recorded_count",)

#: STM rows read to find five turns. A tool-heavy turn is dozens of rows;
#: this is enough for five of them with room to spare, and the file store
#: caches its parsed lines, so the read is not the cost it looks like.
DEFAULT_FETCH_ROWS = 400


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            str(b.get("text") or "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
    return ""


def _is_tool_result_only(content: Any) -> bool:
    return (
        isinstance(content, list)
        and bool(content)
        and all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
    )


def _current_instruction_index(messages: List[Dict[str, Any]]) -> Optional[int]:
    """Index of the message that opened THIS turn: the last user message that
    is not only tool results."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") == "user" and not _is_tool_result_only(message.get("content")):
            return index
    return None


class NoReplay(TurnReplay):
    """Replay nothing: the host carries its own history in ``state.messages``."""

    @property
    def name(self) -> str:
        return "none"

    @property
    def description(self) -> str:
        return "Replay nothing — each turn sees only what the host put in state"

    async def replay(
        self, state: PipelineState, provider: Optional[Any]
    ) -> Optional[Dict[str, Any]]:
        return None


class TurnWindowReplay(TurnReplay):
    """The last few turns from short-term memory, as real messages.

    The nearest ``full_turns`` keep their tool calls and results; the
    ``dialogue_turns`` behind them keep what was said plus one line naming
    what the tools did. The whole window gets ``window_share`` of the
    route's input window, measured in tokens, and the newest turn is kept
    whatever it costs.
    """

    def __init__(
        self,
        full_turns: int = DEFAULT_FULL_TURNS,
        dialogue_turns: int = DEFAULT_DIALOGUE_TURNS,
        window_share: float = DEFAULT_WINDOW_RATIO,
        fetch_rows: int = DEFAULT_FETCH_ROWS,
        silent_markers: Optional[List[str]] = None,
    ) -> None:
        self._full_turns = max(0, int(full_turns))
        self._dialogue_turns = max(0, int(dialogue_turns))
        self._window_share = float(window_share)
        self._fetch_rows = max(1, int(fetch_rows))
        self._silent_markers: List[str] = list(
            DEFAULT_SILENT_MARKERS if silent_markers is None else silent_markers
        )

    @property
    def name(self) -> str:
        return "turn_window"

    @property
    def description(self) -> str:
        return (
            f"Replay the last {self._full_turns + self._dialogue_turns} turns as messages "
            f"({self._full_turns} with their tools, {self._dialogue_turns} conversation only)"
        )

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name="turn_window",
            fields=[
                ConfigField(
                    name="full_turns",
                    type="integer",
                    label="Turns kept whole",
                    description=(
                        "The most recent turns, replayed with every tool call and result "
                        "— the evidence of what was already done."
                    ),
                    default=DEFAULT_FULL_TURNS,
                    min_value=0,
                    max_value=10,
                ),
                ConfigField(
                    name="dialogue_turns",
                    type="integer",
                    label="Turns kept as conversation",
                    description=(
                        "Turns behind those, replayed as what was said, with one line "
                        "naming the tools that ran."
                    ),
                    default=DEFAULT_DIALOGUE_TURNS,
                    min_value=0,
                    max_value=20,
                ),
                ConfigField(
                    name="window_share",
                    type="number",
                    label="Share of the context window",
                    description=(
                        "The most the replay may take of what the model can read, after "
                        "room for its answer. When the turns are larger, older bulk goes "
                        "first; the newest turn always stays."
                    ),
                    default=DEFAULT_WINDOW_RATIO,
                    min_value=0.02,
                    max_value=0.5,
                ),
                ConfigField(
                    name="silent_markers",
                    type="array",
                    label="Answers that mean silence",
                    description=(
                        "A turn whose only answer is one of these, and that ran no tools, "
                        "does not take a slot — so autonomous wake-ups the agent chose not "
                        "to answer do not push the conversation out."
                    ),
                    default=list(DEFAULT_SILENT_MARKERS),
                    item_type="string",
                ),
            ],
        )

    def configure(self, config: Dict[str, Any]) -> None:
        for key, attr, cast in (
            ("full_turns", "_full_turns", int),
            ("dialogue_turns", "_dialogue_turns", int),
            ("fetch_rows", "_fetch_rows", int),
        ):
            value = config.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                setattr(self, attr, cast(value))
        markers = config.get("silent_markers")
        if isinstance(markers, (list, tuple)):
            self._silent_markers = [str(m) for m in markers if str(m).strip()]
        share = config.get("window_share")
        if isinstance(share, (int, float)) and not isinstance(share, bool) and 0 < share <= 0.5:
            self._window_share = float(share)

    def get_config(self) -> Dict[str, Any]:
        return {
            "full_turns": self._full_turns,
            "dialogue_turns": self._dialogue_turns,
            "window_share": self._window_share,
            "silent_markers": list(self._silent_markers),
        }

    async def replay(
        self, state: PipelineState, provider: Optional[Any]
    ) -> Optional[Dict[str, Any]]:
        if provider is None or self._full_turns + self._dialogue_turns <= 0:
            return None

        current = _current_instruction_index(state.messages)
        if current is None:
            return None
        if current > 0:
            # The host already put history in front of this turn — a restored
            # checkpoint, or a host that keeps its state across turns. That
            # history is the real thing; a replay on top would say it twice.
            return None

        try:
            rows = list(await provider.stm().recent(n=self._fetch_rows))
        except Exception:  # noqa: BLE001 — a turn without its past beats no turn
            logger.warning("replay: short-term memory unreadable", exc_info=True)
            return None
        rows = self._without_this_turn(rows, state.messages[current])
        if not rows:
            return None

        budget = window_token_budget(
            getattr(state, "context_window_budget", None),
            reserved_output=getattr(state, "max_tokens", None),
            ratio=self._window_share,
        )
        window = build_window(
            rows,
            WindowConfig(
                full_turns=self._full_turns,
                dialogue_turns=self._dialogue_turns,
                max_tokens=budget,
                silent_markers=tuple(self._silent_markers),
            ),
        )
        if not window.messages:
            return None

        count = len(window.messages)
        state.messages[:0] = window.messages
        self._shift_watermarks(state, count)
        description = {**window.as_metadata(), "messages": count}
        state.metadata[WINDOW_METADATA_KEY] = description
        return description

    @staticmethod
    def _without_this_turn(rows: List[Any], opener: Dict[str, Any]) -> List[Any]:
        """Drop the current instruction if a host recorded it before the turn.

        Stage 18 records it after, which is the common case and leaves
        nothing to drop; a host that records on receipt would otherwise see
        its own question twice — once as history, once as the question.
        """
        want = _text_of(opener.get("content"))
        if not want:
            return rows
        for index in range(len(rows) - 1, -1, -1):
            row = rows[index]
            role = str(getattr(row, "role", "") or "")
            content = getattr(row, "content", "")
            if role != "user" or _is_tool_result_only(content):
                continue
            if _text_of(content) == want and all(
                str(getattr(r, "role", "")) == "user" for r in rows[index:]
            ):
                return rows[:index]
            break
        return rows

    @staticmethod
    def _shift_watermarks(state: PipelineState, count: int) -> None:
        state.metadata[_RECORDED_KEY] = int(state.metadata.get(_RECORDED_KEY, 0) or 0) + count
        for key in _HOST_RECORDED_KEYS:
            value = state.metadata.get(key)
            if isinstance(value, int) and value > 0:
                state.metadata[key] = value + count


__all__ = [
    "DEFAULT_FETCH_ROWS",
    "NoReplay",
    "TurnWindowReplay",
    "WINDOW_METADATA_KEY",
]
