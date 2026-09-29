"""History compactors — concrete implementations for history compression."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from geny_executor.core.config import ModelConfig
from geny_executor.core.message_repair import strip_leading_orphan_tool_results
from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.core.state import PipelineState
from geny_executor.core.token_estimate import estimate_message_tokens
from geny_executor.memory.turn_text import turn_text
from geny_executor.stages.s02_context.interface import HistoryCompactor


def _safe_recent(messages: list, keep: int) -> list:
    """Last ``keep`` messages, with the window snapped so it never opens
    on a tool_result whose tool_use was dropped (audit D4 / C3)."""
    return strip_leading_orphan_tool_results(messages[-keep:]) if keep > 0 else []


#: Share of the context window the kept tail may fill, by default. Compaction
#: starts at 80%; keeping a quarter leaves the next request well under the
#: line, so one compaction is not followed by another two iterations later.
DEFAULT_KEEP_RECENT_RATIO = 0.25


def _is_tool_results(message: Any) -> bool:
    """A user message that carries tool results (a note may ride along)."""
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    content = message.get("content")
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    )


def _recent_by_tokens(messages: list, budget: int) -> list:
    """The longest tail of ``messages`` that fits ``budget`` tokens.

    A fixed count kept ten messages whatever they held: ten tool results of
    a large file each left the context over the line (compaction again, next
    iteration), ten one-line exchanges threw away most of what fit. The tail
    always holds the last message — and, when that is a batch of tool
    results, the call they answer — even when that alone is over the budget.
    """
    if not messages:
        return []
    floor = len(messages) - 1
    if floor > 0 and _is_tool_results(messages[floor]):
        floor -= 1
    total = 0
    start = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        cost = estimate_message_tokens([messages[i]])
        if i < floor and total + cost > budget:
            break
        total += cost
        start = i
    return strip_leading_orphan_tool_results(messages[start:])


def _is_instruction(message: Any) -> bool:
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    if _is_tool_results(message):
        return False
    content = message.get("content")
    if isinstance(content, list):
        return bool(content)
    return bool(str(content or "").strip())


def _pinned_instruction(messages: list, recent: list) -> Optional[dict]:
    """The request the model is working on, when compaction is about to
    summarise it away.

    The newest instruction is what the whole tool loop answers to. Keeping
    only the last N messages dropped it as soon as the loop ran longer than
    N — a turn deep in its work lost the question it was working on, and
    the summary held it only in paraphrase. It is kept verbatim, right after
    the summary.
    """
    kept = {id(m) for m in recent}
    for message in reversed(messages):
        if _is_instruction(message):
            return None if id(message) in kept else message
    return None


#: Per-message and total size of the transcript handed to the summariser.
_TRANSCRIPT_MESSAGE_CHARS = 1500
_TRANSCRIPT_TOTAL_CHARS = 40_000
_TRANSCRIPT_HEAD_CHARS = 6_000

SUMMARY_PREFIX = "[Earlier in this conversation — compacted to save context]\n"
SUMMARY_ACK = "[Noted — continuing from the summary above.]"


def _transcript(messages: list) -> str:
    """The messages being summarised, tool calls and results included.

    It kept text blocks only, so a stretch of tool work — usually most of
    what gets compacted — reached the summariser as nothing, and its
    results were never summarised at all.
    """
    lines = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        line = turn_text(str(m.get("role") or "user"), m.get("content", ""))
        if len(line) > _TRANSCRIPT_MESSAGE_CHARS:
            line = line[: _TRANSCRIPT_MESSAGE_CHARS - 1] + "…"
        lines.append(line)
    text = "\n".join(lines)
    if len(text) <= _TRANSCRIPT_TOTAL_CHARS:
        return text
    # The start (what was asked, how it began) and the most recent part.
    tail = _TRANSCRIPT_TOTAL_CHARS - _TRANSCRIPT_HEAD_CHARS
    return text[:_TRANSCRIPT_HEAD_CHARS] + "\n[…]\n" + text[-tail:]


def _compacted(summary: str, pinned: Optional[dict], recent: list) -> list:
    head = [
        {"role": "user", "content": SUMMARY_PREFIX + summary},
        {"role": "assistant", "content": SUMMARY_ACK},
    ]
    return head + ([pinned] if pinned is not None else []) + recent


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

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name="truncate",
            fields=[
                ConfigField(
                    name="keep_last",
                    type="integer",
                    label="Keep last (messages)",
                    description="Drop everything older than the last N messages once history exceeds the threshold.",
                    default=20,
                    min_value=1,
                ),
            ],
        )

    def configure(self, config: Dict[str, Any]) -> None:
        n = config.get("keep_last")
        if isinstance(n, int) and n > 0:
            self._keep_last = n

    def get_config(self) -> Dict[str, Any]:
        return {"keep_last": self._keep_last}

    async def compact(self, state: PipelineState) -> None:
        if len(state.messages) > self._keep_last:
            state.messages = _safe_recent(state.messages, self._keep_last)


class SummaryCompactor(HistoryCompactor):
    """Replace old messages with a summary placeholder.

    Non-LLM fallback: replaces dropped messages with a static placeholder.
    See :class:`LLMSummaryCompactor` for the real summarization path that
    calls ``state.llm_client`` when the hosting stage has a model override.
    """

    def __init__(
        self,
        keep_recent: int = 10,
        summary_text: str = "",
        *,
        keep_recent_ratio: float = DEFAULT_KEEP_RECENT_RATIO,
    ):
        self._keep_recent = keep_recent
        self._summary_text = summary_text
        self._keep_recent_ratio = keep_recent_ratio

    def _split(self, state: Any) -> Optional[tuple]:
        """``(older, recent)`` for this compaction, or None when nothing
        would be removed.

        The tail is sized by tokens — ``keep_recent_ratio`` of the context
        window — and by ``keep_recent`` messages only when the ratio is 0.
        """
        messages = list(state.messages or [])
        budget = int(getattr(state, "context_window_budget", 0) or 0)
        recent: Optional[list] = None
        if self._keep_recent_ratio > 0 and budget > 0:
            recent = _recent_by_tokens(messages, int(budget * self._keep_recent_ratio))
            if len(recent) >= len(messages):
                # Every message fits the share: what fills the context is the
                # system prompt and the tools. Removing the oldest messages
                # by count is the only relief left.
                recent = None
        if recent is None:
            if len(messages) <= self._keep_recent:
                return None
            recent = _safe_recent(messages, self._keep_recent)
        kept = {id(m) for m in recent}
        older = [m for m in messages if id(m) not in kept]
        if not older:
            return None
        return older, recent

    @property
    def name(self) -> str:
        return "summary"

    @property
    def description(self) -> str:
        return "Replace old messages with summary"

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name="summary",
            fields=[
                ConfigField(
                    name="keep_recent",
                    type="integer",
                    label="Keep recent (messages)",
                    description="Number of most recent messages to keep verbatim.",
                    default=10,
                    min_value=1,
                ),
                ConfigField(
                    name="keep_recent_ratio",
                    type="number",
                    label="Keep recent (share of the context window)",
                    description=(
                        "How much of the context window the kept recent messages may "
                        "fill. 0 keeps a fixed number of messages instead."
                    ),
                    default=DEFAULT_KEEP_RECENT_RATIO,
                    min_value=0,
                    max_value=0.7,
                ),
                ConfigField(
                    name="summary_text",
                    type="string",
                    label="Summary placeholder",
                    description="Optional static summary text. If empty, a generic placeholder is used.",
                    default="",
                    ui_widget="textarea",
                ),
            ],
        )

    def configure(self, config: Dict[str, Any]) -> None:
        keep = config.get("keep_recent")
        if isinstance(keep, int) and keep > 0:
            self._keep_recent = keep
        text = config.get("summary_text")
        if isinstance(text, str):
            self._summary_text = text
        ratio = config.get("keep_recent_ratio")
        if isinstance(ratio, (int, float)) and not isinstance(ratio, bool) and 0 <= ratio <= 0.7:
            self._keep_recent_ratio = float(ratio)

    def get_config(self) -> Dict[str, Any]:
        return {
            "keep_recent": self._keep_recent,
            "keep_recent_ratio": self._keep_recent_ratio,
            "summary_text": self._summary_text,
        }

    async def compact(self, state: PipelineState) -> None:
        split = self._split(state)
        if split is None:
            return
        older, recent = split
        old_count = len(older)
        pinned = _pinned_instruction(state.messages, recent)

        summary = self._summary_text or (
            f"[{old_count} earlier messages were removed to save context; "
            "no summary of them is available.]"
        )

        state.messages = _compacted(summary, pinned, recent)


class LLMSummaryCompactor(SummaryCompactor):
    """Summary compactor that asks a model to recap what it removes.

    Uses the hosting stage's model override when there is one, otherwise
    the session's own model. Falls back to the static
    :class:`SummaryCompactor` placeholder only when:
      - there is no client (``client_getter`` returns ``None``), or
      - the LLM call raises.

    The request being worked on is kept verbatim after the recap, and the
    recap is written from a transcript that includes the tool calls and
    what they returned.

    Args:
        keep_recent: Number of recent messages to keep verbatim when
            ``keep_recent_ratio`` is 0.
        summary_text: Optional static fallback; used when the LLM returns
            empty text.
        resolve_cfg: Callable taking ``state`` and returning the effective
            :class:`ModelConfig`. Typically bound to
            ``lambda s: parent_stage.resolve_model_config(s)`` by the
            enclosing stage.
        has_override: Callable returning True iff the enclosing stage has
            an explicit override. When False, ``resolve_cfg`` is not used and
            the recap is written by the session's model (``state.model``),
            without thinking and capped at 2048 output tokens.
        client_getter: Callable taking ``state`` and returning the
            :class:`BaseClient`. Defaults to ``state.llm_client``.
    """

    def __init__(
        self,
        keep_recent: int = 10,
        summary_text: str = "",
        *,
        resolve_cfg: Optional[Callable[[PipelineState], ModelConfig]] = None,
        has_override: Optional[Callable[[], bool]] = None,
        client_getter: Optional[Callable[[PipelineState], Any]] = None,
        keep_recent_ratio: float = DEFAULT_KEEP_RECENT_RATIO,
    ):
        super().__init__(
            keep_recent=keep_recent,
            summary_text=summary_text,
            keep_recent_ratio=keep_recent_ratio,
        )
        self._resolve_cfg = resolve_cfg
        self._has_override = has_override or (lambda: False)
        self._client_getter = client_getter or (lambda s: getattr(s, "llm_client", None))

    @property
    def name(self) -> str:
        return "llm_summary"

    @property
    def description(self) -> str:
        return (
            "Model-written recap of the removed messages (placeholder only when there is no client)"
        )

    async def compact(self, state: PipelineState) -> None:
        split = self._split(state)
        if split is None:
            return

        # Resolve the model config. Explicit wiring wins; otherwise self-wire
        # from the live state's model so selecting the ``llm_summary`` compactor
        # in a manifest "just works" (proper LLM compaction instead of the
        # static placeholder) without the host threading resolve_cfg through.
        resolve = self._resolve_cfg
        has_override = self._has_override()
        if resolve is None or not has_override:
            # No model chosen for compaction: summarise with the session's
            # own. The static placeholder this used to fall back to kept
            # nothing of what it replaced — on Geny, which installs this
            # compactor without a per-stage model, every compaction threw
            # the earlier turn away and said so in one line.
            state_model = getattr(state, "model", None)
            if state_model:
                resolve = lambda s: ModelConfig(  # noqa: E731
                    model=getattr(s, "model", state_model) or state_model,
                    max_tokens=2048,
                    temperature=0.0,
                    thinking_enabled=False,
                )
                has_override = True

        if not has_override or resolve is None:
            await super().compact(state)
            return

        client = self._client_getter(state)
        if client is None:
            await super().compact(state)
            return

        old_msgs, recent = split
        old_count = len(old_msgs)
        pinned = _pinned_instruction(state.messages, recent)
        transcript = _transcript(old_msgs)

        prompt = (
            "The conversation context is too long for the model. Compress the OLDER "
            "portion below into a faithful recap so the agent loses nothing "
            "load-bearing. ALWAYS preserve: concrete facts & figures, the user's "
            "requests / preferences / commitments, decisions made and why, named "
            "entities, and any unresolved items / open threads. Drop only redundant "
            "chatter. Say what the tools returned where it matters. Keep it under "
            "~500 words; write a flowing recap, not a bullet list, in the "
            "conversation's own language.\n\n"
            f"<transcript>\n{transcript}\n</transcript>"
        )

        cfg = resolve(state)
        try:
            resp = await client.create_message(
                model_config=cfg,
                messages=[{"role": "user", "content": prompt}],
                purpose="s02.compact",
            )
            summary_text = (resp.text or "").strip()
            if not summary_text:
                summary_text = self._summary_text or (
                    f"[{old_count} earlier messages were removed to save context; "
                    "no summary of them is available.]"
                )
        except Exception as exc:
            state.add_event(
                "memory.compaction.llm_failed",
                {"error": str(exc), "compactor": self.name},
            )
            await super().compact(state)
            return

        state.messages = _compacted(summary_text, pinned, recent)
        state.add_event(
            "memory.compaction.summarized",
            {
                "model": cfg.model,
                "provider": getattr(client, "provider", ""),
                "old_count": old_count,
                "summary_chars": len(summary_text),
            },
        )


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

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name="sliding_window",
            fields=[
                ConfigField(
                    name="window_size",
                    type="integer",
                    label="Window size (messages)",
                    description="Fixed size of the rolling window. Older messages collapse into a single summary marker.",
                    default=30,
                    min_value=1,
                ),
            ],
        )

    def configure(self, config: Dict[str, Any]) -> None:
        n = config.get("window_size")
        if isinstance(n, int) and n > 0:
            self._window_size = n

    def get_config(self) -> Dict[str, Any]:
        return {"window_size": self._window_size}

    async def compact(self, state: PipelineState) -> None:
        if len(state.messages) <= self._window_size:
            return

        overflow = len(state.messages) - self._window_size
        summary = {
            "role": "user",
            "content": f"[{overflow} earlier messages summarized and compacted.]",
        }
        state.messages = [summary] + _safe_recent(state.messages, self._window_size)
