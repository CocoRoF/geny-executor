"""History compactors — concrete implementations for history compression."""

from __future__ import annotations

from typing import Any, Callable, Optional

from geny_executor.core.config import ModelConfig
from geny_executor.core.state import PipelineState
from geny_executor.stages.s02_context.interface import HistoryCompactor


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

    Non-LLM fallback: replaces dropped messages with a static placeholder.
    See :class:`LLMSummaryCompactor` for the real summarization path that
    calls ``state.llm_client`` when the hosting stage has a model override.
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


class LLMSummaryCompactor(SummaryCompactor):
    """Summary compactor that calls ``state.llm_client`` when an override is set.

    Falls back to the static :class:`SummaryCompactor` placeholder when:
      - no model override is configured on the hosting stage, or
      - ``state.llm_client`` is ``None``, or
      - the LLM call raises.

    This preserves the no-cost-by-default rule: pipelines that don't opt
    into per-stage compaction models see no new LLM calls.

    Args:
        keep_recent: Number of recent messages to keep verbatim.
        summary_text: Optional static fallback; used when the LLM returns
            empty text.
        resolve_cfg: Callable taking ``state`` and returning the effective
            :class:`ModelConfig`. Typically bound to
            ``lambda s: parent_stage.resolve_model_config(s)`` by the
            enclosing stage.
        has_override: Callable returning True iff the enclosing stage has
            an explicit override. When False, the compactor skips the LLM
            call even though ``resolve_cfg`` would still return a config
            (built from state defaults).
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
    ):
        super().__init__(keep_recent=keep_recent, summary_text=summary_text)
        self._resolve_cfg = resolve_cfg
        self._has_override = has_override or (lambda: False)
        self._client_getter = client_getter or (lambda s: getattr(s, "llm_client", None))

    @property
    def name(self) -> str:
        return "llm_summary"

    @property
    def description(self) -> str:
        return "LLM-backed summary compactor (falls back to placeholder when no override is set)"

    async def compact(self, state: PipelineState) -> None:
        if len(state.messages) <= self._keep_recent:
            return

        if not self._has_override() or self._resolve_cfg is None:
            await super().compact(state)
            return

        client = self._client_getter(state)
        if client is None:
            await super().compact(state)
            return

        old_count = len(state.messages) - self._keep_recent
        old_msgs = state.messages[: -self._keep_recent]
        recent = state.messages[-self._keep_recent :]

        transcript_lines = []
        for m in old_msgs:
            role = m.get("role", "")
            content = m.get("content", "")
            if isinstance(content, list):
                text_parts = [
                    b.get("text", "")
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                content = "\n".join(text_parts)
            transcript_lines.append(f"{role}: {content}")
        transcript = "\n".join(transcript_lines)[:12000]

        prompt = (
            "Summarize the following conversation transcript so the essential "
            "facts, user requests, decisions, and unresolved items are preserved. "
            "Keep it under ~500 words. Write a flat recap, not a bullet list.\n\n"
            f"<transcript>\n{transcript}\n</transcript>"
        )

        cfg = self._resolve_cfg(state)
        try:
            resp = await client.create_message(
                model_config=cfg,
                messages=[{"role": "user", "content": prompt}],
                purpose="s02.compact",
            )
            summary_text = (resp.text or "").strip()
            if not summary_text:
                summary_text = self._summary_text or (
                    f"[Summary of {old_count} previous messages.]"
                )
        except Exception as exc:
            state.add_event(
                "memory.compaction.llm_failed",
                {"error": str(exc), "compactor": self.name},
            )
            await super().compact(state)
            return

        state.messages = [
            {"role": "user", "content": summary_text},
            {
                "role": "assistant",
                "content": "Understood, I have the context from our previous conversation.",
            },
        ] + recent
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

    async def compact(self, state: PipelineState) -> None:
        if len(state.messages) <= self._window_size:
            return

        overflow = len(state.messages) - self._window_size
        summary = {
            "role": "user",
            "content": f"[{overflow} earlier messages summarized and compacted.]",
        }
        state.messages = [summary] + state.messages[-self._window_size :]
