"""Multi-format yield formatter (S7.12).

Emits the final result in several formats simultaneously so downstream
consumers can pick the one they need without re-running the pipeline:

* ``text`` — plain ``state.final_text``.
* ``structured`` — same dict shape as :class:`StructuredFormatter`
  (text + model + iterations + cost + token usage + completion signal).
* ``markdown`` — human-readable markdown with optional thinking
  block, completion signal, and a small metadata footer.

Set ``include_thinking=True`` to fold the most recent thinking turn
into the markdown output. Thinking is sourced from
``state.thinking_history``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

from geny_executor.core.state import PipelineState
from geny_executor.stages.s21_yield.interface import ResultFormatter


_DEFAULT_FORMATS: Sequence[str] = ("text", "structured", "markdown")
_VALID_FORMATS = frozenset({"text", "structured", "markdown"})


def build_structured(state: PipelineState) -> Dict[str, Any]:
    """Same shape as :class:`StructuredFormatter` for parity."""
    return {
        "text": state.final_text,
        "model": state.model,
        "iterations": state.iteration,
        "total_cost_usd": state.total_cost_usd,
        "token_usage": {
            "input_tokens": state.token_usage.input_tokens,
            "output_tokens": state.token_usage.output_tokens,
            "total_tokens": state.token_usage.total_tokens,
        },
        "completion_signal": state.completion_signal,
    }


def _last_thinking_text(state: PipelineState) -> str:
    """Return the text of the most recent thinking turn, or ''."""
    history = getattr(state, "thinking_history", None) or []
    for entry in reversed(history):
        if isinstance(entry, dict):
            text = entry.get("text")
            if isinstance(text, str) and text:
                return text
    return ""


def build_markdown(state: PipelineState, *, include_thinking: bool = False) -> str:
    """Render the final state as a markdown document.

    Layout:

        # Result
        <state.final_text>

        ## Thinking            ← only when include_thinking=True and history non-empty
        <last thinking text>

        ## Status              ← only when completion_signal is set
        - signal: <completion_signal>
        - detail: <completion_detail>   ← only when set

        ---
        *Model `<model>` · iterations <n> · cost $<usd>*
    """
    parts: List[str] = []
    parts.append("# Result")
    parts.append("")
    parts.append(state.final_text or "")

    if include_thinking:
        thinking_text = _last_thinking_text(state)
        if thinking_text:
            parts.append("")
            parts.append("## Thinking")
            parts.append("")
            parts.append(thinking_text)

    if state.completion_signal:
        parts.append("")
        parts.append("## Status")
        parts.append(f"- signal: `{state.completion_signal}`")
        if state.completion_detail:
            parts.append(f"- detail: {state.completion_detail}")

    parts.append("")
    parts.append("---")
    parts.append(
        f"*Model `{state.model}` · iterations {state.iteration} · cost ${state.total_cost_usd:.4f}*"
    )
    return "\n".join(parts)


def _validate_formats(formats: Iterable[str]) -> tuple[str, ...]:
    cleaned: List[str] = []
    seen: set[str] = set()
    for fmt in formats:
        if fmt not in _VALID_FORMATS:
            raise ValueError(f"unknown format {fmt!r} (expected one of {sorted(_VALID_FORMATS)})")
        if fmt in seen:
            continue
        seen.add(fmt)
        cleaned.append(fmt)
    if not cleaned:
        raise ValueError("at least one format must be requested")
    return tuple(cleaned)


class MultiFormatFormatter(ResultFormatter):
    """Emit several output formats in one pass.

    ``state.final_output`` becomes a dict keyed by the requested
    format names. Consumers select what they need; unselected formats
    are not computed.
    """

    DEFAULT_FORMATS: Sequence[str] = _DEFAULT_FORMATS

    def __init__(
        self,
        *,
        formats: Sequence[str] = _DEFAULT_FORMATS,
        include_thinking: bool = False,
    ) -> None:
        self._formats = _validate_formats(formats)
        self._include_thinking = bool(include_thinking)

    @property
    def name(self) -> str:
        return "multi_format"

    @property
    def description(self) -> str:
        return "Emits text + structured + markdown formats simultaneously"

    @property
    def formats(self) -> tuple[str, ...]:
        return self._formats

    @property
    def include_thinking(self) -> bool:
        return self._include_thinking

    def format(self, state: PipelineState) -> None:
        output: Dict[str, Any] = {}
        if "text" in self._formats:
            output["text"] = state.final_text or ""
        if "structured" in self._formats:
            output["structured"] = build_structured(state)
        if "markdown" in self._formats:
            output["markdown"] = build_markdown(state, include_thinking=self._include_thinking)
        state.final_output = output


__all__ = [
    "MultiFormatFormatter",
    "build_markdown",
    "build_structured",
]
