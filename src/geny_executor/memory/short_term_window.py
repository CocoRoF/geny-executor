"""The last few turns, replayed as MESSAGES.

## What was wrong

A Geny turn starts from a fresh ``PipelineState``: ``state.messages`` is
empty every time. The only thing carrying the conversation across turns was
the retriever's ``recent_turns`` layer, and that layer

* counted STM **rows**, not turns — one tool use makes a turn four rows
  (instruction · ``tool_use`` · ``tool_result`` · answer), so ``recent(6)``
  was about one and a half turns;
* flattened each row to its **text blocks only**, dropping every
  ``tool_use`` and ``tool_result``;
* rendered the survivors as ``[role] text`` lines inside the system
  prompt's ``# Relevant Knowledge``, left-truncated to a character cap.

So the agent read its own past as *knowledge to act on* rather than as
*history*, and the evidence of what it had already done — which lives in
tool results — was not there at all. It re-ran finished work, could not tell
a task was complete, and treated three-turn-old statements as facts on a par
with the user's pinned notes.

## What this does instead

Rebuild the recent conversation as real messages and put them in front of
the turn. Every reference implementation replays history as messages:
Anthropic's own context management (``clear_tool_uses_20250919``) keeps the
``tool_use`` block and clears only the *result*; Hermes protects a head and
a tail of real messages and compresses only the middle.

The window is measured in **logical turns** — one user instruction and
everything that followed it, however many tool calls that took:

* **The nearest turns keep their tools.** User text, assistant text,
  ``tool_use`` and ``tool_result``, in order, with ids intact. A large
  result keeps its head and says how much was cut — the block stays, which
  is what tells the model the call happened.
* **The turns behind those keep only the conversation.** User text and the
  assistant's final answer, plus one line naming what the tools did and to
  what — ``[used tools: Read(inv.txt, cfg.yaml), Bash ×2 (1 failed)]``.
  Without the targets the line cannot answer "have I already read that?",
  which is the question that makes an agent repeat itself.
* **Turn outcome** rides on that line when anything failed, because "it is
  done" and "it was attempted" are the distinction the agent keeps missing.

Pairs are never split: a ``tool_use`` without its result gets a synthetic
one, and an orphan result at the head is dropped. ``thinking`` blocks are
not replayed — they are bound to the model that produced them and a
different hop rejects them.

## Budget

A turn count is a floor, not a bound — five turns of large tool results do
not fit a 32k-token model. :func:`window_char_budget` takes the smaller of
an absolute ceiling and a fraction of the route's context window, so the
constant governs frontier models and the fraction protects small ones.
Under pressure the window degrades in a fixed order rather than dropping the
newest thing it has.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from geny_executor.core.message_repair import (
    repair_dangling_tool_calls,
    strip_leading_orphan_tool_results,
)

logger = logging.getLogger(__name__)

__all__ = [
    "WindowConfig",
    "WindowResult",
    "LogicalTurn",
    "build_window",
    "group_logical_turns",
    "window_char_budget",
]

#: How many recent turns keep their tool blocks verbatim.
DEFAULT_FULL_TURNS = 2
#: How many turns behind those keep conversation only.
DEFAULT_DIALOGUE_TURNS = 3

#: Absolute ceiling on the whole window, in characters. Roughly 10k tokens —
#: this is "the last few turns", not "as much as fits".
DEFAULT_MAX_CHARS = 40_000
#: Fraction of the route's context window the window may occupy. Only binds
#: on small models; at 200k tokens the ceiling above is far lower.
DEFAULT_WINDOW_RATIO = 0.15
#: Characters per token used to convert the token-denominated context budget.
#: Matches the estimator the pipeline's guards use.
CHARS_PER_TOKEN = 4
#: Never shrink the window below this — under it the history is fragments.
MIN_MAX_CHARS = 4_000

#: A tool result longer than this keeps its head only.
DEFAULT_RESULT_TRIM_OVER = 4_000
DEFAULT_RESULT_KEEP = 1_200
#: Budget pressure never trims a kept result below this.
MIN_RESULT_KEEP = 300
#: Cap on one utterance in a dialogue-only turn.
DEFAULT_DIALOGUE_MESSAGE_CHARS = 4_000
MIN_DIALOGUE_MESSAGE_CHARS = 400
#: Targets named on the ``[used tools: …]`` line, per tool.
MAX_TOOL_TARGETS = 3

_TRIM_NOTE = "…[+{n} chars trimmed]"


def window_char_budget(
    context_window_budget: Optional[int],
    *,
    ceiling: int = DEFAULT_MAX_CHARS,
    ratio: float = DEFAULT_WINDOW_RATIO,
) -> int:
    """Characters the window may spend, given the route's token window.

    ``min(ceiling, ratio × window)`` — the ceiling is what a reasonable "last
    few turns" costs and governs the frontier models; the ratio is what stops
    the window eating a third of a 32k-token endpoint. ``None`` (no hop could
    say) keeps the ceiling, which is the pre-existing behaviour.
    """
    if not context_window_budget or context_window_budget <= 0:
        return ceiling
    share = int(context_window_budget * ratio) * CHARS_PER_TOKEN
    return max(MIN_MAX_CHARS, min(ceiling, share))


@dataclass
class WindowConfig:
    full_turns: int = DEFAULT_FULL_TURNS
    dialogue_turns: int = DEFAULT_DIALOGUE_TURNS
    max_chars: int = DEFAULT_MAX_CHARS
    result_trim_over: int = DEFAULT_RESULT_TRIM_OVER
    result_keep: int = DEFAULT_RESULT_KEEP
    dialogue_message_chars: int = DEFAULT_DIALOGUE_MESSAGE_CHARS
    used_tools_line: bool = True

    @property
    def turns(self) -> int:
        return self.full_turns + self.dialogue_turns

    @property
    def enabled(self) -> bool:
        return self.turns > 0


@dataclass
class WindowResult:
    """The messages to prepend, and what they cost."""

    messages: List[Dict[str, Any]] = field(default_factory=list)
    #: Logical turns actually represented.
    turns: int = 0
    #: How many of those kept their tool blocks.
    full: int = 0
    #: How many were reduced to conversation only.
    dialogue: int = 0
    chars: int = 0
    #: Degradation steps taken, in order, for the event payload.
    degraded: List[str] = field(default_factory=list)

    def as_metadata(self) -> Dict[str, Any]:
        return {
            "turns": self.turns,
            "full": self.full,
            "dialogue": self.dialogue,
            "chars": self.chars,
            "degraded": list(self.degraded),
        }


def _blocks(content: Any) -> List[Dict[str, Any]]:
    """Normalise message content to a block list."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content.strip() else []
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return []


def _is_tool_result_only(content: Any) -> bool:
    """A user message that is only tool results does not start a turn.

    Counting turns by role alone is what made six rows read as six turns: a
    tool result is a user message too.
    """
    blocks = _blocks(content)
    return bool(blocks) and all(b.get("type") == "tool_result" for b in blocks)


@dataclass
class LogicalTurn:
    """One user instruction and everything that followed it."""

    messages: List[Any] = field(default_factory=list)

    def tool_calls(self) -> List[Tuple[str, str, Any]]:
        """``(tool_use_id, name, input)`` for every call in this turn."""
        out: List[Tuple[str, str, Any]] = []
        for m in self.messages:
            for b in _blocks(getattr(m, "content", "")):
                if b.get("type") == "tool_use":
                    out.append((str(b.get("id") or ""), str(b.get("name") or "?"), b.get("input")))
        return out

    def results_by_id(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for m in self.messages:
            for b in _blocks(getattr(m, "content", "")):
                if b.get("type") == "tool_result":
                    out[str(b.get("tool_use_id") or "")] = b
        return out


def group_logical_turns(turns: Sequence[Any], limit: int) -> List[LogicalTurn]:
    """Group STM rows into logical turns and return the last *limit*.

    The boundary is a user message that is not purely tool results. Anything
    before the first boundary is dropped: an answer with no instruction in
    front of it is a fragment, and a fragment reads as a claim.
    """
    if limit <= 0:
        return []
    groups: List[List[Any]] = []
    for row in turns:
        role = str(getattr(row, "role", "") or "")
        content = getattr(row, "content", "")
        starts = role == "user" and not _is_tool_result_only(content)
        if starts:
            groups.append([row])
        elif groups:
            groups[-1].append(row)
    return [LogicalTurn(g) for g in groups[-limit:]]


def _clip(text: str, limit: int) -> str:
    """Keep *limit* characters of content, and say what was dropped.

    The limit counts CONTENT, not the result string: the note is added on
    top. Counting the note inside the limit made the floor a lie — asking to
    keep at least 300 characters returned 298 of them, and a floor that does
    not hold is worse than no floor, because the number in the config reads
    as a guarantee.

    Silently cutting is how a truncated directory listing reads as an empty
    one, so the note is not optional.
    """
    s = str(text)
    if limit <= 0 or len(s) <= limit:
        return s
    return s[:limit] + _TRIM_NOTE.format(n=len(s) - limit)


def _result_text(block: Dict[str, Any]) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(b.get("text", ""))
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p)
    return "" if content is None else json.dumps(content, ensure_ascii=False)


def _target_of(tool_input: Any) -> str:
    """The one argument that says WHICH thing a call touched.

    ``Read ×3`` cannot answer "have I already read that file"; ``Read(a, b)``
    can, and that question is the one behind most repeated work.
    """
    if not isinstance(tool_input, dict):
        return ""
    for key in (
        "file_path",
        "path",
        "filename",
        "file",
        "notebook_path",
        "pattern",
        "query",
        "url",
        "command",
        "cmd",
    ):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            short = value.strip().split("/")[-1] if "/" in value else value.strip()
            return _clip(short, 40)
    return ""


def _used_tools_line(turn: LogicalTurn) -> str:
    """``[used tools: Read(inv.txt), Bash ×2 (1 failed)]`` or ``""``."""
    calls = turn.tool_calls()
    if not calls:
        return ""
    results = turn.results_by_id()
    order: List[str] = []
    counts: Dict[str, int] = {}
    failed: Dict[str, int] = {}
    targets: Dict[str, List[str]] = {}
    for call_id, name, tool_input in calls:
        if name not in counts:
            order.append(name)
            counts[name] = 0
            failed[name] = 0
            targets[name] = []
        counts[name] += 1
        if results.get(call_id, {}).get("is_error"):
            failed[name] += 1
        target = _target_of(tool_input)
        if target and target not in targets[name] and len(targets[name]) < MAX_TOOL_TARGETS:
            targets[name].append(target)

    parts: List[str] = []
    for name in order:
        piece = name
        if targets[name]:
            piece += f"({', '.join(targets[name])})"
        if counts[name] > 1:
            piece += f" ×{counts[name]}"
        if failed[name]:
            piece += f" ({failed[name]} failed)"
        parts.append(piece)
    return f"[used tools: {', '.join(parts)}]"


def _text_of(message: Any) -> str:
    parts = [
        str(b.get("text", ""))
        for b in _blocks(getattr(message, "content", ""))
        if b.get("type") == "text"
    ]
    return "\n".join(p for p in parts if p.strip()).strip()


def _render_full(turn: LogicalTurn, cfg: WindowConfig, result_keep: int) -> List[Dict[str, Any]]:
    """A turn with its tools intact; only large results lose their tail."""
    out: List[Dict[str, Any]] = []
    for message in turn.messages:
        role = str(getattr(message, "role", "") or "user")
        rendered: List[Dict[str, Any]] = []
        for block in _blocks(getattr(message, "content", "")):
            btype = block.get("type")
            if btype in ("thinking", "redacted_thinking"):
                # Bound to the model that produced it; a different hop
                # rejects the signature.
                continue
            if btype == "tool_result":
                body = _result_text(block)
                if len(body) > cfg.result_trim_over:
                    body = _clip(body, result_keep)
                kept: Dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": block.get("tool_use_id"),
                    "content": body,
                }
                if block.get("is_error"):
                    kept["is_error"] = True
                rendered.append(kept)
                continue
            if btype == "image":
                # An image from three turns ago costs what a whole turn of
                # text costs, and the answer that used it is right there.
                rendered.append({"type": "text", "text": "[image from an earlier turn]"})
                continue
            rendered.append(dict(block))
        if rendered:
            out.append({"role": role, "content": rendered})
    return out


def _render_dialogue(
    turn: LogicalTurn, cfg: WindowConfig, message_chars: int
) -> List[Dict[str, Any]]:
    """A turn reduced to what was said, plus one line about what was done."""
    out: List[Dict[str, Any]] = []
    instruction = ""
    answer = ""
    for message in turn.messages:
        role = str(getattr(message, "role", "") or "user")
        content = getattr(message, "content", "")
        text = _text_of(message)
        if not text:
            continue
        if role == "user" and not _is_tool_result_only(content):
            if not instruction:
                instruction = text
        elif role == "assistant":
            answer = text  # the last assistant text is the turn's answer

    if instruction:
        out.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": _clip(instruction, message_chars)}],
            }
        )
    tail = _clip(answer, message_chars) if answer else ""
    if cfg.used_tools_line:
        line = _used_tools_line(turn)
        if line:
            tail = f"{tail}\n{line}" if tail else line
    if tail:
        out.append({"role": "assistant", "content": [{"type": "text", "text": tail}]})
    return out


def _measure(messages: Iterable[Dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                total += len(str(block.get("text") or ""))
            elif block.get("type") == "tool_result":
                total += len(str(block.get("content") or ""))
            elif block.get("type") == "tool_use":
                total += len(json.dumps(block.get("input") or {}, ensure_ascii=False))
    return total


def _assemble(
    full: List[LogicalTurn],
    dialogue: List[LogicalTurn],
    cfg: WindowConfig,
    result_keep: int,
    message_chars: int,
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    for turn in dialogue:
        messages.extend(_render_dialogue(turn, cfg, message_chars))
    for turn in full:
        messages.extend(_render_full(turn, cfg, result_keep))
    return messages


def build_window(turns: Sequence[Any], cfg: WindowConfig) -> WindowResult:
    """Build the message window from STM rows, oldest first.

    Degrades in a fixed order when the budget will not hold everything,
    because the alternative — dropping whatever happens to be last — loses
    the most recent turn, which is the one that matters most:

    1. halve the kept length of large tool results;
    2. drop the oldest dialogue turn;
    3. demote the oldest full turn to dialogue;
    4. halve the dialogue utterance cap;
    5. keep the newest turn alone, with results at their floor.
    """
    if not cfg.enabled or not turns:
        return WindowResult()

    grouped = group_logical_turns(turns, cfg.turns)
    if not grouped:
        return WindowResult()

    full = grouped[-cfg.full_turns :] if cfg.full_turns else []
    dialogue = grouped[: len(grouped) - len(full)]
    result_keep = cfg.result_keep
    message_chars = cfg.dialogue_message_chars
    degraded: List[str] = []

    messages = _assemble(full, dialogue, cfg, result_keep, message_chars)
    while _measure(messages) > cfg.max_chars:
        if result_keep > MIN_RESULT_KEEP:
            result_keep = max(MIN_RESULT_KEEP, result_keep // 2)
            degraded.append("result_keep")
        elif dialogue:
            dialogue = dialogue[1:]
            degraded.append("drop_dialogue_turn")
        elif len(full) > 1:
            demoted, full = full[0], full[1:]
            dialogue = [demoted]
            degraded.append("demote_full_turn")
        elif message_chars > MIN_DIALOGUE_MESSAGE_CHARS:
            message_chars = max(MIN_DIALOGUE_MESSAGE_CHARS, message_chars // 2)
            degraded.append("message_chars")
        else:
            break
        messages = _assemble(full, dialogue, cfg, result_keep, message_chars)

    # Pair invariants last. Dropping a whole turn can leave the window
    # opening on a tool_result whose call is gone, and a turn that ended
    # mid-loop can hold a call whose result was never recorded — both are
    # rejected by the wire, so both are repaired rather than risked.
    messages = strip_leading_orphan_tool_results(messages)
    repair_dangling_tool_calls(messages)

    return WindowResult(
        messages=messages,
        turns=len(full) + len(dialogue),
        full=len(full),
        dialogue=len(dialogue),
        chars=_measure(messages),
        degraded=degraded,
    )
