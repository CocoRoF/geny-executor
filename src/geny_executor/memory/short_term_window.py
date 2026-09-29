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

Five turns within the budget; the newest one always. The budget is a share
of the route's effective input window, in tokens (:func:`window_token_budget`)
— a 32k local model and a 1M frontier model get windows that fit them, and
the same budget means the same thing in Korean and in English. Under
pressure the window sheds bulk before structure, oldest first, and never the
newest turn (:func:`build_window`).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from geny_executor.core.token_estimate import chars_within_tokens, estimate_text_tokens
from geny_executor.core.message_repair import (
    normalize_messages_for_request,
    strip_leading_orphan_tool_results,
)

logger = logging.getLogger(__name__)

__all__ = [
    "WindowConfig",
    "WindowResult",
    "LogicalTurn",
    "build_window",
    "group_logical_turns",
    "is_silent_turn",
    "window_token_budget",
]

#: How many recent turns keep their tool blocks verbatim.
DEFAULT_FULL_TURNS = 2
#: How many turns behind those keep conversation only.
DEFAULT_DIALOGUE_TURNS = 3

#: Share of the route's EFFECTIVE input window the replay may occupy.
#:
#: Hermes protects a tail of ``threshold × summary_target_ratio`` — 0.75 ×
#: 0.20 of the window for anything under 512K, i.e. 15% — and that is the
#: same job: the part of the recent past kept verbatim while everything older
#: is summarised. A share rather than a constant, because no constant is
#: right for both a 32k local model (where five turns of tool output do not
#: fit at all) and a 1M frontier one (where 10k tokens throws away the very
#: evidence the window exists to keep).
DEFAULT_WINDOW_RATIO = 0.15
#: Used when no hop in the route could say how large it is: 15% of a 200k
#: window with a 32k answer reserved. The common case, not a guess at a
#: small one.
DEFAULT_MAX_TOKENS = 25_000
#: Never size the window below this. Under it the replay is fragments, and
#: the newest turn is kept regardless (see :func:`build_window`).
MIN_MAX_TOKENS = 2_000

#: One tool result may take at most this share of the window before its tail
#: goes: one 200 KB file read must not evict the four turns around it.
RESULT_SHARE = 0.25
#: ...and never more than this, whatever the window. Claude Code caps a
#: single tool output near 25k tokens for the same reason — past that the
#: result is a document, and a document belongs in a file, not in replay.
MAX_RESULT_TOKENS = 25_000
#: Budget pressure never trims a kept result below this — enough for the
#: head of a listing or the whole of an error message.
MIN_RESULT_TOKENS = 150
#: One utterance in a conversation-only turn: its share of the window and its
#: absolute cap. The conversation is the point of those turns, so the cap is
#: generous; it exists for the pasted-document turn, not the ordinary one.
UTTERANCE_SHARE = 0.125
MAX_UTTERANCE_TOKENS = 4_000
MIN_UTTERANCE_TOKENS = 150
#: Targets named on the ``[used tools: …]`` line, per tool.
MAX_TOOL_TARGETS = 3
#: ``[SILENT]`` is how an agent declines to speak (Geny; Hermes' cron uses
#: the same marker to suppress delivery).
DEFAULT_SILENT_MARKERS: Tuple[str, ...] = ("[SILENT]",)
#: Wire framing per message (role, block envelopes). Small, but a window of
#: forty short messages is not free.
_MESSAGE_OVERHEAD_TOKENS = 4

_TRIM_NOTE = "…[+{n} chars trimmed]"
_HEAD_TRIM_NOTE = "[{n} chars trimmed]…"


def window_token_budget(
    context_window_budget: Optional[int],
    *,
    reserved_output: Optional[int] = None,
    ratio: float = DEFAULT_WINDOW_RATIO,
) -> int:
    """Tokens the replay may spend, given what the route can hold.

    ``ratio × (window − reserved_output)`` — the answer has to fit in the
    same window, so a 32k model asked for 8k of output has 24k to give, not
    32k. ``None`` (no hop could say) falls back to :data:`DEFAULT_MAX_TOKENS`.

    Measured in tokens, not characters: the same 40,000 characters are about
    10k tokens of English and about 40k tokens of Korean, so a character
    budget meant four different things in four languages.
    """
    if not context_window_budget or context_window_budget <= 0:
        return DEFAULT_MAX_TOKENS
    effective = int(context_window_budget) - max(0, int(reserved_output or 0))
    if effective <= 0:
        effective = int(context_window_budget)
    return max(MIN_MAX_TOKENS, int(effective * ratio))


@dataclass
class WindowConfig:
    full_turns: int = DEFAULT_FULL_TURNS
    dialogue_turns: int = DEFAULT_DIALOGUE_TURNS
    #: The whole replay, in tokens. See :func:`window_token_budget`.
    max_tokens: int = DEFAULT_MAX_TOKENS
    #: Per-result and per-utterance caps. ``None`` derives them from
    #: ``max_tokens`` so one knob scales the whole window.
    result_tokens: Optional[int] = None
    utterance_tokens: Optional[int] = None
    used_tools_line: bool = True
    #: Answers that mean "said nothing". A turn whose agent side is only
    #: these (or nothing) and that ran no tools is not history — it does
    #: not take one of the slots. An agent woken every few minutes by an
    #: idle or screen trigger answers most of them with silence; without
    #: this, five of those push the last real exchange out of the window.
    silent_markers: Tuple[str, ...] = DEFAULT_SILENT_MARKERS

    def result_cap(self) -> int:
        if self.result_tokens is not None:
            return max(MIN_RESULT_TOKENS, int(self.result_tokens))
        return max(MIN_RESULT_TOKENS, min(MAX_RESULT_TOKENS, int(self.max_tokens * RESULT_SHARE)))

    def utterance_cap(self) -> int:
        if self.utterance_tokens is not None:
            return max(MIN_UTTERANCE_TOKENS, int(self.utterance_tokens))
        return max(
            MIN_UTTERANCE_TOKENS,
            min(MAX_UTTERANCE_TOKENS, int(self.max_tokens * UTTERANCE_SHARE)),
        )

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
    #: Estimated tokens the replay costs (same estimator the guards use).
    tokens: int = 0
    #: What it was allowed to cost.
    budget: int = 0
    #: Degradation steps taken, in order, for the event payload.
    degraded: List[str] = field(default_factory=list)

    def as_metadata(self) -> Dict[str, Any]:
        return {
            "turns": self.turns,
            "full": self.full,
            "dialogue": self.dialogue,
            "tokens": self.tokens,
            "budget": self.budget,
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


def is_silent_turn(turn: LogicalTurn, markers: Sequence[str] = DEFAULT_SILENT_MARKERS) -> bool:
    """Nothing was said and nothing was done.

    Every assistant text is empty or a silence marker, and no tool ran. A
    turn that called a tool is never silent, whatever it said: that call is
    exactly the evidence the window exists to keep.
    """
    if turn.tool_calls():
        return False
    upper = [m.upper() for m in markers if m]
    answered = False
    for message in turn.messages:
        if str(getattr(message, "role", "") or "") != "assistant":
            continue
        answered = True
        text = _text_of(message)
        if not text:
            continue
        if not any(text.upper().startswith(m) and not text[len(m) :].strip() for m in upper):
            return False
    # A turn the agent never answered is not silence: the user still said
    # it. (Its reply may simply not have been recorded — a turn that failed
    # before 2.75 left no answer behind.) Dropping it dropped the user's
    # words from every later turn.
    return answered


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


class _Sizer:
    """Token estimates and clips, memoised for ONE build.

    The degradation loop re-renders the window after every step, and each
    render re-clips every result. Without the memo, twenty 200 KB results
    took 23 seconds — a pathological turn, but a turn the loop must not be
    able to stall on. Per build rather than module-level, so the cache never
    outlives the strings it holds.
    """

    def __init__(self) -> None:
        self._tokens: Dict[str, int] = {}
        self._clips: Dict[Tuple[str, int, bool], str] = {}

    def tokens(self, text: str) -> int:
        cached = self._tokens.get(text)
        if cached is None:
            cached = estimate_text_tokens(text)
            self._tokens[text] = cached
        return cached

    def clip(self, text: str, tokens: int, *, keep_tail: bool = False) -> str:
        """*text* cut to about *tokens*, with a note naming what went.

        ``keep_tail`` keeps the END: an assistant's turn ends in its
        conclusion, and "I'll check the config first" is the wrong half to
        keep when the question later is whether the work got done.
        """
        s = str(text)
        key = (s, tokens, keep_tail)
        cached = self._clips.get(key)
        if cached is not None:
            return cached
        # One pass that stops at the cap: whether the text fits and where to
        # cut are the same question, and asking it by estimating the whole
        # 200 KB result first is what the pass exists to avoid.
        keep = len(s) if tokens <= 0 else chars_within_tokens(s, tokens, from_end=keep_tail)
        if keep >= len(s):
            out = s
        else:
            dropped = len(s) - keep
            if keep_tail:
                out = _HEAD_TRIM_NOTE.format(n=dropped) + s[len(s) - keep :]
            else:
                out = s[:keep] + _TRIM_NOTE.format(n=dropped)
        self._clips[key] = out
        return out


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


def _render_full(turn: LogicalTurn, result_cap: int, sizer: _Sizer) -> List[Dict[str, Any]]:
    """A turn with its tools intact; only a result over *result_cap* tokens
    loses its tail — and keeps its block, which is what says the call ran."""
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
                kept: Dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": block.get("tool_use_id"),
                    "content": sizer.clip(_result_text(block), result_cap),
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


def _has_image(message: Any) -> bool:
    return any(b.get("type") == "image" for b in _blocks(getattr(message, "content", "")))


def _render_dialogue(
    turn: LogicalTurn, cfg: WindowConfig, utterance_cap: int, sizer: _Sizer
) -> List[Dict[str, Any]]:
    """A turn reduced to what was said, plus one line about what was done.

    The agent's side is EVERYTHING it said across the turn, in order — not
    just its last message. A multi-step turn narrates as it goes ("the
    config is missing the key, adding it" … "done"), and keeping only the
    last line is how "done" survives without what was done.
    """
    out: List[Dict[str, Any]] = []
    instruction = ""
    said: List[str] = []
    for message in turn.messages:
        role = str(getattr(message, "role", "") or "user")
        content = getattr(message, "content", "")
        text = _text_of(message)
        if role == "user" and not _is_tool_result_only(content):
            if not instruction:
                # An image-only instruction still happened; dropping it would
                # leave an answer to nothing.
                instruction = text or ("[image]" if _has_image(message) else "")
        elif role == "assistant" and text:
            said.append(text)

    if instruction:
        out.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": sizer.clip(instruction, utterance_cap)}],
            }
        )
    tail = sizer.clip("\n\n".join(said), utterance_cap, keep_tail=True) if said else ""
    if cfg.used_tools_line:
        line = _used_tools_line(turn)
        if line:
            tail = f"{tail}\n{line}" if tail else line
    if tail:
        out.append({"role": "assistant", "content": [{"type": "text", "text": tail}]})
    return out


def _measure(messages: Iterable[Dict[str, Any]], sizer: Optional[_Sizer] = None) -> int:
    """Estimated tokens, with the estimator the pipeline's guards use — so
    the window and the compaction trigger agree on what a message costs."""
    count = (sizer or _Sizer()).tokens
    total = 0
    for message in messages:
        total += _MESSAGE_OVERHEAD_TOKENS
        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                total += count(str(block.get("text") or ""))
            elif btype == "tool_result":
                total += count(str(block.get("content") or ""))
            elif btype == "tool_use":
                total += count(str(block.get("name") or ""))
                total += count(json.dumps(block.get("input") or {}, ensure_ascii=False))
    return total


def _assemble(
    full: List[LogicalTurn],
    dialogue: List[LogicalTurn],
    cfg: WindowConfig,
    result_caps: List[int],
    utterance_cap: int,
    sizer: _Sizer,
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    for turn in dialogue:
        messages.extend(_render_dialogue(turn, cfg, utterance_cap, sizer))
    for turn, cap in zip(full, result_caps):
        messages.extend(_render_full(turn, cap, sizer))
    return messages


def build_window(turns: Sequence[Any], cfg: WindowConfig) -> WindowResult:
    """Build the message window from STM rows, oldest first.

    "Five turns within the budget; the newest one always." When everything
    does not fit, bulk goes before structure — every step below keeps every
    turn it can and every ``tool_use`` block, because *that a call happened*
    is what stops the agent redoing it, and the result's tail rarely is:

    1. shrink tool results, OLDEST full turn first (Anthropic's
       ``clear_tool_uses`` clears oldest results first for the same reason);
    2. shrink the conversation-only turns' utterances;
    3. drop the oldest conversation-only turn;
    4. demote the older full turn to conversation-only;
    5. stop. The newest turn stays, at its floors, even over budget — a
       window that drops the turn the user is replying to has failed at the
       one thing it is for.
    """
    if not cfg.enabled or not turns:
        return WindowResult()

    grouped = [
        turn
        for turn in group_logical_turns(turns, len(turns))
        if not is_silent_turn(turn, cfg.silent_markers)
    ][-cfg.turns :]
    if not grouped:
        return WindowResult()

    full = grouped[-cfg.full_turns :] if cfg.full_turns else []
    dialogue = grouped[: len(grouped) - len(full)]
    result_caps = [cfg.result_cap() for _ in full]
    utterance_cap = cfg.utterance_cap()
    degraded: List[str] = []
    sizer = _Sizer()

    messages = _assemble(full, dialogue, cfg, result_caps, utterance_cap, sizer)
    while _measure(messages, sizer) > cfg.max_tokens:
        shrinkable = [i for i, cap in enumerate(result_caps) if cap > MIN_RESULT_TOKENS]
        if shrinkable:
            i = shrinkable[0]
            result_caps[i] = max(MIN_RESULT_TOKENS, result_caps[i] // 2)
            degraded.append("result_tokens")
        elif dialogue and utterance_cap > MIN_UTTERANCE_TOKENS:
            utterance_cap = max(MIN_UTTERANCE_TOKENS, utterance_cap // 2)
            degraded.append("utterance_tokens")
        elif dialogue:
            dialogue = dialogue[1:]
            degraded.append("drop_dialogue_turn")
        elif len(full) > 1:
            dialogue = [full[0]]
            full, result_caps = full[1:], result_caps[1:]
            degraded.append("demote_full_turn")
        else:
            degraded.append("over_budget_newest_kept")
            break
        messages = _assemble(full, dialogue, cfg, result_caps, utterance_cap, sizer)

    # Pair invariants last. Dropping a whole turn can leave the window
    # opening on a tool_result whose call is gone, and a turn that ended
    # mid-loop can hold a call whose result was never recorded — both are
    # rejected by the wire, so both are repaired rather than risked.
    messages = strip_leading_orphan_tool_results(messages)
    messages = normalize_messages_for_request(messages)
    _close_unanswered(messages)

    return WindowResult(
        messages=messages,
        turns=len(full) + len(dialogue),
        full=len(full),
        dialogue=len(dialogue),
        tokens=_measure(messages, sizer),
        budget=cfg.max_tokens,
        degraded=degraded,
    )


#: Shortest an utterance is cut to when the text rendering is over budget —
#: below this a line reads as a fragment, and a fragment reads as a claim.
MIN_TEXT_UTTERANCE_CHARS = 160


def render_turns_as_text(
    rows: Sequence[Any],
    *,
    turns: int,
    max_chars: int,
    silent_markers: Sequence[str] = DEFAULT_SILENT_MARKERS,
) -> Tuple[str, int]:
    """The last *turns* logical turns as plain text, for when they cannot be
    messages.

    The fallback for a turn with no replay (no memory provider for it, the
    replay switched off, short-term memory unreadable by the replay): the
    same grouping and the same rules as the window — a turn is an
    instruction and everything that followed it, silent turns take no room,
    what the agent did is one ``[used tools: …]`` line — rendered as
    ``[user] …`` / ``[assistant] …``. It replaced a renderer that counted
    ROWS (a tool call is four), kept only text blocks (so the tool rows took
    a slot and showed nothing) and cut from the tail (so the first
    instruction was the first thing lost).

    Over budget, utterances get shorter; turns are not dropped. Returns the
    text and the number of turns in it.
    """
    if turns <= 0 or max_chars <= 0 or not rows:
        return "", 0
    grouped = [
        t for t in group_logical_turns(rows, len(rows)) if not is_silent_turn(t, silent_markers)
    ][-turns:]
    if not grouped:
        return "", 0

    def _render(cap: int) -> str:
        lines: List[str] = []
        for turn in grouped:
            instruction = ""
            said: List[str] = []
            for message in turn.messages:
                role = str(getattr(message, "role", "") or "user")
                content = getattr(message, "content", "")
                text = _text_of(message)
                if role == "user" and not _is_tool_result_only(content):
                    if not instruction:
                        instruction = text or ("[image]" if _has_image(message) else "")
                elif role == "assistant" and text:
                    said.append(text)
            if instruction:
                lines.append(f"[user] {_clip_chars(instruction, cap)}")
            answer = _clip_chars(" ".join(said), cap, keep_tail=True) if said else ""
            used = _used_tools_line(turn)
            tail = " ".join(p for p in (answer, used) if p)
            if tail:
                lines.append(f"[assistant] {tail}")
        return "\n".join(lines)

    cap = max(MIN_TEXT_UTTERANCE_CHARS, max_chars // max(1, 2 * len(grouped)))
    body = _render(cap)
    while len(body) > max_chars and cap > MIN_TEXT_UTTERANCE_CHARS:
        cap = max(MIN_TEXT_UTTERANCE_CHARS, cap // 2)
        body = _render(cap)
    return body, len(grouped)


def _clip_chars(text: str, limit: int, *, keep_tail: bool = False) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return ("…" + text[-(limit - 1) :]) if keep_tail else (text[: limit - 1] + "…")


#: What stands in for a reply that was never recorded.
UNANSWERED_NOTE = "[No reply to this was recorded — that turn did not finish.]"


def _is_instruction(message: Dict[str, Any]) -> bool:
    return message.get("role") == "user" and not _is_tool_result_only(message.get("content"))


def _close_unanswered(messages: List[Dict[str, Any]]) -> None:
    """An instruction followed by another instruction, or ending the window,
    gets a one-line assistant reply saying none was recorded.

    Two user turns in a row read as one request; a window that ends on the
    user reads as a question still waiting — and the current instruction
    comes right after it. The note keeps the roles alternating and says
    what actually happened.
    """
    i = 0
    while i < len(messages):
        if _is_instruction(messages[i]):
            nxt = messages[i + 1] if i + 1 < len(messages) else None
            if nxt is None or _is_instruction(nxt):
                messages.insert(
                    i + 1,
                    {"role": "assistant", "content": [{"type": "text", "text": UNANSWERED_NOTE}]},
                )
                i += 1
        i += 1
