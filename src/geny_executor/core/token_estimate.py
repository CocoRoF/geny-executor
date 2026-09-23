"""Cheap, single-source token estimation for context-budget decisions.

The Stage 2 *proactive* compaction trigger and the Stage 4 *reactive*
token-budget guard must share ONE notion of "how big is the next API
call" — otherwise compaction (which shrinks ``state.messages``) cannot
relieve a guard that measures something else. Before 2.5.0 the guard
read ``state.token_usage`` (session/turn-cumulative usage, which
compaction never lowers) and compared it against the per-call context
window, so a long tool-loop turn could trip the guard with no way for
compaction to help. This module is the shared estimator both stages now
use against ``state.context_window_budget``.

The estimate *gates compaction*, it does not bill — but it has to be right
about Korean, and ``len(text) // 4`` was not. Measured against Claude's own
``usage.input_tokens`` on 26 Geny-shaped samples (2026-09-23, differential
calls on the production Claude Code account):

    len // 4          MAPE 58%   bias −58%   Korean subset MAPE 73%
    this estimator    MAPE 9.5%  held-out 10.5% (20 random half splits)

Korean runs about one token per character on Claude, English about one per
three; four-characters-per-token made a Korean conversation look a quarter of
its size, so compaction and the Stage-4 headroom guard both fired after the
request had already outgrown the window.

It counts **runs of one character class**, which is how BPE actually
behaves: Latin letters merge (≈2.9 chars/token), Hangul costs per syllable,
digits pair up, the first space rides on the next word. Coefficients are
fitted to Claude — Geny's primary route — and deliberately so for every
route: a conversation must survive failover to any hop, so the gate uses the
most expensive tokenizer it may meet (a GPT hop spends ~0.75 per syllable,
so there it compacts early rather than late, which is the safe direction). Image blocks
are counted at a flat per-image estimate rather than their base64 length:
a single 1568px screenshot is ~1.6k vision tokens but tens of thousands
of base64 characters, and counting the characters would trip compaction
on the first image.
"""

from __future__ import annotations

from typing import Any, List, Union

# Anthropic vision tokens land roughly here for a typical full-size image.
# A flat estimate beats counting base64 characters (which over-counts by
# ~50x and would trip compaction on a single screenshot).
_IMAGE_TOKEN_ESTIMATE = 1_600

#: Tokens per Hangul syllable (Claude). A GPT tokenizer spends ~0.75.
_HANGUL_PER_CHAR = 1.40
#: Characters one Latin-letter run spends per token (minimum one token).
_LATIN_CHARS_PER_TOKEN = 2.9
_DIGIT_CHARS_PER_TOKEN = 2.0
_PUNCT_PER_CHAR = 0.95
#: Only spaces after the first — the first rides on the following word.
_SPACE_PER_EXTRA = 0.25
_NEWLINE_PER_CHAR = 1.5
_CJK_PER_CHAR = 1.0
#: Emoji and other non-ASCII symbols are several bytes each.
_SYMBOL_PER_CHAR = 3.0
#: The fit under-reads by ~5.6% on average. A gate should err the other way.
_CALIBRATION = 1.06

_SP, _NL, _NUM, _LAT, _HAN, _CJK, _PUN, _SYM = range(8)


def _class_of(ch: str) -> int:
    code = ord(ch)
    if ch == " " or ch == "\t":
        return _SP
    if ch == "\n" or ch == "\r":
        return _NL
    if "0" <= ch <= "9":
        return _NUM
    if ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
        return _LAT
    if 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF or 0x3130 <= code <= 0x318F:
        return _HAN
    if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0x3040 <= code <= 0x30FF:
        return _CJK
    if code < 0x80:
        return _PUN
    if ch.isalpha():
        # Cyrillic, Greek, Arabic … merge like Latin.
        return _LAT
    return _SYM


def _estimate_text(text: str) -> int:
    """Tokens in *text*, counted by character-class runs. See module doc."""
    if not text:
        return 0
    total = 0.0
    index = 0
    length = len(text)
    while index < length:
        kind = _class_of(text[index])
        end = index + 1
        while end < length and _class_of(text[end]) == kind:
            end += 1
        total += _run_cost(kind, end - index)
        index = end
    return int(total * _CALIBRATION + 0.5)


def estimate_text_tokens(text: str) -> int:
    """Public wrapper so callers outside this module share one estimate."""
    return _estimate_text(str(text or ""))


def _run_cost(kind: int, run: int) -> float:
    if run <= 0:
        return 0.0
    if kind == _LAT:
        return max(1.0, run / _LATIN_CHARS_PER_TOKEN)
    if kind == _NUM:
        return max(1.0, run / _DIGIT_CHARS_PER_TOKEN)
    if kind == _HAN:
        return run * _HANGUL_PER_CHAR
    if kind == _CJK:
        return run * _CJK_PER_CHAR
    if kind == _SP:
        return (run - 1) * _SPACE_PER_EXTRA
    if kind == _NL:
        return run * _NEWLINE_PER_CHAR
    if kind == _PUN:
        return run * _PUNCT_PER_CHAR
    return run * _SYMBOL_PER_CHAR


def _chars_of_run_within(kind: int, run: int, room: float) -> int:
    """The longest leading part of a *kind* run that costs at most *room*."""
    if _run_cost(kind, run) <= room:
        return run
    if kind in (_LAT, _NUM):
        per = _LATIN_CHARS_PER_TOKEN if kind == _LAT else _DIGIT_CHARS_PER_TOKEN
        return 0 if room < 1.0 else min(run, int(room * per))
    if kind == _SP:
        return min(run, int(room / _SPACE_PER_EXTRA) + 1)
    per_char = {
        _HAN: _HANGUL_PER_CHAR,
        _CJK: _CJK_PER_CHAR,
        _NL: _NEWLINE_PER_CHAR,
        _PUN: _PUNCT_PER_CHAR,
    }.get(kind, _SYMBOL_PER_CHAR)
    return min(run, int(room / per_char))


def chars_within_tokens(text: str, tokens: int, *, from_end: bool = False) -> int:
    """How many characters of *text* — from its head, or from its tail — the
    estimator prices at no more than *tokens*.

    One linear pass over the same runs :func:`estimate_text_tokens` walks, so
    ``estimate_text_tokens(text[:n]) <= tokens`` holds for the returned ``n``
    by construction rather than by search. Searching (estimate the prefix,
    halve, repeat) was quadratic in practice: a replay window clipping twenty
    200 KB tool results spent seconds doing it.
    """
    s = str(text or "")
    if tokens <= 0 or not s:
        return 0
    # int(total × calibration + 0.5) <= tokens  ⇔  total < (tokens + 0.5) / calibration
    limit = (tokens + 0.5) / _CALIBRATION - 1e-9
    length = len(s)
    total = 0.0
    taken = 0
    index = length - 1 if from_end else 0
    step = -1 if from_end else 1
    while 0 <= index < length:
        kind = _class_of(s[index])
        end = index + step
        while 0 <= end < length and _class_of(s[end]) == kind:
            end += step
        run = abs(end - index)
        cost = _run_cost(kind, run)
        if total + cost <= limit:
            total += cost
            taken += run
            index = end
            continue
        return taken + _chars_of_run_within(kind, run, limit - total)
    return taken


def _estimate_block(block: Any) -> int:
    """Estimate one content block (text / image / tool_use / tool_result)."""
    if not isinstance(block, dict):
        return _estimate_text(str(block))
    btype = block.get("type")
    if btype == "image" or "source" in block:
        return _IMAGE_TOKEN_ESTIMATE
    total = 0
    for key in ("text", "content", "input", "thinking"):
        val = block.get(key)
        if isinstance(val, str):
            total += _estimate_text(val)
        elif isinstance(val, list):
            total += sum(_estimate_block(sub) for sub in val)
        elif val is not None:
            total += _estimate_text(str(val))
    return total or _estimate_text(str(block))


def _estimate_content(content: Union[str, List[Any], Any]) -> int:
    if isinstance(content, str):
        return _estimate_text(content)
    if isinstance(content, list):
        return sum(_estimate_block(b) for b in content)
    if content is None:
        return 0
    return _estimate_text(str(content))


def estimate_message_tokens(messages: List[Any]) -> int:
    """Rough token estimate of a message list (content only)."""
    return sum(
        _estimate_content(m.get("content", "")) for m in (messages or []) if isinstance(m, dict)
    )


#: ``state.shared`` slot for the per-turn estimate memo (TTFT program).
_ESTIMATE_MEMO_KEY = "_prompt_tokens_memo"


def _estimate_fingerprint(state: Any) -> tuple:
    """Cheap identity of the inputs the estimate depends on.

    The estimate is a full scan of system + every message + every tool
    schema — O(context). Stage 2's proactive check and Stage 4's budget
    guard both scan the SAME unchanged state within one iteration, so a
    fingerprint memo halves the per-iteration cost (2026-07-12 TTFT
    audit, finding B4). Mutations that matter all move the fingerprint:
    appends change the count, compaction replaces the head message
    (new object id), Stage 3 rebuilding system changes its length, and
    a tool-registry rebuild changes the tools count. In-place content
    edits with identical length CAN slip through — acceptable for an
    estimator that is documented as ±rough.
    """
    messages = getattr(state, "messages", []) or []
    system = getattr(state, "system", "") or ""
    tools = getattr(state, "tools", None) or []
    if isinstance(system, str):
        sys_len = len(system)
    else:
        sys_len = sum(len(str(b)) for b in system)
    return (
        len(messages),
        id(messages[0]) if messages else 0,
        id(messages[-1]) if messages else 0,
        sys_len,
        len(tools),
    )


def estimate_prompt_tokens(state: Any) -> int:
    """Rough INPUT-token estimate for the next API call.

    Sums the system prompt, the message list, and the tool schemas —
    everything that travels in the request and therefore counts against
    ``state.context_window_budget``. Used identically by the Stage 2
    compaction trigger and the Stage 4 token-budget guard so that
    compacting ``state.messages`` measurably lowers the same number the
    guard checks.

    Memoized per state via a fingerprint in ``state.shared`` (see
    ``_estimate_fingerprint``) so repeat callers within one iteration
    don't re-scan an unchanged context.
    """
    shared = getattr(state, "shared", None)
    fingerprint = _estimate_fingerprint(state)
    if isinstance(shared, dict):
        memo = shared.get(_ESTIMATE_MEMO_KEY)
        if isinstance(memo, tuple) and len(memo) == 2 and memo[0] == fingerprint:
            return memo[1]

    total = _estimate_content(getattr(state, "system", "") or "")
    total += estimate_message_tokens(getattr(state, "messages", []) or [])
    tools = getattr(state, "tools", None)
    if tools:
        total += sum(_estimate_text(str(tool)) for tool in tools)

    if isinstance(shared, dict):
        shared[_ESTIMATE_MEMO_KEY] = (fingerprint, total)
    return total
