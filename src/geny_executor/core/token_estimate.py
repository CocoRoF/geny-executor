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

The estimate is deliberately rough (≈4 chars/token, the heuristic Stage 2
has always used) — it *gates compaction*, it does not bill. Image blocks
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
_CHARS_PER_TOKEN = 4


def _estimate_text(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


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
