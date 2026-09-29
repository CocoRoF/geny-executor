"""Stop a turn from spending itself on calls that cannot produce anything new.

Two loops, measured in the sibling runtime (XGEN, 2026-08..09) before any
threshold here was chosen:

* **The same failure, again.** A model passed ``max_results: "3"`` (a
  string), got ``'3' is not of type 'integer'``, and changed only the query
  — fifteen times. Every call re-sent the whole conversation.
* **The same call, the same answer.** ``pwd`` 21 times, a help tool 100
  times, one memory write 62 times. Over turns with 12+ tool calls the
  same-result rule caught 13 of 14 wasteful loops with one false positive in
  55 normal turns (a game button clicked four times — four is the most).

Geny had no guard at all, which is one of the two ways "the agent repeats
itself" happens; the other is not knowing the work was already done, which
the turn replay addresses (``memory.short_term_window``).

How a failure is keyed depends on its kind:

* **Input errors** (``ERROR invalid_input``) key on (tool, normalised
  message) — NOT the arguments. The loop above changed its arguments every
  time; the message staying the same is what says the cause was not fixed.
* **Other failures** ("not found", an upstream 5xx) can be legitimate for
  different items with the same message, so they are counted twice: the same
  arguments against the tight thresholds, any arguments against looser ones
  (an auth failure fails whatever you pass). Any tool succeeding clears
  them — "edit, rebuild, fail differently" is how coding works.

At the warn threshold the result carries a note telling the model to stop;
at the block threshold the tool is not executed for the rest of the turn.
Identical calls with identical results are answered from the previous result
without executing — which also stops a repeated write or send.

State lives in ``state.shared``: per turn (a fresh state per turn), shared by
every iteration of it.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "ANY_INPUT_BLOCK_AT",
    "ANY_INPUT_WARN_AT",
    "BLOCK_AT",
    "REFUSED_KEY",
    "SAME_RESULT_SKIP_AT",
    "SAME_RESULT_WARN_AT",
    "WARN_AT",
    "blocked_result",
    "guard_calls",
    "merge_in_order",
    "normalize_error",
    "observe",
    "observe_same",
    "report",
    "skip_identical",
]

WARN_AT = 3
BLOCK_AT = 4
ANY_INPUT_WARN_AT = 5
ANY_INPUT_BLOCK_AT = 8
SAME_RESULT_WARN_AT = 4
#: The Nth identical call is answered from the previous result. XGEN moved
#: this from 8 to 5 after one conversation spent its whole budget on four
#: identical graph reads before the guard engaged.
SAME_RESULT_SKIP_AT = 5

_COUNTS_KEY = "tool.repeat_error_counts"
_BLOCKED_KEY = "tool.repeat_error_blocked"
_SAME_KEY = "tool.same_result_counts"
#: Calls refused this turn (blocked or answered without executing). Stage 16
#: ends the turn when these pile up — see ``s16_loop.repeat_stop``.
REFUSED_KEY = "tool.refused_calls"

_BLOCKED_MARK = "ERROR repeated_failure_blocked"
_SKIPPED_MARK = "[identical call — not executed]"

#: Parts of an error that change per call and must not split one cause into
#: many keys: ids and hashes, long numbers (times, ports, lengths), spacing.
_VOLATILE = [
    (re.compile(r"\b[0-9a-f]{8,}\b", re.I), "#"),
    (re.compile(r"\b\d{2,}\b"), "#"),
    (re.compile(r"\s+"), " "),
]
#: The JSON body after a structured ``ERROR <code>: <message>`` header —
#: request ids, paths. Stripped only when the header is there: cutting at the
#: first brace of a build log would cut the actual error.
_STRUCTURED_BODY = re.compile(r"\{.*", re.S)
_KEY_HEAD = 80
_KEY_TAIL = 200


def _text(result: Dict[str, Any]) -> Optional[str]:
    content = result.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(b.get("text", ""))
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        joined = "\n".join(p for p in parts if p)
        return joined or None
    return None


def normalize_error(text: str) -> str:
    """The error as a key: head AND tail.

    Command output starts the same every time (the banner, the echoed
    command) and ends with the actual error. Keying on the head alone made
    ``npm run build`` failing for two different reasons count as one
    failure, and blocked the fix-and-rebuild loop that is supposed to happen.
    """
    out = str(text or "").strip()
    if out.startswith("ERROR "):
        out = _STRUCTURED_BODY.sub("", out)
    for pattern, repl in _VOLATILE:
        out = pattern.sub(repl, out)
    out = out.strip()
    if len(out) > _KEY_HEAD + _KEY_TAIL:
        out = f"{out[:_KEY_HEAD]}…{out[-_KEY_TAIL:]}"
    return out


def _is_input_error(text: str) -> bool:
    return str(text or "").lstrip().startswith("ERROR invalid_input")


def _input_sig(tool_input: Any) -> str:
    try:
        raw = json.dumps(tool_input, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        raw = str(tool_input)
    return hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:12]


def blocked_result(tool_call: Dict[str, Any], shared: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The result to return instead of executing, if the tool is blocked."""
    name = str(tool_call.get("tool_name") or "")
    reason = (shared.get(_BLOCKED_KEY) or {}).get(name)
    if not reason:
        return None
    return {
        "type": "tool_result",
        "tool_use_id": tool_call.get("tool_use_id", ""),
        "is_error": True,
        "content": (
            f"{_BLOCKED_MARK}: '{name}' failed the same way {BLOCK_AT}+ times this turn "
            f"and will not run again in it.\nLast error: {reason}\n"
            "Do not call it again. Explain the cause (argument type, a required value, "
            "a path) to the user, and use another way if there is one."
        ),
    }


def observe(
    tool_calls: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    shared: Dict[str, Any],
) -> List[Tuple[str, int]]:
    """Count failures, note the warn threshold on the result, arm the block.

    ``results`` is edited in place. Returns ``(tool, count)`` for every tool
    that crossed the warn threshold — for the event.
    """
    counts: Dict[str, int] = shared.setdefault(_COUNTS_KEY, {})
    blocked: Dict[str, str] = shared.setdefault(_BLOCKED_KEY, {})
    calls = {str(tc.get("tool_use_id") or ""): tc for tc in tool_calls}
    flagged: List[Tuple[str, int]] = []

    def _bump(key: str) -> int:
        counts[key] = counts.get(key, 0) + 1
        return counts[key]

    for result in results:
        tc = calls.get(str(result.get("tool_use_id") or "")) or {}
        name = str(tc.get("tool_name") or "")
        if not name:
            continue
        if not result.get("is_error"):
            # A success ends this tool's failures, and any tool's success ends
            # the execution failures: editing a file and rebuilding is not
            # repeating an unfixed cause. Input errors stay — doing something
            # else did not make the arguments right.
            for key in [k for k in counts if k.startswith(f"{name}\x1f") or "\x1fR\x1f" in k]:
                counts.pop(key, None)
            continue
        text = _text(result)
        if not text or text.startswith(_BLOCKED_MARK):
            continue
        err = normalize_error(text)
        if _is_input_error(text):
            n = _bump(f"{name}\x1fI\x1f{err}")
            warn, block, block_at = n >= WARN_AT, n >= BLOCK_AT, BLOCK_AT
        else:
            same = _bump(f"{name}\x1fR\x1f{_input_sig(tc.get('tool_input'))}\x1f{err}")
            any_ = _bump(f"{name}\x1fR\x1f*\x1f{err}")
            warn = same >= WARN_AT or any_ >= ANY_INPUT_WARN_AT
            block = same >= BLOCK_AT or any_ >= ANY_INPUT_BLOCK_AT
            n = max(same, any_)
            block_at = BLOCK_AT if same >= WARN_AT else ANY_INPUT_BLOCK_AT
        if block:
            blocked[name] = err
        if warn:
            flagged.append((name, n))
            if isinstance(result.get("content"), str):
                result["content"] = (
                    f"{result['content']}\n\n[failed the same way {n} times] '{name}' has "
                    f"failed with this error {n} times. Do not call it the same way again: fix "
                    "the arguments the error names, or tell the user what is blocking you. "
                    f"From the {block_at}th time it will not run in this turn."
                )
    return flagged


def _result_sig(result: Dict[str, Any]) -> Optional[str]:
    text = _text(result)
    if text is None:
        return None
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:16]


def _call_key(tool_call: Dict[str, Any]) -> str:
    return f"{tool_call.get('tool_name') or ''}\x1f{_input_sig(tool_call.get('tool_input'))}"


def skip_identical(tool_call: Dict[str, Any], shared: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The previous result, if this exact call already returned it
    ``SAME_RESULT_SKIP_AT - 1`` times. Executing again would tell the model
    nothing it does not have — and would repeat any side effect."""
    entry = (shared.get(_SAME_KEY) or {}).get(_call_key(tool_call))
    if not entry or entry.get("n", 0) < SAME_RESULT_SKIP_AT - 1:
        return None
    name = str(tool_call.get("tool_name") or "")
    return {
        "type": "tool_result",
        "tool_use_id": tool_call.get("tool_use_id", ""),
        "content": (
            f"{entry.get('last', '')}\n\n{_SKIPPED_MARK} '{name}' was called with this exact "
            f"input {entry['n']} times and returned the result above every time. Calling it "
            "again gives nothing new. Move on with this result, or tell the user what is "
            "blocking you."
        ),
    }


def observe_same(
    tool_calls: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    shared: Dict[str, Any],
) -> List[Tuple[str, int]]:
    """Count identical call → identical result; note it from the warn threshold."""
    same: Dict[str, Dict[str, Any]] = shared.setdefault(_SAME_KEY, {})
    calls = {str(tc.get("tool_use_id") or ""): tc for tc in tool_calls}
    flagged: List[Tuple[str, int]] = []
    for result in results:
        tc = calls.get(str(result.get("tool_use_id") or ""))
        if not tc or result.get("is_error"):
            continue  # repeated failures are observe()'s business
        content = result.get("content")
        if isinstance(content, str) and _SKIPPED_MARK in content:
            continue  # already answered without executing — not counted again
        sig = _result_sig(result)
        if sig is None:
            continue
        key = _call_key(tc)
        entry = same.get(key)
        if entry and entry.get("sig") == sig:
            entry["n"] += 1
        else:
            entry = same[key] = {"sig": sig, "n": 1, "last": ""}
        if isinstance(content, str):
            entry["last"] = content[:4000]
        n = entry["n"]
        if n >= SAME_RESULT_WARN_AT:
            name = str(tc.get("tool_name") or "")
            flagged.append((name, n))
            if isinstance(content, str):
                result["content"] = (
                    f"{content}\n\n[same call, same result, {n} times] '{name}' returned this "
                    f"for this exact input {n} times. Nothing new will come from repeating it: "
                    "change the approach, or tell the user what is blocking you. From the "
                    f"{SAME_RESULT_SKIP_AT}th time it is answered with this result instead of "
                    "running."
                )
    return flagged


def guard_calls(
    tool_calls: List[Dict[str, Any]], shared: Dict[str, Any]
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], List[str], List[str]]:
    """Split a round of calls into answered-without-running and runnable.

    Returns ``(precomputed_by_id, runnable, blocked_names, skipped_names)``.
    """
    precomputed: Dict[str, Dict[str, Any]] = {}
    runnable: List[Dict[str, Any]] = []
    blocked_names: List[str] = []
    skipped_names: List[str] = []
    from geny_executor.stages.s10_tool import denial_guard

    for tc in tool_calls:
        # A refusal is answered first: the first "no" is the answer, and
        # asking again is how the wrong button gets clicked.
        result = denial_guard.refused_result(tc, shared)
        if result is None:
            result = blocked_result(tc, shared)
        if result is not None:
            blocked_names.append(str(tc.get("tool_name") or ""))
        else:
            result = skip_identical(tc, shared)
            if result is not None:
                skipped_names.append(str(tc.get("tool_name") or ""))
        if result is not None:
            precomputed[str(tc.get("tool_use_id") or "")] = result
        else:
            runnable.append(tc)
    if precomputed:
        shared[REFUSED_KEY] = int(shared.get(REFUSED_KEY) or 0) + len(precomputed)
    return precomputed, runnable, blocked_names, skipped_names


def merge_in_order(
    tool_calls: List[Dict[str, Any]],
    precomputed: Dict[str, Dict[str, Any]],
    executed: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Results in the order the calls were made, answered and executed alike."""
    if not precomputed:
        return executed
    by_id = {str(r.get("tool_use_id") or ""): r for r in executed}
    out: List[Dict[str, Any]] = []
    for tc in tool_calls:
        key = str(tc.get("tool_use_id") or "")
        result = precomputed.get(key) or by_id.get(key)
        if result is not None:
            out.append(result)
    return out


def report(
    state: Any,
    tool_calls: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    blocked_names: List[str],
    skipped_names: List[str],
) -> None:
    """Observe a finished round and emit what the guard did."""
    from geny_executor.stages.s10_tool import denial_guard

    shared = state.shared
    if blocked_names:
        state.add_event("tool.repeat_blocked", {"tools": sorted(set(blocked_names))})
    newly_denied = denial_guard.observe(tool_calls, results, shared)
    if newly_denied:
        state.add_event("tool.user_denied", {"tools": sorted(set(newly_denied))})
    flagged = observe(tool_calls, results, shared)
    if flagged:
        state.add_event(
            "tool.repeat_failure", {"tools": [{"name": n, "count": c} for n, c in flagged]}
        )
    same = observe_same(tool_calls, results, shared)
    if same or skipped_names:
        state.add_event(
            "tool.same_result",
            {
                "tools": [{"name": n, "count": c} for n, c in same],
                "skipped": sorted(set(skipped_names)),
            },
        )
