"""Tool-call/result pairing repairs for ``state.messages`` (audit D4).

The Anthropic Messages API (and OpenAI's tool protocol) require that
every assistant ``tool_use`` block is answered by a following
``tool_result`` and, conversely, that a ``tool_result`` names a
``tool_use`` that precedes it. A turn that dies between appending the
assistant ``tool_use`` and appending the tool results — a user stop, a
tool-access denial, a crash inside the dispatch ``gather`` — leaves
``state.messages`` ending in a dangling ``tool_use``, and every
subsequent request 400s ("tool_use ids without tool_result"). Compaction
can independently orphan the OTHER side, slicing a kept window that
begins with a ``tool_result`` whose ``tool_use`` was dropped.

Both repairs are pure list surgery — no message content is invented
beyond a synthetic "[interrupted]" result — and are backend-agnostic
(the invariant they enforce is shared by every tool-using provider).
"""

from __future__ import annotations

from typing import Any, Dict, List


def _tool_use_ids(msg: Dict[str, Any]) -> List[str]:
    content = msg.get("content")
    if not isinstance(content, list):
        return []
    return [
        b["id"]
        for b in content
        if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id")
    ]


def _tool_result_ids(msg: Dict[str, Any]) -> set:
    content = msg.get("content")
    if not isinstance(content, list):
        return set()
    return {
        b.get("tool_use_id")
        for b in content
        if isinstance(b, dict) and b.get("type") == "tool_result"
    }


def _is_tool_result_only(msg: Dict[str, Any]) -> bool:
    """True when the message is a user turn made ENTIRELY of tool_result
    blocks — the shape that is orphaned when its tool_use is dropped."""
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if not isinstance(content, list) or not content:
        return False
    return all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)


def repair_dangling_tool_calls(messages: List[Dict[str, Any]]) -> int:
    """Append synthetic error tool_results for any unanswered tool_use.

    Scans for the last assistant message that issued ``tool_use`` blocks;
    if any of its ids are not answered by a following ``tool_result``,
    inserts a user ``tool_result`` (``is_error``, ``"[interrupted]"``)
    for each unmatched id immediately after that assistant message — the
    exact position the API requires. Returns the number of results
    injected (0 when history is already consistent). Mutates ``messages``
    in place.
    """
    if not messages:
        return 0

    last_use_idx = None
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, dict) and msg.get("role") == "assistant" and _tool_use_ids(msg):
            last_use_idx = i
            break
    if last_use_idx is None:
        return 0

    use_ids = _tool_use_ids(messages[last_use_idx])
    answered: set = set()
    for j in range(last_use_idx + 1, len(messages)):
        if isinstance(messages[j], dict):
            answered |= _tool_result_ids(messages[j])
    missing = [uid for uid in use_ids if uid not in answered]
    if not missing:
        return 0

    synthetic = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": uid,
                "content": "[interrupted — the previous turn ended before this tool finished]",
                "is_error": True,
            }
            for uid in missing
        ],
    }
    messages.insert(last_use_idx + 1, synthetic)
    return len(missing)


def strip_leading_orphan_tool_results(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop leading messages that are orphaned ``tool_result`` turns.

    A compaction that keeps ``messages[-keep:]`` can begin the window
    with a user ``tool_result`` message whose ``tool_use`` was in the
    dropped prefix — invalid. Returns a copy with such leading orphans
    removed so the kept window always opens on a real turn. Only drops a
    leading tool_result message when its ids are NOT answered by a
    ``tool_use`` still present earlier in the (already-sliced) window,
    which for a leading message is always — so it simply skips any
    tool_result-only message at the head.
    """
    i = 0
    while i < len(messages) and _is_tool_result_only(messages[i]):
        i += 1
    return messages[i:] if i else messages


_INTERRUPTED_RESULT = "[interrupted — the previous turn ended before this tool finished]"


def _synthetic_result(tool_use_id: str) -> Dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": _INTERRUPTED_RESULT,
        "is_error": True,
    }


def normalize_messages_for_request(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """A provider-safe copy of a whole history: every call answered, every
    answer to a call that is there.

    :func:`repair_dangling_tool_calls` fixes the NEWEST unanswered call, in
    place, at the start of a turn. A history can hold more than that — a
    replayed turn that was stopped mid-tool, a compaction that sliced a
    pair, a checkpoint from before a crash — and any one of them 400s the
    request (Anthropic and OpenAI both refuse an unanswered call and an
    answer without a call). This runs on the copy that goes to the wire,
    so the stored history is never rewritten by a repair.

    * a ``tool_result`` naming no earlier ``tool_use`` is removed (the rest
      of its message stays; a message left empty is dropped);
    * a ``tool_use`` with no later ``tool_result`` gets a synthetic error
      result — merged into the tool-result message that follows the call
      when there is one, so the result stays in the turn the API expects
      it in, else inserted right after the call.

    Idempotent and allocation-light: a clean history comes back with its
    original message objects, so prompt-cache prefixes stay byte-stable.
    """
    out: List[Dict[str, Any]] = []
    seen_calls: set = set()
    for message in messages:
        if not isinstance(message, dict):
            out.append(message)
            continue
        calls = _tool_use_ids(message) if message.get("role") == "assistant" else []
        content = message.get("content")
        if isinstance(content, list) and message.get("role") == "user":
            kept = [
                b
                for b in content
                if not (
                    isinstance(b, dict)
                    and b.get("type") == "tool_result"
                    and b.get("tool_use_id") not in seen_calls
                )
            ]
            if len(kept) != len(content):
                if kept:
                    out.append({**message, "content": kept})
                seen_calls.update(calls)
                continue
        out.append(message)
        seen_calls.update(calls)

    answered: set = set()
    for message in out:
        if isinstance(message, dict):
            answered |= {i for i in _tool_result_ids(message) if i}

    # Reverse order keeps earlier indices valid while inserting.
    for index in range(len(out) - 1, -1, -1):
        message = out[index]
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        missing = [uid for uid in _tool_use_ids(message) if uid not in answered]
        if not missing:
            continue
        synthetic = [_synthetic_result(uid) for uid in missing]
        following = out[index + 1] if index + 1 < len(out) else None
        if (
            isinstance(following, dict)
            and following.get("role") == "user"
            and isinstance(following.get("content"), list)
            and any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in following["content"]
            )
        ):
            out[index + 1] = {**following, "content": [*following["content"], *synthetic]}
        else:
            out.insert(index + 1, {"role": "user", "content": synthetic})
        answered.update(missing)
    return out


def repair_all_tool_pairs(messages: List[Dict[str, Any]]) -> bool:
    """:func:`normalize_messages_for_request`, applied in place.

    For the few places that must leave the STORED history valid — a turn
    that is being closed after a stop or a failure, so the next turn's
    replay does not inherit a call that was never answered. Returns whether
    anything changed.
    """
    fixed = normalize_messages_for_request(messages)
    if len(fixed) == len(messages) and all(a is b for a, b in zip(fixed, messages)):
        return False
    messages[:] = fixed
    return True
