"""A call a person refused is not asked again in the same turn.

Measured on XGEN's runtime (same stack, 2026-09-23): a user refused
``rm -rf …`` at the approval prompt and the model called the same command
again — three times, three prompts. The repeat guard only warns until the
fourth identical failure, so nothing stopped it. An approval prompt is the
whole of "a person stops what cannot be undone"; asked the same question
over and over, a person eventually clicks the wrong button, and one "allow
once" runs it.

Rules:

* A result starting ``ERROR user_denied`` / ``ERROR access_denied`` is a
  refusal (the permission matrix, a HITL "no", a host that maps its own
  refusals onto that header).
* The refused call's signature is remembered for the rest of the turn; the
  same action again is answered without running — and without a prompt.
* "The same action" is read broadly: quotes and runs of whitespace are
  ignored. Matching a near-retry of a refused action is the safe mistake.
* Per turn (``state.shared``). Next turn the user can say "go ahead".
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional

#: ``state.shared`` key — refused call signature → reason.
DENIED_KEY = "tool.denied_calls"

_DENIAL_PREFIXES = ("ERROR user_denied", "ERROR access_denied")
_QUOTES = re.compile(r"[\"'`]")
_SPACES = re.compile(r"\s+")


def _text(result: Dict[str, Any]) -> str:
    content = result.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content if isinstance(b, dict)]
        return "\n".join(p for p in parts if isinstance(p, str))
    return str(content or "")


def is_denial(result: Dict[str, Any]) -> bool:
    return bool(result.get("is_error")) and _text(result).lstrip().startswith(_DENIAL_PREFIXES)


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        return _SPACES.sub(" ", _QUOTES.sub("", value)).strip()
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    return value


def signature(tool_call: Dict[str, Any]) -> str:
    """Tool name + normalised input: a retry that only changes quoting is the same."""
    name = str(tool_call.get("tool_name") or "")
    try:
        raw = json.dumps(
            _normalize(tool_call.get("tool_input") or {}),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    except (TypeError, ValueError):
        raw = str(tool_call.get("tool_input"))
    return name + ":" + hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:16]


def refused_result(tool_call: Dict[str, Any], shared: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The answer, instead of running, for an action refused earlier this turn."""
    denied = shared.get(DENIED_KEY) or {}
    reason = denied.get(signature(tool_call))
    if reason is None:
        return None
    return {
        "type": "tool_result",
        "tool_use_id": tool_call.get("tool_use_id", ""),
        "is_error": True,
        "content": (
            "ERROR user_denied_repeat: this action was already refused in this request. "
            "It was not run, and no approval was asked again.\n"
            f"Refused because: {reason}\n"
            "Do not try it again (a change of quotes or spacing is the same action), and do "
            "not reach the same effect another way. Tell the user it was not done and ask "
            "how to proceed."
        ),
    }


def observe(
    tool_calls: List[Dict[str, Any]], results: List[Dict[str, Any]], shared: Dict[str, Any]
) -> List[str]:
    """Remember the calls refused this round; returns the tools newly refused."""
    by_id = {str(r.get("tool_use_id") or ""): r for r in results}
    denied = dict(shared.get(DENIED_KEY) or {})
    added: List[str] = []
    for tc in tool_calls:
        r = by_id.get(str(tc.get("tool_use_id") or ""))
        if r is None or not is_denial(r):
            continue
        sig = signature(tc)
        if sig not in denied:
            text = _text(r).strip()
            denied[sig] = (text.splitlines()[0] if text else "denied")[:300]
            added.append(str(tc.get("tool_name") or ""))
    if added:
        shared[DENIED_KEY] = denied
    return added


__all__ = ["DENIED_KEY", "is_denial", "observe", "refused_result", "signature"]
