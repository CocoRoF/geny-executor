"""Fix the tool inputs whose meaning is unambiguous, before validation.

Every rejected call costs a model round trip — the whole conversation sent
again — and the model often repeats the same mistake several times before it
sees it. Measured on XGEN's runtime (same stack): ``max_results: "3"`` alone
drew ``'3' is not of type 'integer'`` six to fifteen times in a row; arrays
and objects arrived as JSON strings; parameter names were mixed up between
tools (``file_path`` where the schema says ``path``).

The rules look at the schema only — no word lists, no per-tool exceptions —
and change a value only when there is one reading of it. Anything ambiguous
is left for the validator to reject as before. Inputs are never mutated; an
unchanged input comes back as the same object.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

#: The provider's tool-call arguments could not be parsed as JSON; the raw
#: text is kept under this key. Collapsing it to ``{}`` made it look as if
#: the model had called the tool with nothing, and the model was told it had
#: left out a required field — a mistake it never made.
UNPARSED_ARGUMENTS_KEY = "__unparsed_arguments__"

#: How much of unparsable arguments is kept.
UNPARSED_KEEP_CHARS = 4000

_INT_RE = re.compile(r"^\s*-?\d+\s*$")
_NUM_RE = re.compile(r"^\s*-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?\s*$")


def unparsed_arguments(raw: Any) -> Dict[str, Any]:
    """The input for a call whose arguments could not be read."""
    text = raw if isinstance(raw, str) else str(raw)
    return {UNPARSED_ARGUMENTS_KEY: text[:UNPARSED_KEEP_CHARS]}


def unparsed_arguments_reason(raw: Any) -> str:
    """What to tell the model about a call whose arguments were unreadable."""
    length = len(raw) if isinstance(raw, str) else 0
    return (
        f"the tool-call arguments were not valid JSON and could not be parsed "
        f"({length} chars, usually cut off mid-generation). Nothing ran. Send the "
        f"call again with valid JSON; if the arguments carry a long body (a file, a "
        f"script), write it in smaller pieces instead of one call."
    )


def _types(schema: Dict[str, Any]) -> List[str]:
    t = schema.get("type")
    return [t] if isinstance(t, str) else [x for x in (t or []) if isinstance(x, str)]


def _parse_container(text: str, types: List[str]) -> Any:
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        parsed = json.loads(stripped)
    except (ValueError, TypeError):
        return None
    if isinstance(parsed, list) and "array" in types:
        return parsed
    if isinstance(parsed, dict) and "object" in types:
        return parsed
    return None


def coerce_input(schema: Any, payload: Any) -> Any:
    """Turn a string the schema does not accept into the value it plainly means.

    Only where the schema does not allow a string: an integer string becomes
    an integer, a number string a number, ``"true"``/``"false"`` a boolean,
    and ``"[…]"``/``"{…}"`` the array or object the schema asks for.
    Recurses into ``properties`` and ``items``.
    """
    if not isinstance(schema, dict):
        return payload
    types = _types(schema)

    if isinstance(payload, str) and types and "string" not in types:
        if "integer" in types and _INT_RE.match(payload):
            return int(payload.strip())
        if "number" in types and _NUM_RE.match(payload):
            text = payload.strip()
            return int(text) if _INT_RE.match(text) else float(text)
        if "boolean" in types and payload.strip().lower() in ("true", "false"):
            return payload.strip().lower() == "true"
        if "array" in types or "object" in types:
            parsed = _parse_container(payload, types)
            if parsed is not None:
                return parsed
        return payload

    if isinstance(payload, dict):
        props = schema.get("properties")
        if not isinstance(props, dict):
            return payload
        out: Optional[Dict[str, Any]] = None
        for key, value in payload.items():
            sub = props.get(key)
            if not isinstance(sub, dict):
                continue
            fixed = coerce_input(sub, value)
            if fixed is not value:
                if out is None:
                    out = dict(payload)
                out[key] = fixed
        return payload if out is None else out

    if isinstance(payload, list) and isinstance(schema.get("items"), dict):
        fixed_items = [coerce_input(schema["items"], v) for v in payload]
        if any(a is not b for a, b in zip(fixed_items, payload)):
            return fixed_items
    return payload


def _value_fits(subschema: Dict[str, Any], value: Any) -> bool:
    import jsonschema

    try:
        jsonschema.validate(instance=value, schema=subschema)
    except Exception:  # noqa: BLE001 — a broken schema decides nothing
        return False
    return True


def _name_akin(a: str, b: str) -> bool:
    """One name contains the other — ``file_path``/``path``, ``prompt``/``positive_prompt``."""
    x = "".join(ch for ch in a.lower() if ch.isalnum())
    y = "".join(ch for ch in b.lower() if ch.isalnum())
    return bool(x and y) and (x in y or y in x)


def _pick(
    key: str, candidates: List[str], subschema: Dict[str, Any], payload: Dict[str, Any]
) -> Optional[str]:
    fits = [c for c in candidates if _value_fits(subschema, payload[c])]
    if len(fits) == 1:
        return fits[0]
    akin = [c for c in fits if _name_akin(key, c)]
    return akin[0] if len(akin) == 1 else None


def repair_missing_required(
    schema: Any, payload: Any
) -> Optional[Tuple[Dict[str, Any], List[str]]]:
    """Move a key the schema does not have into a required field that is missing.

    Only when (1) the key is not a parameter at all, so its value would be
    thrown away anyway, (2) the value passes the missing field's schema, and
    (3) exactly one key fits — or, among several, exactly one whose name
    contains the other. Returns ``(fixed input, notes for the model)`` or None.
    """
    if not isinstance(schema, dict) or not isinstance(payload, dict):
        return None
    props = schema.get("properties")
    required = schema.get("required")
    if not isinstance(props, dict) or not isinstance(required, list):
        return None
    missing = [k for k in required if isinstance(k, str) and k not in payload]
    unknown = [k for k in payload if isinstance(k, str) and k not in props]
    if not missing or not unknown:
        return None

    out = dict(payload)
    notes: List[str] = []
    remaining = list(unknown)
    for key in missing:
        sub = props.get(key)
        if not isinstance(sub, dict) or not remaining:
            continue
        cand = _pick(key, remaining, sub, out)
        if cand is None:
            continue
        out[key] = out.pop(cand)
        remaining.remove(cand)
        notes.append(f"'{cand}' is not a parameter of this tool — used it as '{key}'.")
    return (out, notes) if notes else None


def describe_validation_failure(schema: Any, payload: Any, message: str) -> str:
    """A missing-field error that says what was needed and what was sent.

    ``'path' is a required property`` alone leaves the model guessing what
    it sent; side by side, a mixed-up name is fixed in one go.
    """
    if "is a required property" not in message:
        return message
    if not isinstance(schema, dict) or not isinstance(payload, dict):
        return message
    required = [k for k in schema.get("required") or [] if isinstance(k, str)]
    if not required:
        return message
    sent = ", ".join(str(k) for k in payload) or "(nothing)"
    return f"{message}. required: {', '.join(required)}; you sent: {sent}"


__all__ = [
    "UNPARSED_ARGUMENTS_KEY",
    "coerce_input",
    "describe_validation_failure",
    "repair_missing_required",
    "unparsed_arguments",
    "unparsed_arguments_reason",
]
