"""HITL request / decision types for Stage 15 (S9b.3)."""

from __future__ import annotations

import enum
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_token() -> str:
    return secrets.token_urlsafe(16)


class HITLDecision(str, enum.Enum):
    """Possible verdicts a human operator can return."""

    APPROVE = "approve"
    REJECT = "reject"
    CANCEL = "cancel"  # treat as escalate / abort


@dataclass(frozen=True)
class HITLRequest:
    """An approval request emitted by the host or Stage 11 review.

    Hosts populate ``state.shared['hitl_request']`` with one of these
    (or a dict that coerces to one) before Stage 15 runs. The
    ``token`` is opaque — used by request handlers / future
    Pipeline.resume API to correlate decisions with requests.
    """

    token: str = field(default_factory=_new_token)
    reason: str = ""
    severity: str = "warn"  # info | warn | error
    tool_call_id: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token": self.token,
            "reason": self.reason,
            "severity": self.severity,
            "tool_call_id": self.tool_call_id,
            "payload": dict(self.payload),
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class HITLEntry:
    """Audit entry combining request + decision + outcome.

    Stage 15 appends one of these to ``state.shared['hitl_history']``
    after each request resolves.
    """

    request: HITLRequest
    decision: HITLDecision
    decided_at: datetime = field(default_factory=_now)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "decision": self.decision.value,
            "decided_at": self.decided_at.isoformat(),
            "note": self.note,
        }


def coerce_request(value: Any) -> Optional[HITLRequest]:
    """Accept an :class:`HITLRequest` or a dict and return a request."""
    if value is None:
        return None
    if isinstance(value, HITLRequest):
        return value
    if not isinstance(value, dict):
        return None
    severity = str(value.get("severity", "warn")).lower()
    if severity not in {"info", "warn", "error"}:
        severity = "warn"
    return HITLRequest(
        token=str(value.get("token") or _new_token()),
        reason=str(value.get("reason") or ""),
        severity=severity,
        tool_call_id=str(value.get("tool_call_id") or ""),
        payload=dict(value.get("payload") or {}),
    )


def coerce_decision(value: Any) -> Optional[HITLDecision]:
    """Best-effort decision coercion."""
    if value is None:
        return None
    if isinstance(value, HITLDecision):
        return value
    try:
        return HITLDecision(str(value).lower())
    except ValueError:
        return None


__all__ = [
    "HITLDecision",
    "HITLEntry",
    "HITLRequest",
    "coerce_decision",
    "coerce_request",
]
