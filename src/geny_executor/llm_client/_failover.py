"""Account cool-downs and failure classification for multi-account routing.

A route is a list of accounts. When one of them fails in a way another
account could survive — a usage limit, a dead login, an outage — the router
moves on and marks that account cooling so every other agent in the process
skips it too. This module owns that shared, process-wide state and the
classification that decides whether a failure is worth failing over at all.
"""
from __future__ import annotations

import re
import time
from typing import Any, Callable, Optional

from geny_executor.core.errors import APIError, ErrorCategory

Notify = Callable[[dict[str, Any]], None]

# ── cooldowns (process-wide, keyed by account) ────────────────────────
# A pool across agents: when one agent's turn exhausts an account, another
# agent routed to the same account should not walk into the same wall.
_COOLDOWN: dict[str, tuple[float, str]] = {}

RATE_LIMIT_COOLDOWN_S = 15 * 60
AUTH_COOLDOWN_S = 5 * 60
SERVER_COOLDOWN_S = 60


def cool_down(account_id: str, seconds: float, reason: str) -> None:
    if account_id:
        _COOLDOWN[account_id] = (time.time() + max(1.0, seconds), reason)


def cooling(account_id: str) -> Optional[tuple[float, str]]:
    entry = _COOLDOWN.get(account_id)
    if entry is None:
        return None
    if entry[0] <= time.time():
        _COOLDOWN.pop(account_id, None)
        return None
    return entry


def clear_cooldown(account_id: str) -> None:
    _COOLDOWN.pop(account_id, None)


def snapshot() -> dict[str, dict[str, Any]]:
    now = time.time()
    return {
        k: {"until": v[0], "remaining": round(v[0] - now), "reason": v[1]}
        for k, v in list(_COOLDOWN.items())
        if v[0] > now
    }


# ── error classification ──────────────────────────────────────────────
#: "This account cannot answer right now, and another one might."
#:
#: Rate limits and spend caps are the same thing to a router: the account is
#: alive, the request is fine, and the next hop should take it. Keeping them
#: apart cost a production turn — a Claude subscription answered
#: ``You've hit your monthly spend limit``, which matched none of the
#: rate-limit wording, was classified as a bad request, and failed the turn
#: outright while a working OpenAI account sat next in the route.
_RATE_HINTS = re.compile(
    r"rate.?limit|usage limit|limit reached|quota|too many requests|429|"
    r"resets? (at|in)|out of (extra )?usage|overloaded|capacity|"
    # spend caps, credit exhaustion, billing stops — an empty wallet is a
    # limit, and the fallback exists for exactly this
    r"spend limit|spending limit|credit balance|insufficient[_ ]quota|"
    r"billing|payment required|402",
    re.I,
)
_AUTH_HINTS = re.compile(
    r"not logged in|please run /login|invalid api key|unauthori[sz]ed|401|"
    r"oauth token (has )?expired|authentication|invalid_grant|token_expired|"
    r"refresh token|login required|forbidden|403",
    re.I,
)


def classify_text(message: str, status: Optional[int] = None) -> ErrorCategory:
    if status == 429 or (status is None and _RATE_HINTS.search(message or "")):
        return ErrorCategory.RATE_LIMITED
    if status in (401, 403) or (status is None and _AUTH_HINTS.search(message or "")):
        return ErrorCategory.AUTH
    if status is not None and status >= 500:
        return ErrorCategory.SERVER_ERROR
    if status is not None and 400 <= status < 500:
        return ErrorCategory.BAD_REQUEST
    return ErrorCategory.UNKNOWN


def api_error(message: str, category: ErrorCategory, status: Optional[int] = None) -> APIError:
    return APIError(message, category=category, status_code=status)


def category_of(exc: BaseException) -> ErrorCategory:
    try:
        import httpx

        if isinstance(exc, httpx.TimeoutException):
            return ErrorCategory.TIMEOUT
        if isinstance(exc, httpx.TransportError):
            return ErrorCategory.NETWORK
    except Exception:
        pass
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return ErrorCategory.NETWORK if isinstance(exc, ConnectionError) else ErrorCategory.TIMEOUT
    if isinstance(exc, APIError):
        cat = exc.category
        if cat in (ErrorCategory.UNKNOWN, ErrorCategory.CLI_PROTOCOL_ERROR):
            refined = classify_text(str(exc), getattr(exc, "status_code", None))
            if refined != ErrorCategory.UNKNOWN:
                return refined
        if cat == ErrorCategory.CLI_AUTH_FAILED:
            return ErrorCategory.AUTH
        return cat
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    return classify_text(str(exc), status if isinstance(status, int) else None)


#: failures where retrying the SAME route cannot help once every hop has
#: failed — a usage limit or a dead login does not recover in seconds
EXHAUSTED_TERMINAL = {
    ErrorCategory.RATE_LIMITED,
    ErrorCategory.AUTH,
    ErrorCategory.CLI_NOT_FOUND,
    ErrorCategory.CLI_AUTH_FAILED,
}

#: failures where another account/model may well succeed
FAILOVER = {
    ErrorCategory.RATE_LIMITED,
    ErrorCategory.AUTH,
    ErrorCategory.SERVER_ERROR,
    ErrorCategory.NETWORK,
    ErrorCategory.TIMEOUT,
    ErrorCategory.CLI_NOT_FOUND,
    ErrorCategory.CLI_TIMEOUT,
}


def cooldown_for(category: ErrorCategory) -> float:
    if category == ErrorCategory.RATE_LIMITED:
        return RATE_LIMIT_COOLDOWN_S
    if category in (ErrorCategory.AUTH, ErrorCategory.CLI_NOT_FOUND):
        return AUTH_COOLDOWN_S
    return SERVER_COOLDOWN_S


def strip_cache_markers(value: Any) -> Any:
    """Deep copy without Anthropic `cache_control` keys — they are an
    Anthropic extension and other wires may reject them."""
    if isinstance(value, dict):
        return {k: strip_cache_markers(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [strip_cache_markers(v) for v in value]
    return value
