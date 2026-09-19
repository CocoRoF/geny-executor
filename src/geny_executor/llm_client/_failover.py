"""Account cool-downs and failure classification for multi-account routing.

A route is a list of accounts. When one of them fails in a way another
account could survive — a usage limit, a dead login, an outage — the router
moves on and marks that account cooling so every other agent in the process
skips it too. This module owns that shared, process-wide state and the
classification that decides whether a failure is worth failing over at all.
"""

from __future__ import annotations

import re
from datetime import datetime
import time
from typing import Any, Callable, Mapping, Optional

from geny_executor.core.errors import APIError, ErrorCategory

Notify = Callable[[dict[str, Any]], None]

# ── cooldowns (process-wide, keyed by account) ────────────────────────
# A pool across agents: when one agent's turn exhausts an account, another
# agent routed to the same account should not walk into the same wall.
_COOLDOWN: dict[str, tuple[float, str]] = {}

RATE_LIMIT_COOLDOWN_S = 15 * 60
AUTH_COOLDOWN_S = 5 * 60
SERVER_COOLDOWN_S = 60

#: What a bench costs when there is nowhere else to go. Benching the ONLY
#: account for fifteen minutes does not protect a quota — it just stops the
#: user working. A transient throttle gets a short sit-down instead, and the
#: turn after it either succeeds or benches again.
SOLE_ACCOUNT_COOLDOWN_S = 60

#: A rate limit on one model says nothing about its siblings: the metered
#: APIs enforce requests/tokens per minute PER MODEL, so a 429 on opus leaves
#: haiku on the same key usable. Keyed (account, model) alongside the
#: credential-wide bench.
_MODEL_COOLDOWN: dict[tuple[str, str], tuple[float, str]] = {}

#: …but only where that is true. A subscription plan meters the ACCOUNT: when
#: Claude Code or a ChatGPT plan says "usage limit reached", every model on
#: that login is out, and benching just the one that happened to be asked
#: would send the next turn straight into the same wall on a sibling.
PER_MODEL_RATE_LIMITS = frozenset({"anthropic", "openai", "google", "vllm"})


def rate_limit_is_per_model(provider: str) -> bool:
    """True when a 429 from *provider* is about the model, not the account."""
    return provider in PER_MODEL_RATE_LIMITS


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
    """Lift every bench on this account — credential-wide and per model.

    One call, because "this account is fine now" is one fact; leaving a model
    bench behind after clearing the account is how a recovered account stays
    half-benched with nothing to explain it.
    """
    _COOLDOWN.pop(account_id, None)
    clear_model_cooldowns(account_id)


# ── load (process-wide, keyed by account) ─────────────────────────────
# Cooldowns answer "who is broken". This answers "whose turn is it".
# A route of several healthy accounts used to send EVERY turn to the
# first one until it hit its cap — failover, which only reacts once an
# account is already exhausted. For subscription plans that is backwards:
# the point of holding two logins is that neither reaches its limit.
#
# Process-wide for the same reason cooldowns are: sibling agents share
# the accounts, so a per-client counter would let three agents each
# "balance" onto the same first account.
_LAST_USED: dict[str, float] = {}


def mark_used(account_id: str) -> None:
    if account_id:
        _LAST_USED[account_id] = time.time()


def last_used(account_id: str) -> float:
    """When this account last answered; 0.0 if never (so it goes first)."""
    return _LAST_USED.get(account_id, 0.0)


def snapshot() -> dict[str, dict[str, Any]]:
    """Everything currently benched, keyed by account.

    Model benches belong here too. This is the one view of "what cannot
    answer right now" — the host renders it, and callers clear from it — so
    a bench that does not appear is a bench nobody can see or lift.
    """
    now = time.time()
    out: dict[str, dict[str, Any]] = {}
    for account, (until, reason) in list(_COOLDOWN.items()):
        if until > now:
            out[account] = {
                "until": until,
                "remaining": round(until - now),
                "reason": reason,
                "scope": "account",
            }
    for (account, model), (until, reason) in list(_MODEL_COOLDOWN.items()):
        if until <= now:
            continue
        entry = out.setdefault(
            account,
            {"until": until, "remaining": round(until - now), "reason": reason, "scope": "model"},
        )
        models = entry.setdefault("models", {})
        models[model] = {"until": until, "remaining": round(until - now), "reason": reason}
        if entry.get("scope") == "model" and until > entry["until"]:
            entry["until"] = until
            entry["remaining"] = round(until - now)
    return out


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
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
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


def cooldown_for(category: ErrorCategory, *, sole_account: bool = False) -> float:
    """How long to bench an account for this kind of failure.

    ``sole_account`` caps a transient throttle: with nothing to rotate to, a
    long bench is not protection, it is downtime. A dead login is exempt —
    waiting cannot fix it and retrying it every minute only spams the vendor.
    """
    if category == ErrorCategory.RATE_LIMITED:
        base = RATE_LIMIT_COOLDOWN_S
    elif category in (ErrorCategory.AUTH, ErrorCategory.CLI_NOT_FOUND):
        return AUTH_COOLDOWN_S
    else:
        base = SERVER_COOLDOWN_S
    return min(base, SOLE_ACCOUNT_COOLDOWN_S) if sole_account else base


# ── when the provider tells us when to come back ──────────────────────
# A blind fifteen minutes is a guess in both directions: too short and the
# next turn walks into the same wall, too long and a recovered account sits
# idle. Providers usually say. Take the answer when they do.

_RETRY_IN = re.compile(
    r"(?:try|retry|available|come back|again)\b[^.\n]{0,24}?\bin\s+"
    r"(\d+(?:\.\d+)?)\s*(second|sec|minute|min|hour|hr|day|s|m|h|d)s?\b",
    re.I,
)
_RETRY_AFTER = re.compile(r"retry[-_ ]?after[\s:=]+(\d+(?:\.\d+)?)", re.I)
_UNIT_S = {
    "s": 1,
    "sec": 1,
    "second": 1,
    "m": 60,
    "min": 60,
    "minute": 60,
    "h": 3600,
    "hr": 3600,
    "hour": 3600,
    "d": 86400,
    "day": 86400,
}


def _absolute(value: Any) -> Optional[float]:
    """Epoch seconds from epoch-s, epoch-ms or ISO-8601; ``None`` otherwise."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if number <= 0:
            return None
        return number / 1000.0 if number > 1_000_000_000_000 else number
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            number = float(raw)
            return number / 1000.0 if number > 1_000_000_000_000 else number
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def reset_at_from(
    message: str = "", *, fields: Optional[Mapping[str, Any]] = None
) -> Optional[float]:
    """When the provider says this account is usable again, as epoch seconds.

    Reads, in order of authority: an explicit field the provider sent
    (``reset_at`` / ``resets_at`` / ``retry_after`` and their spellings), then
    the error text — vendors put it there far more often than in a field
    ("try again in 42 seconds", "retry-after: 90"). Returns ``None`` when
    nothing says, and the caller falls back to a category default.
    """
    for key in ("reset_at", "resets_at", "resetAt", "retry_until", "reset_time"):
        found = _absolute((fields or {}).get(key))
        if found is not None:
            return found
    for key in ("retry_after", "retryAfter", "retry-after"):
        raw = (fields or {}).get(key)
        try:
            seconds = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if seconds > 0:
            return time.time() + seconds

    text = message or ""
    match = _RETRY_AFTER.search(text)
    if match:
        return time.time() + float(match.group(1))
    match = _RETRY_IN.search(text)
    if match:
        unit = _UNIT_S.get(match.group(2).lower())
        if unit:
            return time.time() + float(match.group(1)) * unit
    return None


# ── model-scoped cooldowns ────────────────────────────────────────────


def cool_down_model(account_id: str, model: str, until: float, reason: str) -> None:
    """Bench one model on this account, leaving its siblings usable."""
    if account_id and model and until > time.time():
        _MODEL_COOLDOWN[(account_id, model)] = (until, reason)


def model_cooling(account_id: str, model: Optional[str]) -> Optional[tuple[float, str]]:
    """The active model bench blocking this hop, or ``None``.

    A caller that does not name a model stays conservative: ANY active model
    bench on the account blocks it, because an unscoped route could land on
    the very model that just failed.
    """
    now = time.time()
    if model:
        entry = _MODEL_COOLDOWN.get((account_id, model))
        if entry is None:
            return None
        if entry[0] <= now:
            _MODEL_COOLDOWN.pop((account_id, model), None)
            return None
        return entry
    live = [
        v for (acct, _m), v in list(_MODEL_COOLDOWN.items()) if acct == account_id and v[0] > now
    ]
    return max(live, key=lambda v: v[0]) if live else None


def clear_model_cooldowns(account_id: str) -> None:
    for key in [k for k in _MODEL_COOLDOWN if k[0] == account_id]:
        _MODEL_COOLDOWN.pop(key, None)


def strip_cache_markers(value: Any) -> Any:
    """Deep copy without Anthropic `cache_control` keys — they are an
    Anthropic extension and other wires may reject them."""
    if isinstance(value, dict):
        return {k: strip_cache_markers(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [strip_cache_markers(v) for v in value]
    return value
