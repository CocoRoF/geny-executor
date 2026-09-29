"""How long a model call may take before it is treated as stuck.

Without these a provider that stopped answering held the turn until the
SDK's own 10-minute read timeout — per attempt, and the SDK retried twice
underneath the stage's own retries, so one stalled call could hold a turn
for over half an hour with nothing on screen. Values come from XGEN's
runtime (same stack), except the first-chunk wait, which is longer here:
Geny also drives local llama.cpp / vLLM servers whose prompt processing on a
long context takes minutes before the first token.

Every value can be overridden per process by its environment variable.
"""

from __future__ import annotations

import os
from typing import Any, Dict

#: TCP connect / pool acquire.
CONNECT_TIMEOUT_S = 10.0
#: Stream open → first content chunk (text, thinking, tool call).
FIRST_CHUNK_TIMEOUT_S = 300.0
#: Between content chunks once the answer has started.
IDLE_TIMEOUT_S = 120.0
#: A whole non-streaming request.
REQUEST_TIMEOUT_S = 600.0
#: Retries after a timeout (other recoverable errors keep the stage's own
#: count). A provider that stalled once usually stalls again; four stalls
#: in a row were most of an hour.
TIMEOUT_RETRIES = 1
#: The SDKs' own retries. The stage retries (with events a host can see);
#: an SDK retrying underneath multiplied every attempt by three, silently.
SDK_MAX_RETRIES = 0


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def connect_timeout_s() -> float:
    return _env_float("GENY_LLM_CONNECT_TIMEOUT_S", CONNECT_TIMEOUT_S)


def first_chunk_timeout_s() -> float:
    return _env_float("GENY_LLM_FIRST_CHUNK_TIMEOUT_S", FIRST_CHUNK_TIMEOUT_S)


def idle_timeout_s() -> float:
    return _env_float("GENY_LLM_IDLE_TIMEOUT_S", IDLE_TIMEOUT_S)


def request_timeout_s() -> float:
    return _env_float("GENY_LLM_REQUEST_TIMEOUT_S", REQUEST_TIMEOUT_S)


def timeout_retries() -> int:
    raw = os.environ.get("GENY_LLM_TIMEOUT_RETRIES")
    try:
        return max(0, int(raw)) if raw else TIMEOUT_RETRIES
    except ValueError:
        return TIMEOUT_RETRIES


def sdk_client_kwargs(timeout_cls: Any) -> Dict[str, Any]:
    """``timeout`` / ``max_retries`` for an httpx-based SDK client.

    The read timeout sits above the stage's own limits so the stage's
    watchdog, which says what happened, fires first.
    """
    read = max(first_chunk_timeout_s(), idle_timeout_s(), request_timeout_s()) + 30.0
    connect = connect_timeout_s()
    return {
        "timeout": timeout_cls(read, connect=connect, pool=connect),
        "max_retries": SDK_MAX_RETRIES,
    }


__all__ = [
    "CONNECT_TIMEOUT_S",
    "FIRST_CHUNK_TIMEOUT_S",
    "IDLE_TIMEOUT_S",
    "REQUEST_TIMEOUT_S",
    "SDK_MAX_RETRIES",
    "TIMEOUT_RETRIES",
    "connect_timeout_s",
    "first_chunk_timeout_s",
    "idle_timeout_s",
    "request_timeout_s",
    "sdk_client_kwargs",
    "timeout_retries",
]
