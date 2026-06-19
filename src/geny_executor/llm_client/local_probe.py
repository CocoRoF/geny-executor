"""Context-window discovery for local LLM endpoints.

The pipeline's ``context_window_budget`` (Stage-2 proactive compaction +
Stage-4 ``TokenBudgetGuard``) defaults to 200_000 — correct for the frontier
cloud models but disastrous for a local 8K-context model, where it means
compaction never fires and the request overflows the server.

This module discovers a local model's real context window so a host can set
``context_window_budget`` to match. It is an explicit, host-driven utility
(NOT auto-run during pipeline build): pipeline construction stays free of
implicit network I/O, and the host decides when to pay the probe.

Ollama exposes the window via its native ``/api/show`` endpoint (the
OpenAI-compatible ``/v1`` surface does not), so the probe targets that.
Everything here is best-effort: any failure (server down, unexpected shape,
timeout) resolves to ``None`` and the caller keeps its existing budget.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Optional

from geny_executor.llm_client.profiles import resolve_profile


logger = logging.getLogger(__name__)

#: ``async def transport(url, json_body) -> Optional[dict]`` — test hook to
#: stub HTTP without a live server (mirrors the embedding clients' pattern).
Transport = Callable[[str, dict], Awaitable[Optional[dict]]]

_DEFAULT_TIMEOUT_S = 5.0


def _ollama_native_root(base_url: str) -> str:
    """``http://host:11434/v1`` → ``http://host:11434`` (native API root)."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")].rstrip("/")
    return root


def _extract_num_ctx(body: Any) -> Optional[int]:
    """Pull the context window out of an Ollama ``/api/show`` response.

    Precedence mirrors Ollama itself: a Modelfile ``num_ctx`` parameter (the
    window the model is actually *loaded* with) wins over the GGUF's trained
    ``*.context_length`` (the maximum the weights support).
    """
    if not isinstance(body, dict):
        return None

    # 1) Modelfile parameter: ``parameters`` is a newline-delimited string
    #    like ``"num_ctx 8192\nstop \"<|im_end|>\""``.
    params = body.get("parameters")
    if isinstance(params, str):
        for line in params.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "num_ctx":
                try:
                    return int(parts[1])
                except ValueError:
                    pass

    # 2) GGUF metadata: ``model_info["<arch>.context_length"]``.
    info = body.get("model_info")
    if isinstance(info, dict):
        arch = info.get("general.architecture")
        if isinstance(arch, str):
            val = info.get(f"{arch}.context_length")
            if isinstance(val, int):
                return val
        # Architecture key absent/odd — accept any ``*.context_length`` int.
        for key, val in info.items():
            if key.endswith(".context_length") and isinstance(val, int):
                return val

    return None


async def _post(
    url: str, body: dict, transport: Optional[Transport], timeout: float
) -> Optional[dict]:
    if transport is not None:
        return await transport(url, body)
    try:
        import httpx  # transitive via anthropic>=0.52
    except ImportError:
        logger.debug("local context probe: httpx unavailable; skipping")
        return None
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=body)
    except httpx.HTTPError as exc:
        logger.debug("local context probe: transport failure for %s: %s", url, exc)
        return None
    if resp.status_code != 200:
        logger.debug("local context probe: HTTP %s from %s", resp.status_code, url)
        return None
    try:
        return resp.json()
    except ValueError:
        return None


async def probe_ollama_num_ctx(
    base_url: str,
    model: str,
    *,
    transport: Optional[Transport] = None,
    timeout: float = _DEFAULT_TIMEOUT_S,
) -> Optional[int]:
    """Context window for *model* on an Ollama server, or ``None``.

    Hits ``<base_url without /v1>/api/show`` with ``{"model": model}``.
    Best-effort: returns ``None`` on any error so the caller falls back to
    its configured ``context_window_budget``.
    """
    if not base_url or not model:
        return None
    url = f"{_ollama_native_root(base_url)}/api/show"
    body = await _post(url, {"model": model}, transport, timeout)
    num_ctx = _extract_num_ctx(body)
    if num_ctx:
        logger.debug("local context probe: %s @ %s → num_ctx=%d", model, url, num_ctx)
    return num_ctx


async def resolve_local_context_window(
    provider: str,
    base_url: str,
    model: str,
    *,
    transport: Optional[Transport] = None,
    timeout: float = _DEFAULT_TIMEOUT_S,
) -> Optional[int]:
    """Discover the context window for a profiled local *provider*, or ``None``.

    Dispatches by profile: ``ollama`` (incl. a ``custom``/``local`` endpoint
    the host knows points at Ollama — pass ``provider="ollama"`` for those)
    probes ``/api/show``. Non-Ollama profiles return ``None`` today (LM Studio
    has no comparable metadata endpoint); the signature is the stable
    extension point as more probes are added.

    Hosts use the result to set ``PipelineConfig.context_window_budget`` so a
    small local model gets correctly-sized compaction instead of the 200_000
    cloud default.
    """
    if not provider:
        return None
    try:
        profile = resolve_profile(provider)
    except ValueError:
        return None
    if not profile.is_local:
        return None
    if profile.name == "ollama":
        return await probe_ollama_num_ctx(base_url, model, transport=transport, timeout=timeout)
    return None


__all__ = [
    "Transport",
    "probe_ollama_num_ctx",
    "resolve_local_context_window",
]
