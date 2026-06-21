"""Per-provider model discovery — list the models a backend actually serves.

A host (e.g. a UI that builds a model picker) should show the models the
selected backend *really* offers, refreshed as the backend changes (a new
Ollama pull, an upgraded Claude Code CLI, a new cloud model), and only fall
back to a hand-maintained catalogue when live discovery is impossible.

This module is that live-discovery utility. Like :mod:`local_probe`, it is an
explicit host-driven call (never auto-run during pipeline build — construction
stays free of network I/O) and entirely best-effort: any failure (server down,
missing key, unexpected shape, timeout) resolves to a result with
``source="unavailable"`` and a short ``error``, so the caller keeps its own
fallback list.

Coverage:
  * ``openai``                         → GET ``<base>/v1/models`` (Bearer key)
  * ``ollama``                         → GET ``<root>/api/tags`` (native)
  * ``lmstudio`` / ``vllm`` / ``custom`` / ``local`` → GET ``<base>/v1/models``
  * ``anthropic``                      → GET ``/v1/models`` (x-api-key)
  * ``google``                         → GET ``/v1beta/models`` (?key=)
  * ``claude_code_cli``                → ``unavailable`` (the CLI exposes no
                                         model-list command; the host uses its
                                         version-robust aliases instead)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Optional

logger = logging.getLogger(__name__)

#: ``async def transport(url, headers) -> Optional[(status, json)]`` — test
#: hook to stub HTTP without a live server (mirrors local_probe's pattern).
Transport = Callable[[str, dict], Awaitable[Optional["_HttpResult"]]]

_DEFAULT_TIMEOUT_S = 6.0

_OPENAI_COMPATIBLE = {"openai", "vllm", "lmstudio", "custom", "local"}

_CLOUD_BASE = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta",
}


@dataclass(frozen=True)
class _HttpResult:
    status: int
    json: Any


@dataclass(frozen=True)
class ModelInfo:
    """One model a backend reports it can serve."""

    id: str
    display_name: Optional[str] = None


@dataclass(frozen=True)
class ModelDiscovery:
    """Result of a discovery attempt.

    ``source="live"`` → ``models`` came from the backend. ``source="unavailable"``
    → discovery could not run (no key, unreachable, unsupported provider); the
    host should fall back to its static catalogue. ``error`` is a short reason.
    """

    provider: str
    models: List[ModelInfo] = field(default_factory=list)
    source: str = "unavailable"
    error: Optional[str] = None


def _ollama_native_root(base_url: str) -> str:
    root = (base_url or "").rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")].rstrip("/")
    return root


async def _get(
    url: str, headers: dict, transport: Optional[Transport], timeout: float
) -> Optional[_HttpResult]:
    if transport is not None:
        return await transport(url, headers)
    try:
        import httpx  # transitive via anthropic>=0.52
    except ImportError:
        logger.debug("model discovery: httpx unavailable; skipping")
        return None
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        logger.debug("model discovery: transport failure for %s: %s", url, exc)
        return None
    try:
        body = resp.json()
    except ValueError:
        body = None
    return _HttpResult(status=resp.status_code, json=body)


def _parse_openai_models(body: Any) -> List[ModelInfo]:
    """``{"data": [{"id": ...}, ...]}`` (OpenAI / LM Studio / vLLM / custom)."""
    out: List[ModelInfo] = []
    data = body.get("data") if isinstance(body, dict) else None
    if isinstance(data, list):
        for m in data:
            if isinstance(m, dict) and m.get("id"):
                out.append(ModelInfo(id=str(m["id"])))
    return out


def _parse_ollama_tags(body: Any) -> List[ModelInfo]:
    """``{"models": [{"name": "llama3:latest", ...}, ...]}`` (Ollama native)."""
    out: List[ModelInfo] = []
    models = body.get("models") if isinstance(body, dict) else None
    if isinstance(models, list):
        for m in models:
            if isinstance(m, dict) and m.get("name"):
                out.append(ModelInfo(id=str(m["name"])))
    return out


def _parse_anthropic_models(body: Any) -> List[ModelInfo]:
    """``{"data": [{"id": ..., "display_name": ...}]}`` (Anthropic)."""
    out: List[ModelInfo] = []
    data = body.get("data") if isinstance(body, dict) else None
    if isinstance(data, list):
        for m in data:
            if isinstance(m, dict) and m.get("id"):
                out.append(
                    ModelInfo(
                        id=str(m["id"]),
                        display_name=(
                            str(m["display_name"])
                            if m.get("display_name")
                            else None
                        ),
                    )
                )
    return out


def _parse_google_models(body: Any) -> List[ModelInfo]:
    """``{"models": [{"name": "models/gemini-…", "displayName": …,
    "supportedGenerationMethods": [...]}]}``. Keep generateContent-capable."""
    out: List[ModelInfo] = []
    models = body.get("models") if isinstance(body, dict) else None
    if isinstance(models, list):
        for m in models:
            if not isinstance(m, dict) or not m.get("name"):
                continue
            methods = m.get("supportedGenerationMethods") or []
            if isinstance(methods, list) and methods and "generateContent" not in methods:
                continue
            raw = str(m["name"])
            mid = raw[len("models/"):] if raw.startswith("models/") else raw
            out.append(
                ModelInfo(
                    id=mid,
                    display_name=(
                        str(m["displayName"]) if m.get("displayName") else None
                    ),
                )
            )
    return out


async def discover_models(
    provider: str,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    transport: Optional[Transport] = None,
    timeout: float = _DEFAULT_TIMEOUT_S,
) -> ModelDiscovery:
    """Discover the models *provider* currently serves (best-effort).

    ``api_key`` is required for the cloud providers (openai/anthropic/google);
    ``base_url`` for the local/openai-compatible ones (defaults per provider).
    Never raises — returns ``source="unavailable"`` with an ``error`` on any
    problem so the caller falls back to its static catalogue.
    """
    p = (provider or "").strip().lower()
    if not p:
        return ModelDiscovery(provider=provider, error="no provider")

    # The Claude Code CLI exposes no model-list command — version-robust
    # aliases (sonnet/opus/haiku) are the host's correct fallback.
    if p == "claude_code_cli":
        return ModelDiscovery(
            provider=p, source="unavailable", error="cli has no model-list command"
        )

    try:
        if p == "ollama":
            root = _ollama_native_root(base_url or "http://localhost:11434/v1")
            res = await _get(f"{root}/api/tags", {}, transport, timeout)
            parse = _parse_ollama_tags
        elif p in _OPENAI_COMPATIBLE:
            base = (base_url or _CLOUD_BASE.get(p) or "").rstrip("/")
            if not base:
                return ModelDiscovery(provider=p, error="no base_url")
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            res = await _get(f"{base}/models", headers, transport, timeout)
            parse = _parse_openai_models
        elif p == "anthropic":
            if not api_key:
                return ModelDiscovery(provider=p, error="no api_key")
            base = (base_url or _CLOUD_BASE["anthropic"]).rstrip("/")
            headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
            res = await _get(f"{base}/models?limit=1000", headers, transport, timeout)
            parse = _parse_anthropic_models
        elif p == "google":
            if not api_key:
                return ModelDiscovery(provider=p, error="no api_key")
            base = (base_url or _CLOUD_BASE["google"]).rstrip("/")
            res = await _get(
                f"{base}/models?pageSize=1000&key={api_key}", {}, transport, timeout
            )
            parse = _parse_google_models
        else:
            return ModelDiscovery(provider=p, error=f"unsupported provider {p!r}")
    except Exception as exc:  # noqa: BLE001 — best-effort
        logger.debug("model discovery dispatch failed for %s: %s", p, exc)
        return ModelDiscovery(provider=p, error=str(exc))

    if res is None:
        return ModelDiscovery(provider=p, error="unreachable")
    if res.status != 200:
        return ModelDiscovery(provider=p, error=f"http {res.status}")

    models = parse(res.json)
    if not models:
        return ModelDiscovery(provider=p, error="no models in response")
    return ModelDiscovery(provider=p, models=models, source="live")


__all__ = ["ModelInfo", "ModelDiscovery", "discover_models", "Transport"]
