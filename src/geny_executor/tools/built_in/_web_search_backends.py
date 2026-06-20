"""Pluggable search backends for the ``WebSearch`` built-in tool.

The default backend remains DuckDuckGo (``ddg``) via the optional
``ddgs`` package, so the legacy WebSearch path is byte-for-byte
unchanged. Hosts that want a different provider can switch backends
without touching the tool surface — selection happens in
:meth:`WebSearchTool.execute` (input param ``backend`` > extras >
``GENY_WEBSEARCH_BACKEND`` env > ``"ddg"``).

Every backend implements the async :class:`WebSearchBackend` protocol
and returns *normalized hits* — the exact shape
:meth:`WebSearchTool._normalise_hit` produces (``rank`` / ``title`` /
``url`` / ``snippet``) — so the tool's formatting + metadata stay
identical regardless of provider.

Non-ddg backends speak HTTP through ``httpx`` (already a hard
dependency, used by ``WebFetch``), so adding them introduces **no new
required packages**. ``ddgs`` stays the optional ``[web]`` extra.

Config / credentials are read from ``ToolContext.extras["web_search"]``
first, then environment variables:

* ``brave``   — ``brave_api_key`` / ``BRAVE_SEARCH_API_KEY``
* ``tavily``  — ``tavily_api_key`` / ``TAVILY_API_KEY``
* ``searxng`` — ``searxng_url``   / ``SEARXNG_URL``

When a backend is missing its key/url, it raises
:class:`WebSearchConfigError` with a clear config hint; the tool turns
that into a ``ToolResult(is_error=True)``.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

import httpx

from geny_executor.tools.base import ToolContext

# Shared HTTP timeout for the API-backed backends (seconds).
_HTTP_TIMEOUT = 15.0


class WebSearchBackendError(Exception):
    """Base class for backend-level failures surfaced to the tool."""


class WebSearchConfigError(WebSearchBackendError):
    """Raised when a backend is missing required credentials / config.

    The message is a user-facing hint (which extras key / env var to
    set) and is rendered verbatim into ``ToolResult.content``.
    """


def _extras_web_search(context: ToolContext) -> Dict[str, Any]:
    """Return the ``web_search`` sub-dict from ``ctx.extras`` (or empty)."""
    raw = (context.extras or {}).get("web_search")
    return raw if isinstance(raw, dict) else {}


def _normalise_hit(index: int, raw: Dict[str, Any]) -> Dict[str, Any]:
    """Map a provider result dict to the stable WebSearch hit shape.

    Mirrors :meth:`WebSearchTool._normalise_hit`: ``rank`` is the
    zero-based position; ``href``/``url`` and ``body``/``snippet`` are
    accepted interchangeably so each backend can hand back whichever
    key its API uses.
    """
    return {
        "rank": index,
        "title": str(raw.get("title") or "").strip(),
        "url": str(raw.get("href") or raw.get("url") or "").strip(),
        "snippet": str(raw.get("body") or raw.get("snippet") or "").strip(),
    }


@runtime_checkable
class WebSearchBackend(Protocol):
    """Async search backend contract.

    Implementations return a list of normalized hits (the
    :func:`_normalise_hit` shape). ``name`` is the registry key used in
    the ``backend`` input enum + selection precedence.
    """

    name: str

    async def search(
        self,
        query: str,
        max_results: int,
        region: str,
        safesearch: str,
    ) -> List[Dict[str, Any]]:
        """Run ``query`` and return up to ``max_results`` normalized hits."""
        ...


# ─────────────────────────────────────────────────────────────────
# ddg — default backend, wraps the existing ddgs logic
# ─────────────────────────────────────────────────────────────────


def _load_ddgs() -> Optional[Any]:
    """Return the ``DDGS`` class or ``None`` if ``ddgs`` is not installed.

    Imported lazily so core hosts don't pay the startup cost of
    ``ddgs`` + ``primp`` + ``lxml`` unless WebSearch is actually used.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        return None
    return DDGS


class DdgBackend:
    """DuckDuckGo backend via the optional ``ddgs`` package.

    This is the default and preserves the legacy WebSearch behaviour
    exactly. ``ddgs`` is blocking, so the blocking body is pushed to a
    worker thread via :func:`asyncio.to_thread`.

    The DDGS-class loader and the blocking search body are injected by
    the tool (``load_ddgs`` / ``search_sync``) rather than referenced
    directly here, so existing hosts / tests that monkey-patch
    ``web_search_tool._load_ddgs`` or ``WebSearchTool._search_sync``
    continue to take effect through the indirection.
    """

    name = "ddg"

    def __init__(
        self,
        load_ddgs: Optional[Callable[[], Optional[Any]]] = None,
        search_sync: Optional[Callable[..., List[Dict[str, Any]]]] = None,
    ) -> None:
        self._load_ddgs = load_ddgs or _load_ddgs
        self._search_sync = search_sync or _default_ddg_search_sync

    async def search(
        self,
        query: str,
        max_results: int,
        region: str,
        safesearch: str,
    ) -> List[Dict[str, Any]]:
        ddgs_cls = self._load_ddgs()
        if ddgs_cls is None:
            raise WebSearchConfigError(
                "WebSearch requires the 'ddgs' package. Install the "
                "executor's [web] extra:\n"
                "    pip install 'geny-executor[web]'\n"
                "or pin ddgs directly:\n"
                "    pip install 'ddgs>=9.11'"
            )
        raw = await asyncio.to_thread(
            self._search_sync,
            ddgs_cls,
            query,
            max_results,
            region,
            safesearch,
        )
        return [_normalise_hit(i, r) for i, r in enumerate(raw[:max_results])]


def _default_ddg_search_sync(
    ddgs_cls: Any,
    query: str,
    max_results: int,
    region: str,
    safesearch: str,
) -> List[Dict[str, Any]]:
    """Blocking ddgs body — runs inside ``asyncio.to_thread``.

    Used only when the tool does not inject its own (back-compat)
    ``_search_sync``.
    """
    with ddgs_cls() as client:
        return list(
            client.text(
                query,
                region=region,
                safesearch=safesearch,
                max_results=max_results,
            )
        )


# ─────────────────────────────────────────────────────────────────
# brave — Brave Search API
# ─────────────────────────────────────────────────────────────────


class BraveBackend:
    """Brave Search API backend (``api.search.brave.com``).

    Key resolution: ``ctx.extras["web_search"]["brave_api_key"]`` then
    the ``BRAVE_SEARCH_API_KEY`` environment variable.
    """

    name = "brave"
    _ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, context: ToolContext) -> None:
        self._api_key = _extras_web_search(context).get("brave_api_key") or os.environ.get(
            "BRAVE_SEARCH_API_KEY"
        )

    async def search(
        self,
        query: str,
        max_results: int,
        region: str,
        safesearch: str,
    ) -> List[Dict[str, Any]]:
        if not self._api_key:
            raise WebSearchConfigError(
                "Brave backend requires an API key. Set "
                "extras['web_search']['brave_api_key'] or the "
                "BRAVE_SEARCH_API_KEY environment variable. Get a key at "
                "https://brave.com/search/api/."
            )
        # Brave caps web results at 20 per request.
        count = max(1, min(20, max_results))
        params: Dict[str, Any] = {"q": query, "count": count}
        # Brave safesearch vocabulary: off | moderate | strict.
        params["safesearch"] = "strict" if safesearch == "on" else safesearch
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self._api_key,
        }
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.get(self._ENDPOINT, params=params, headers=headers)
        if resp.status_code == 401 or resp.status_code == 403:
            raise WebSearchConfigError(
                f"Brave rejected the API key (HTTP {resp.status_code}). "
                "Check extras['web_search']['brave_api_key'] / "
                "BRAVE_SEARCH_API_KEY."
            )
        resp.raise_for_status()
        data = resp.json()
        results = ((data or {}).get("web") or {}).get("results") or []
        hits: List[Dict[str, Any]] = []
        for i, r in enumerate(results[:max_results]):
            hits.append(
                _normalise_hit(
                    i,
                    {
                        "title": r.get("title"),
                        "url": r.get("url"),
                        "snippet": r.get("description"),
                    },
                )
            )
        return hits


# ─────────────────────────────────────────────────────────────────
# tavily — Tavily Search API
# ─────────────────────────────────────────────────────────────────


class TavilyBackend:
    """Tavily Search API backend (``api.tavily.com``).

    Key resolution: ``ctx.extras["web_search"]["tavily_api_key"]`` then
    the ``TAVILY_API_KEY`` environment variable.
    """

    name = "tavily"
    _ENDPOINT = "https://api.tavily.com/search"

    def __init__(self, context: ToolContext) -> None:
        self._api_key = _extras_web_search(context).get("tavily_api_key") or os.environ.get(
            "TAVILY_API_KEY"
        )

    async def search(
        self,
        query: str,
        max_results: int,
        region: str,
        safesearch: str,
    ) -> List[Dict[str, Any]]:
        if not self._api_key:
            raise WebSearchConfigError(
                "Tavily backend requires an API key. Set "
                "extras['web_search']['tavily_api_key'] or the "
                "TAVILY_API_KEY environment variable. Get a key at "
                "https://tavily.com/."
            )
        payload: Dict[str, Any] = {
            "api_key": self._api_key,
            "query": query,
            "max_results": max(1, min(20, max_results)),
        }
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.post(self._ENDPOINT, json=payload)
        if resp.status_code == 401 or resp.status_code == 403:
            raise WebSearchConfigError(
                f"Tavily rejected the API key (HTTP {resp.status_code}). "
                "Check extras['web_search']['tavily_api_key'] / "
                "TAVILY_API_KEY."
            )
        resp.raise_for_status()
        data = resp.json()
        results = (data or {}).get("results") or []
        hits: List[Dict[str, Any]] = []
        for i, r in enumerate(results[:max_results]):
            hits.append(
                _normalise_hit(
                    i,
                    {
                        "title": r.get("title"),
                        "url": r.get("url"),
                        "snippet": r.get("content"),
                    },
                )
            )
        return hits


# ─────────────────────────────────────────────────────────────────
# searxng — self-hosted SearXNG JSON API
# ─────────────────────────────────────────────────────────────────


class SearxngBackend:
    """SearXNG JSON API backend (self-hosted meta-search).

    Base URL resolution: ``ctx.extras["web_search"]["searxng_url"]``
    then the ``SEARXNG_URL`` environment variable. The instance must
    have the JSON output format enabled.
    """

    name = "searxng"

    def __init__(self, context: ToolContext) -> None:
        base = _extras_web_search(context).get("searxng_url") or os.environ.get("SEARXNG_URL")
        self._base = base.rstrip("/") if isinstance(base, str) and base else None

    async def search(
        self,
        query: str,
        max_results: int,
        region: str,
        safesearch: str,
    ) -> List[Dict[str, Any]]:
        if not self._base:
            raise WebSearchConfigError(
                "SearXNG backend requires a base URL. Set "
                "extras['web_search']['searxng_url'] or the SEARXNG_URL "
                "environment variable (e.g. 'https://searx.example.org')."
            )
        # SearXNG safesearch is numeric: 0 off / 1 moderate / 2 strict.
        safe_map = {"off": 0, "moderate": 1, "on": 2}
        params: Dict[str, Any] = {
            "q": query,
            "format": "json",
            "safesearch": safe_map.get(safesearch, 1),
        }
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.get(f"{self._base}/search", params=params)
        resp.raise_for_status()
        data = resp.json()
        results = (data or {}).get("results") or []
        hits: List[Dict[str, Any]] = []
        for i, r in enumerate(results[:max_results]):
            hits.append(
                _normalise_hit(
                    i,
                    {
                        "title": r.get("title"),
                        "url": r.get("url"),
                        "snippet": r.get("content"),
                    },
                )
            )
        return hits


# ─────────────────────────────────────────────────────────────────
# Registry + factory
# ─────────────────────────────────────────────────────────────────

#: Valid backend names, in a stable order (also used for the input enum
#: and the "valid options" hint in error messages).
BACKEND_NAMES: tuple[str, ...] = ("ddg", "brave", "tavily", "searxng")

DEFAULT_BACKEND = "ddg"


def build_backend(
    name: str,
    context: ToolContext,
    *,
    ddg_load_ddgs: Optional[Callable[[], Optional[Any]]] = None,
    ddg_search_sync: Optional[Callable[..., List[Dict[str, Any]]]] = None,
) -> WebSearchBackend:
    """Instantiate the backend ``name`` bound to ``context``.

    Raises :class:`WebSearchConfigError` for an unknown name so the
    tool surfaces a clear "valid options" hint. The API backends
    capture their credentials at construction time. ``ddg_load_ddgs`` /
    ``ddg_search_sync`` let the tool inject its own (monkey-patchable)
    DDGS hooks for backward compatibility.
    """
    if name == "ddg":
        return DdgBackend(load_ddgs=ddg_load_ddgs, search_sync=ddg_search_sync)
    if name == "brave":
        return BraveBackend(context)
    if name == "tavily":
        return TavilyBackend(context)
    if name == "searxng":
        return SearxngBackend(context)
    raise WebSearchConfigError(
        f"Unknown WebSearch backend {name!r}. Valid options: {', '.join(BACKEND_NAMES)}."
    )


def select_backend_name(input: Dict[str, Any], context: ToolContext) -> str:
    """Resolve the backend name by precedence.

    Precedence (highest first):
    ``input['backend']`` > ``ctx.extras['web_search']['backend']`` >
    ``GENY_WEBSEARCH_BACKEND`` env > ``"ddg"``.
    """
    candidate = (
        input.get("backend")
        or _extras_web_search(context).get("backend")
        or os.environ.get("GENY_WEBSEARCH_BACKEND")
        or DEFAULT_BACKEND
    )
    return str(candidate).strip().lower()
