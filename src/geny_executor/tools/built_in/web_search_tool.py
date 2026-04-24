"""WebSearch — DuckDuckGo text search for URLs + snippets.

Cycle 20260424 executor uplift — Phase 3 Week 5.

Issues a text search query against DuckDuckGo (via the ``ddgs``
package) and returns a compact, LLM-friendly list of result headlines,
URLs, and snippets. Intended as a companion to ``WebFetch`` — the LLM
uses ``WebSearch`` to discover candidate URLs and then pulls the
interesting ones through ``WebFetch`` for details.

Dependencies: ``ddgs>=9.11`` ships as an optional ``[web]`` extra so
the core executor footprint stays small for hosts that don't need web
search. When the package is unavailable, ``execute`` returns a
``ToolResult(is_error=True)`` with an installation hint — the pipeline
keeps moving instead of crashing at import time.

Capabilities: ``concurrency_safe=True`` + ``read_only=True`` +
``network_egress=True``. Searches are idempotent-ish on the scale of
a single turn (DDG rate-limits heavy fan-out), so the orchestrator is
free to run WebSearch in parallel with Read/Grep/Glob reads.

See ``executor_uplift/06_design_tool_system.md`` §7 and
``executor_uplift/12_detailed_plan.md`` §3 (Week 4-5 Web 계열).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from geny_executor.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RESULTS = 10
_HARD_MAX_RESULTS = 30


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


class WebSearchTool(Tool):
    """Search the web and return ranked hits with title / URL / snippet.

    Usage pattern: pair with ``WebFetch`` — call WebSearch once to
    surface candidate URLs, inspect the snippets, then issue WebFetch
    on the ones worth reading.
    """

    @property
    def name(self) -> str:
        return "WebSearch"

    @property
    def description(self) -> str:
        return (
            "Search the web via DuckDuckGo and return ranked results "
            "(title, URL, snippet). Pair with WebFetch to read a "
            "specific result's contents. Limit is capped at "
            f"{_HARD_MAX_RESULTS} to keep output LLM-friendly."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query.",
                    "minLength": 1,
                },
                "max_results": {
                    "type": "integer",
                    "description": (
                        f"Maximum number of results to return. "
                        f"Default {_DEFAULT_MAX_RESULTS}, hard cap "
                        f"{_HARD_MAX_RESULTS}."
                    ),
                    "exclusiveMinimum": 0,
                },
                "region": {
                    "type": "string",
                    "description": (
                        "Optional region code (e.g. 'us-en', 'kr-kr'). "
                        "Defaults to 'wt-wt' (worldwide, English)."
                    ),
                },
                "safesearch": {
                    "type": "string",
                    "description": "Safe-search strictness: 'on' | 'moderate' | 'off'.",
                    "enum": ["on", "moderate", "off"],
                },
            },
            "required": ["query"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(
            concurrency_safe=True,
            read_only=True,
            idempotent=False,  # ranking may drift between calls
            network_egress=True,
        )

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        query = (input.get("query") or "").strip()
        if not query:
            return ToolResult(content="query must not be empty", is_error=True)

        max_results = int(input.get("max_results", _DEFAULT_MAX_RESULTS))
        max_results = max(1, min(_HARD_MAX_RESULTS, max_results))
        region = input.get("region") or "wt-wt"
        safesearch = input.get("safesearch") or "moderate"

        ddgs_cls = _load_ddgs()
        if ddgs_cls is None:
            return ToolResult(
                content=(
                    "WebSearch requires the 'ddgs' package. Install the "
                    "executor's [web] extra:\n"
                    "    pip install 'geny-executor[web]'\n"
                    "or pin ddgs directly:\n"
                    "    pip install 'ddgs>=9.11'"
                ),
                is_error=True,
            )

        # DDGS is blocking; push it to a worker thread so we don't
        # stall the event loop.
        try:
            raw = await asyncio.to_thread(
                self._search_sync,
                ddgs_cls,
                query,
                max_results,
                region,
                safesearch,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("WebSearch DDG call failed")
            return ToolResult(
                content=f"web search failed: {exc}",
                is_error=True,
            )

        if not raw:
            return ToolResult(
                content=f"No results for {query!r}.",
                metadata={"query": query, "results_count": 0},
            )

        hits = [self._normalise_hit(i, r) for i, r in enumerate(raw[:max_results])]
        header = f"Search results for {query!r} ({len(hits)} of max {max_results}):"
        body = "\n\n".join(self._format_hit(h) for h in hits)
        return ToolResult(
            content=f"{header}\n\n{body}",
            metadata={
                "query": query,
                "results_count": len(hits),
                "results": hits,
            },
        )

    @staticmethod
    def _search_sync(
        ddgs_cls: Any,
        query: str,
        max_results: int,
        region: str,
        safesearch: str,
    ) -> List[Dict[str, Any]]:
        """Blocking body — runs inside ``asyncio.to_thread``.

        Kept as a static method so tests can monkey-patch it cleanly
        without also having to patch the asyncio wrapper.
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

    @staticmethod
    def _normalise_hit(index: int, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Map ddgs's result dict to a stable shape.

        ddgs returns ``title`` / ``href`` / ``body`` — we rename ``href``
        to ``url`` and ``body`` to ``snippet`` to match the more common
        search-API conventions. ``rank`` is the zero-based position.
        """
        return {
            "rank": index,
            "title": str(raw.get("title") or "").strip(),
            "url": str(raw.get("href") or raw.get("url") or "").strip(),
            "snippet": str(raw.get("body") or raw.get("snippet") or "").strip(),
        }

    @staticmethod
    def _format_hit(hit: Dict[str, Any]) -> str:
        rank = hit.get("rank", 0) + 1
        title = hit.get("title") or "(no title)"
        url = hit.get("url") or "(no url)"
        snippet = hit.get("snippet") or ""
        if snippet:
            return f"{rank}. {title}\n   {url}\n   {snippet}"
        return f"{rank}. {title}\n   {url}"
