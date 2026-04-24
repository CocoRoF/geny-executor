"""Memory retrievers — concrete implementations for loading memory into context."""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

from geny_executor.core.state import PipelineState
from geny_executor.stages.s02_context.interface import MemoryRetriever
from geny_executor.stages.s02_context.types import MemoryChunk

logger = logging.getLogger(__name__)


class NullRetriever(MemoryRetriever):
    """No memory retrieval."""

    @property
    def name(self) -> str:
        return "null"

    @property
    def description(self) -> str:
        return "No memory retrieval"

    async def retrieve(self, query: str, state: PipelineState) -> List[MemoryChunk]:
        return []


class StaticRetriever(MemoryRetriever):
    """Returns fixed memory chunks (useful for testing)."""

    def __init__(self, chunks: Optional[List[MemoryChunk]] = None):
        self._chunks = chunks or []

    @property
    def name(self) -> str:
        return "static"

    @property
    def description(self) -> str:
        return "Returns fixed memory chunks"

    def add_chunk(self, key: str, content: str, **kwargs: Any) -> None:
        """Append a new MemoryChunk to the fixed chunk list."""
        self._chunks.append(MemoryChunk(key=key, content=content, **kwargs))

    async def retrieve(self, query: str, state: PipelineState) -> List[MemoryChunk]:
        return list(self._chunks)


# Default cap on resources fetched per retrieve() call. Resource bodies
# land in the system prompt context — pulling everything every turn
# would torch the LLM's context budget.
_DEFAULT_MAX_RESOURCES = 5


class MCPResourceRetriever(MemoryRetriever):
    """Pull MCP server *resources* (the second MCP primitive) as memory chunks.

    Cycle 20260424 executor uplift — Phase 7 Sprint S7.2.

    For every CONNECTED MCP server in the bound manager:

    1. Call ``list_resources`` to enumerate available URIs.
    2. Filter the list against the query (substring match across
       ``uri`` / ``name`` / ``description``; case-insensitive). When
       the query is empty, every entry passes — useful for
       always-attached reference material.
    3. Optionally apply a host-supplied ``filter_fn`` for richer
       selection (e.g. "only file:// URIs", "only mimeType ==
       application/json").
    4. Cap the survivors at ``max_resources``.
    5. Read the body of each survivor via ``read_resource`` and wrap
       as a :class:`MemoryChunk` with ``source="mcp_resource"`` and
       ``metadata={"server", "uri", "name", "mimeType"}``.

    Failure isolation: per-server ``list_resources`` failures and
    per-URI ``read_resource`` failures are logged at WARNING and
    skipped — one broken server cannot prevent the rest from
    contributing context.

    Pair with ``Pipeline.attach_runtime(memory_retriever=...)`` to
    install at session start.
    """

    def __init__(
        self,
        manager: Any,
        *,
        max_resources: int = _DEFAULT_MAX_RESOURCES,
        filter_fn: Optional[Callable[[dict], bool]] = None,
    ):
        self._manager = manager
        self._max_resources = max(1, int(max_resources))
        self._filter_fn = filter_fn

    @property
    def name(self) -> str:
        return "mcp_resource"

    @property
    def description(self) -> str:
        return f"MCP resources (cap {self._max_resources})"

    @property
    def max_resources(self) -> int:
        return self._max_resources

    async def retrieve(self, query: str, state: PipelineState) -> List[MemoryChunk]:
        servers = getattr(self._manager, "_servers", {}) or {}
        if not servers:
            return []

        q = (query or "").strip().lower()
        chunks: List[MemoryChunk] = []
        budget_remaining = self._max_resources

        for server_name, conn in servers.items():
            if budget_remaining <= 0:
                break
            if not getattr(conn, "is_connected", False):
                continue
            try:
                resources = await conn.list_resources()
            except Exception:
                logger.warning(
                    "MCPResourceRetriever: list_resources crashed on %r",
                    server_name,
                    exc_info=True,
                )
                continue

            for raw in resources:
                if budget_remaining <= 0:
                    break
                if not self._matches(raw, q):
                    continue
                if self._filter_fn is not None:
                    try:
                        if not self._filter_fn(raw):
                            continue
                    except Exception:
                        logger.warning(
                            "MCPResourceRetriever: filter_fn raised for %r",
                            raw.get("uri"),
                            exc_info=True,
                        )
                        continue

                uri = raw.get("uri", "")
                if not uri:
                    continue
                try:
                    body = await conn.read_resource(uri)
                except Exception:
                    logger.warning(
                        "MCPResourceRetriever: read_resource crashed on %s/%s",
                        server_name,
                        uri,
                        exc_info=True,
                    )
                    continue
                if body is None:
                    continue

                chunks.append(
                    MemoryChunk(
                        key=uri,
                        content=body,
                        source="mcp_resource",
                        metadata={
                            "server": server_name,
                            "uri": uri,
                            "name": raw.get("name", ""),
                            "mimeType": raw.get("mimeType", ""),
                        },
                    )
                )
                budget_remaining -= 1

        return chunks

    @staticmethod
    def _matches(raw: dict, query_lower: str) -> bool:
        """Case-insensitive substring match on name / uri / description.

        Empty query matches everything — lets hosts attach the
        retriever for static reference material that's always
        relevant.
        """
        if not query_lower:
            return True
        haystack = " ".join(str(raw.get(k, "")).lower() for k in ("uri", "name", "description"))
        return query_lower in haystack
