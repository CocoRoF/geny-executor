"""Memory retrievers — concrete implementations for loading memory into context."""

from __future__ import annotations

from typing import Any, List, Optional

from geny_executor.core.state import PipelineState
from geny_executor.stages.s02_context.interface import MemoryRetriever
from geny_executor.stages.s02_context.types import MemoryChunk


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
