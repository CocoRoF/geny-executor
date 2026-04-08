"""Memory retrievers — Level 2 strategies for loading memory into context."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from geny_executor.core.stage import Strategy
from geny_executor.core.state import PipelineState


@dataclass
class MemoryChunk:
    """A piece of retrieved memory."""

    key: str
    content: str
    source: str = ""  # "long_term", "short_term", "vector", "file"
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryRetriever(Strategy):
    """Base interface for memory retrieval."""

    @abstractmethod
    async def retrieve(self, query: str, state: PipelineState) -> List[MemoryChunk]:
        """Retrieve relevant memory chunks for a query."""
        ...


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
        self._chunks.append(MemoryChunk(key=key, content=content, **kwargs))

    async def retrieve(self, query: str, state: PipelineState) -> List[MemoryChunk]:
        return list(self._chunks)
