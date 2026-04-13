"""Geny-compatible memory integration.

Provides Strategy implementations that bridge geny-executor's pipeline
with Geny's multi-layer memory system (SessionMemoryManager).

Usage::

    from geny_executor.memory import (
        GenyMemoryRetriever,
        GenyMemoryStrategy,
        GenyPersistence,
    )

    retriever = GenyMemoryRetriever(memory_manager)
    strategy = GenyMemoryStrategy(memory_manager)
    persistence = GenyPersistence(memory_manager)

    pipeline = (
        PipelineBuilder("agent", api_key="...")
        .with_context(retriever=retriever)
        .with_memory(strategy=strategy, persistence=persistence)
        .build()
    )
"""

from geny_executor.memory.retriever import GenyMemoryRetriever
from geny_executor.memory.strategy import GenyMemoryStrategy
from geny_executor.memory.persistence import GenyPersistence
from geny_executor.memory.presets import GenyPresets

__all__ = [
    "GenyMemoryRetriever",
    "GenyMemoryStrategy",
    "GenyPersistence",
    "GenyPresets",
]
