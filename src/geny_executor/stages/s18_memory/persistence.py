"""Conversation persistence — backward-compatible re-exports."""

from geny_executor.stages.s18_memory.interface import ConversationPersistence
from geny_executor.stages.s18_memory.artifact.default.persistence import (
    InMemoryPersistence,
    FilePersistence,
)

__all__ = [
    "ConversationPersistence",
    "InMemoryPersistence",
    "FilePersistence",
]
