"""Conversation persistence — Level 2 strategies for storing history."""

from __future__ import annotations

import json
import os
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from geny_executor.core.stage import Strategy


class ConversationPersistence(Strategy):
    """Base interface for conversation persistence."""

    @abstractmethod
    async def save(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Save messages for a session."""
        ...

    @abstractmethod
    async def load(self, session_id: str) -> List[Dict[str, Any]]:
        """Load messages for a session."""
        ...

    @abstractmethod
    async def clear(self, session_id: str) -> None:
        """Clear messages for a session."""
        ...


class InMemoryPersistence(ConversationPersistence):
    """In-memory persistence (testing, ephemeral sessions)."""

    def __init__(self) -> None:
        self._store: Dict[str, List[Dict[str, Any]]] = {}

    @property
    def name(self) -> str:
        return "in_memory"

    @property
    def description(self) -> str:
        return "In-memory conversation storage"

    async def save(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        self._store[session_id] = list(messages)

    async def load(self, session_id: str) -> List[Dict[str, Any]]:
        return list(self._store.get(session_id, []))

    async def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)


class FilePersistence(ConversationPersistence):
    """File-based JSON persistence."""

    def __init__(self, base_dir: str):
        self._base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    @property
    def name(self) -> str:
        return "file"

    @property
    def description(self) -> str:
        return f"File-based persistence at {self._base_dir}"

    def _path(self, session_id: str) -> str:
        safe_id = session_id.replace("/", "_").replace("\\", "_")
        return os.path.join(self._base_dir, f"{safe_id}.json")

    async def save(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        path = self._path(session_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)

    async def load(self, session_id: str) -> List[Dict[str, Any]]:
        path = self._path(session_id)
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def clear(self, session_id: str) -> None:
        path = self._path(session_id)
        if os.path.exists(path):
            os.remove(path)
