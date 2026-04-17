"""Default artifact persistence for Stage 15: Memory."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List

from geny_executor.stages.s15_memory.interface import ConversationPersistence


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


class NullPersistence(ConversationPersistence):
    """Null persistence — accepts writes but stores nothing.

    Used as the default when no persistence backend is configured,
    so every :class:`MemoryStage` has a non-None persistence slot.
    """

    @property
    def name(self) -> str:
        return "null"

    @property
    def description(self) -> str:
        return "No-op persistence (default when memory is not persisted)"

    async def save(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        return None

    async def load(self, session_id: str) -> List[Dict[str, Any]]:
        return []

    async def clear(self, session_id: str) -> None:
        return None


class FilePersistence(ConversationPersistence):
    """File-based JSON persistence."""

    def __init__(self, base_dir: str = "./memory"):
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
        fd, tmp_path = tempfile.mkstemp(dir=self._base_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(messages, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

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
