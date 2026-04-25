"""GenyPersistence — Geny-compatible conversation persistence.

Implements the ConversationPersistence interface (S15 Memory) using
Geny's SessionMemoryManager for JSONL transcript + optional DB dual-write.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from geny_executor.stages.s18_memory.interface import ConversationPersistence

logger = logging.getLogger(__name__)


class GenyPersistence(ConversationPersistence):
    """Geny-compatible conversation persistence via SessionMemoryManager.

    Delegates to the memory manager's short-term memory subsystem,
    which handles JSONL file storage and optional DB dual-write.

    Args:
        memory_manager: Geny's SessionMemoryManager (or duck-typed equivalent).
    """

    def __init__(self, memory_manager: Any):
        self._mgr = memory_manager

    @property
    def name(self) -> str:
        return "geny_persistence"

    @property
    def description(self) -> str:
        return "Geny-compatible persistence via SessionMemoryManager"

    async def save(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Save messages by recording them through the memory manager.

        The memory manager's short-term memory handles the actual
        storage (JSONL file + optional DB dual-write).

        Note: We don't re-record all messages each time — we rely on
        GenyMemoryStrategy to record messages incrementally during
        execution. This method serves as a final consistency checkpoint.
        """
        if not self._mgr:
            return

        try:
            stm = getattr(self._mgr, "short_term", None)
            if stm is None:
                return

            # Write a session summary if we have enough messages
            if len(messages) >= 4:
                self._update_summary(messages)

        except Exception:
            logger.debug(
                "geny_persistence: save failed for session %s",
                session_id,
                exc_info=True,
            )

    async def load(self, session_id: str) -> List[Dict[str, Any]]:
        """Load messages from the memory manager's short-term memory.

        Returns messages in Anthropic API format.
        """
        if not self._mgr:
            return []

        try:
            stm = getattr(self._mgr, "short_term", None)
            if stm is None:
                return []

            entries = stm.load_all()
            messages: List[Dict[str, Any]] = []

            for entry in entries:
                meta = getattr(entry, "metadata", {}) or {}
                role = meta.get("role", "")
                content = getattr(entry, "content", "")

                if role in ("user", "assistant") and content:
                    # Strip the "[role] " prefix that STM adds
                    if content.startswith(f"[{role}] "):
                        content = content[len(f"[{role}] ") :]
                    messages.append({"role": role, "content": content})

            return messages

        except Exception:
            logger.debug(
                "geny_persistence: load failed for session %s",
                session_id,
                exc_info=True,
            )
            return []

    async def clear(self, session_id: str) -> None:
        """Clear is a no-op — Geny's memory is append-only by design."""
        logger.debug("geny_persistence: clear called for session %s (no-op)", session_id)

    # ── Internal ─────────────────────────────────────────────────────

    def _update_summary(self, messages: List[Dict[str, Any]]) -> None:
        """Build and write a lightweight session summary."""
        try:
            stm = getattr(self._mgr, "short_term", None)
            if stm is None:
                return

            # Build a concise summary from the last few exchanges
            recent = messages[-6:]  # last 3 exchanges
            lines = []
            for msg in recent:
                role = msg.get("role", "?")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                preview = str(content)[:200]
                lines.append(f"- **{role}**: {preview}")

            summary = "## Session Summary\n\n" + "\n".join(lines)
            stm.write_summary(summary)

        except Exception:
            logger.debug("geny_persistence: summary update failed", exc_info=True)
