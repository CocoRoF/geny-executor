"""Session persistence — save/restore pipeline state to disk.

Provides file-based persistence for PipelineState, enabling session
resumption across process restarts (equivalent to CLI --resume).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from geny_executor.core.state import PipelineState, TokenUsage, CacheMetrics

logger = logging.getLogger(__name__)

_STATE_FILENAME = ".pipeline_state.json"


class FileSessionPersistence:
    """File-based session state persistence.

    Saves PipelineState to ``{storage_root}/{session_id}/.pipeline_state.json``.
    Enables session resumption after server restarts.

    Usage:
        persistence = FileSessionPersistence("/data/sessions")

        # Save after execution
        persistence.save(session_id, state)

        # Resume on restart
        state = persistence.load(session_id)
    """

    def __init__(self, storage_root: str) -> None:
        self.storage_root = Path(storage_root)

    def _state_path(self, session_id: str) -> Path:
        return self.storage_root / session_id / _STATE_FILENAME

    def save(self, session_id: str, state: PipelineState) -> None:
        """Save pipeline state to disk.

        Only persists the fields needed for resumption:
        - messages (conversation history)
        - system prompt
        - token usage + cost
        - iteration count
        - memory refs
        - metadata
        """
        path = self._state_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        data: Dict[str, Any] = {
            "version": 1,
            "session_id": session_id,
            "saved_at": datetime.now(timezone.utc).isoformat(),

            # Conversation state (essential for --resume equivalent)
            "system": state.system,
            "messages": state.messages,

            # Execution stats
            "iteration": state.iteration,
            "total_cost_usd": state.total_cost_usd,

            # Token tracking
            "token_usage": {
                "input_tokens": state.token_usage.input_tokens,
                "output_tokens": state.token_usage.output_tokens,
                "cache_creation_input_tokens": state.token_usage.cache_creation_input_tokens,
                "cache_read_input_tokens": state.token_usage.cache_read_input_tokens,
            },

            # Cache metrics
            "cache_metrics": {
                "total_cache_writes": state.cache_metrics.total_cache_writes,
                "total_cache_reads": state.cache_metrics.total_cache_reads,
                "estimated_savings_usd": state.cache_metrics.estimated_savings_usd,
                "cache_hit_rate": state.cache_metrics.cache_hit_rate,
            },

            # Memory references
            "memory_refs": state.memory_refs,

            # Model config snapshot
            "model": state.model,
            "max_tokens": state.max_tokens,

            # Metadata
            "metadata": state.metadata,
        }

        try:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.debug(
                "Session state saved: %s (%d messages, cost=$%.6f)",
                session_id, len(state.messages), state.total_cost_usd,
            )
        except OSError as e:
            logger.error("Failed to save session state for %s: %s", session_id, e)

    def load(self, session_id: str) -> Optional[PipelineState]:
        """Load pipeline state from disk.

        Returns None if no saved state exists.
        """
        path = self._state_path(session_id)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to load session state for %s: %s", session_id, e)
            return None

        version = data.get("version", 0)
        if version != 1:
            logger.warning("Unknown state version %s for %s, skipping", version, session_id)
            return None

        state = PipelineState(session_id=session_id)

        # Restore conversation
        state.system = data.get("system", "")
        state.messages = data.get("messages", [])

        # Restore stats
        state.iteration = data.get("iteration", 0)
        state.total_cost_usd = data.get("total_cost_usd", 0.0)

        # Restore token usage
        tu = data.get("token_usage", {})
        state.token_usage = TokenUsage(
            input_tokens=tu.get("input_tokens", 0),
            output_tokens=tu.get("output_tokens", 0),
            cache_creation_input_tokens=tu.get("cache_creation_input_tokens", 0),
            cache_read_input_tokens=tu.get("cache_read_input_tokens", 0),
        )

        # Restore cache metrics
        cm = data.get("cache_metrics", {})
        state.cache_metrics = CacheMetrics(
            total_cache_writes=cm.get("total_cache_writes", 0),
            total_cache_reads=cm.get("total_cache_reads", 0),
            estimated_savings_usd=cm.get("estimated_savings_usd", 0.0),
            cache_hit_rate=cm.get("cache_hit_rate", 0.0),
        )

        # Restore memory refs
        state.memory_refs = data.get("memory_refs", [])

        # Restore model config
        state.model = data.get("model", state.model)
        state.max_tokens = data.get("max_tokens", state.max_tokens)

        # Restore metadata
        state.metadata = data.get("metadata", {})

        logger.info(
            "Session state loaded: %s (%d messages, iteration=%d, cost=$%.6f)",
            session_id, len(state.messages), state.iteration, state.total_cost_usd,
        )

        return state

    def exists(self, session_id: str) -> bool:
        """Check if saved state exists for a session."""
        return self._state_path(session_id).exists()

    def delete(self, session_id: str) -> bool:
        """Delete saved state for a session."""
        path = self._state_path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False
