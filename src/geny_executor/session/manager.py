"""Session manager — CRUD and lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from geny_executor.core.pipeline import Pipeline
from geny_executor.core.config import PipelineConfig
from geny_executor.session.session import Session
from geny_executor.session.freshness import FreshnessPolicy


@dataclass
class SessionInfo:
    """Lightweight session metadata."""

    session_id: str
    freshness: str
    message_count: int
    iteration: int
    total_cost_usd: float


class SessionManager:
    """Session CRUD and lifecycle management."""

    def __init__(
        self,
        default_config: Optional[PipelineConfig] = None,
        freshness_policy: Optional[FreshnessPolicy] = None,
    ):
        self._sessions: Dict[str, Session] = {}
        self._default_config = default_config or PipelineConfig()
        self._freshness = freshness_policy or FreshnessPolicy()

    def create(
        self,
        pipeline: Pipeline,
        session_id: Optional[str] = None,
    ) -> Session:
        """Create a new session."""
        session = Session(
            session_id=session_id,
            pipeline=pipeline,
            freshness_policy=self._freshness,
        )
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        """Delete a session."""
        return self._sessions.pop(session_id, None) is not None

    def list_sessions(self) -> List[SessionInfo]:
        """List all sessions with metadata."""
        return [
            SessionInfo(
                session_id=s.id,
                freshness=s.freshness.value,
                message_count=len(s.state.messages),
                iteration=s.state.iteration,
                total_cost_usd=s.state.total_cost_usd,
            )
            for s in self._sessions.values()
        ]

    def __len__(self) -> int:
        return len(self._sessions)
