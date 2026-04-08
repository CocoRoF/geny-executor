"""Session — pipeline + state execution unit."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional

from geny_executor.core.pipeline import Pipeline
from geny_executor.core.config import PipelineConfig
from geny_executor.core.result import PipelineResult
from geny_executor.core.state import PipelineState
from geny_executor.events.types import PipelineEvent
from geny_executor.session.freshness import FreshnessPolicy, FreshnessStatus


class Session:
    """Agent session — Pipeline + State execution unit.

    Manages state persistence across multiple run() calls.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        pipeline: Optional[Pipeline] = None,
        config: Optional[PipelineConfig] = None,
        freshness_policy: Optional[FreshnessPolicy] = None,
    ):
        self.id = session_id or uuid.uuid4().hex[:12]
        self._pipeline = pipeline or Pipeline(config)
        self._state = PipelineState(session_id=self.id)
        self._freshness = freshness_policy or FreshnessPolicy()
        self._created_at = datetime.now(timezone.utc)
        self._last_active = self._created_at

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def freshness(self) -> FreshnessStatus:
        return self._freshness.evaluate(
            self._created_at,
            self._last_active,
            len(self._state.messages),
        )

    async def run(self, input: Any) -> PipelineResult:
        """Execute input through the pipeline, preserving state."""
        self._last_active = datetime.now(timezone.utc)
        result = await self._pipeline.run(input, self._state)
        self._last_active = datetime.now(timezone.utc)
        return result

    async def run_stream(self, input: Any) -> AsyncIterator[PipelineEvent]:
        """Streaming execution."""
        self._last_active = datetime.now(timezone.utc)
        async for event in self._pipeline.run_stream(input, self._state):
            yield event
        self._last_active = datetime.now(timezone.utc)

    def reset_state(self) -> None:
        """Reset state for a fresh start."""
        self._state = PipelineState(session_id=self.id)
        self._last_active = datetime.now(timezone.utc)
