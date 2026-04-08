"""Pipeline execution result."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from geny_executor.core.state import PipelineState, TokenUsage, CacheMetrics


@dataclass
class PipelineResult:
    """Final result of a pipeline execution."""

    # Output
    text: str = ""
    output: Optional[Any] = None

    # Execution summary
    success: bool = True
    error: Optional[str] = None
    iterations: int = 0

    # Token & Cost
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    turn_token_usage: List[TokenUsage] = field(default_factory=list)
    total_cost_usd: float = 0.0
    cache_metrics: CacheMetrics = field(default_factory=CacheMetrics)

    # Thinking
    thinking_history: List[Dict[str, Any]] = field(default_factory=list)

    # Events
    events: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    session_id: str = ""
    pipeline_id: str = ""
    model: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_state(cls, state: PipelineState) -> PipelineResult:
        """Create a result from final pipeline state."""
        return cls(
            text=state.final_text,
            output=state.final_output,
            success=state.loop_decision != "error",
            iterations=state.iteration,
            token_usage=state.token_usage,
            turn_token_usage=list(state.turn_token_usage),
            total_cost_usd=state.total_cost_usd,
            cache_metrics=state.cache_metrics,
            thinking_history=list(state.thinking_history),
            events=list(state.events),
            session_id=state.session_id,
            pipeline_id=state.pipeline_id,
            model=state.model,
            metadata=dict(state.metadata),
        )

    @classmethod
    def error_result(cls, error: str, state: Optional[PipelineState] = None) -> PipelineResult:
        """Create an error result."""
        if state:
            result = cls.from_state(state)
            result.success = False
            result.error = error
            return result
        return cls(success=False, error=error)
