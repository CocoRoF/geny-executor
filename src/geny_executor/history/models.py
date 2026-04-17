"""Data models for execution history, replay, monitoring, and cost analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Execution & Timing Records ───────────────────────────


@dataclass
class StageTimingRecord:
    """Single stage timing entry for persistence."""

    iteration: int
    stage_order: int
    stage_name: str
    started_at: str
    finished_at: str
    duration_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    was_cached: bool = False
    was_skipped: bool = False
    tool_name: Optional[str] = None
    tool_success: Optional[bool] = None
    tool_duration_ms: Optional[int] = None


@dataclass
class ToolCallRecord:
    """Single tool call entry for persistence."""

    iteration: int
    tool_name: str
    called_at: str
    input_json: Optional[str] = None
    output_text: Optional[str] = None
    is_error: bool = False
    duration_ms: int = 0


@dataclass
class ExecutionRecord:
    """Full execution record (from DB query)."""

    id: str
    session_id: str
    model: str
    user_input: str
    started_at: str
    status: str = "running"
    environment_id: Optional[str] = None
    finished_at: Optional[str] = None
    result_text: Optional[str] = None
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    iterations: int = 0
    tool_calls: int = 0
    thinking_tokens: int = 0
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_stage: Optional[int] = None
    duration_ms: int = 0
    tags: List[str] = field(default_factory=list)

    # Detail fields (populated by get_execution_detail)
    stage_timings: List[Dict[str, Any]] = field(default_factory=list)
    tool_call_records: List[Dict[str, Any]] = field(default_factory=list)


# ── Replay ───────────────────────────────────────────────


@dataclass
class ReplayEvent:
    """Event yielded during replay."""

    type: str  # "event" | "breakpoint" | "waiting"
    event: Optional[Dict[str, Any]] = None
    stage_order: Optional[int] = None


# ── Performance Waterfall ────────────────────────────────


@dataclass
class StageWaterfall:
    """Single stage bar in waterfall chart."""

    order: int
    name: str
    duration_ms: int
    was_cached: bool = False
    was_skipped: bool = False
    tokens: int = 0


@dataclass
class IterationWaterfall:
    """One iteration (loop) in waterfall."""

    iteration: int
    stages: List[StageWaterfall] = field(default_factory=list)


@dataclass
class WaterfallData:
    """Full waterfall chart data."""

    execution_id: str
    total_duration_ms: int
    iterations: List[IterationWaterfall] = field(default_factory=list)


@dataclass
class StageStats:
    """Aggregate stage performance statistics."""

    order: int
    name: str
    count: int
    avg_ms: float
    min_ms: float
    max_ms: float
    cache_hit_rate: float = 0.0
    skip_rate: float = 0.0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0


# ── Cost Analysis ────────────────────────────────────────


@dataclass
class ModelCostBreakdown:
    """Cost breakdown for a single model."""

    model: str
    executions: int
    total_cost: float
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read: int = 0
    total_cache_write: int = 0
    total_thinking: int = 0
    total_tool_calls: int = 0
    avg_cost_per_execution: float = 0.0


@dataclass
class CostSummary:
    """Session cost summary."""

    session_id: str
    by_model: List[ModelCostBreakdown] = field(default_factory=list)
    total_cost: float = 0.0
    total_executions: int = 0


@dataclass
class CostTrendPoint:
    """Single point in cost trend chart."""

    period: str
    executions: int = 0
    cost: float = 0.0
    tokens: int = 0


# ── A/B Testing ──────────────────────────────────────────


@dataclass
class ABSide:
    """Result for one side of an A/B test."""

    environment_id: str
    execution_id: str
    result_text: str = ""
    tokens: Dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    duration_ms: int = 0
    iterations: int = 0
    tool_calls_count: int = 0


@dataclass
class ABTestResult:
    """Complete A/B test result."""

    env_a: ABSide = field(default_factory=lambda: ABSide(environment_id="", execution_id=""))
    env_b: ABSide = field(default_factory=lambda: ABSide(environment_id="", execution_id=""))
    user_input: str = ""
