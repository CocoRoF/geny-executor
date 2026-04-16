"""Pipeline state — the mutable context flowing through all stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union


@dataclass
class TokenUsage:
    """Token usage for a single API call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __iadd__(self, other: TokenUsage) -> TokenUsage:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        return self

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
            cache_read_input_tokens=(self.cache_read_input_tokens + other.cache_read_input_tokens),
        )


@dataclass
class CacheMetrics:
    """Prompt caching efficiency metrics."""

    total_cache_writes: int = 0
    total_cache_reads: int = 0
    estimated_savings_usd: float = 0.0
    cache_hit_rate: float = 0.0


@dataclass
class PipelineState:
    """Pipeline execution state — readable/writable by all stages.

    Accumulates across loop iterations.
    """

    # ── Identity ──
    session_id: str = ""
    pipeline_id: str = ""

    # ── Messages (Anthropic API format) ──
    system: Union[str, List[Dict[str, Any]]] = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)

    # ── Execution tracking ──
    iteration: int = 0
    max_iterations: int = 50
    current_stage: str = ""
    stage_history: List[str] = field(default_factory=list)

    # ── Behavior (from PipelineConfig) ──
    stream: bool = True
    single_turn: bool = False

    # ── Model config ──
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 8192
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    tools: List[Dict[str, Any]] = field(default_factory=list)
    tool_choice: Optional[Dict[str, Any]] = None
    stop_sequences: Optional[List[str]] = None

    # ── Extended Thinking ──
    thinking_enabled: bool = False
    thinking_budget_tokens: int = 10000
    thinking_type: str = "enabled"  # "enabled" | "disabled" | "adaptive"
    thinking_display: Optional[str] = None  # "summarized" | "omitted" | None
    thinking_history: List[Dict[str, Any]] = field(default_factory=list)

    # ── Token & Cost tracking ──
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    turn_token_usage: List[TokenUsage] = field(default_factory=list)
    total_cost_usd: float = 0.0
    cost_budget_usd: Optional[float] = None

    # ── Cache tracking ──
    cache_metrics: CacheMetrics = field(default_factory=CacheMetrics)

    # ── Context ──
    memory_refs: List[Dict[str, Any]] = field(default_factory=list)
    context_window_budget: int = 200_000

    # ── Loop control ──
    loop_decision: str = "continue"  # continue | complete | error | escalate
    completion_signal: Optional[str] = None
    completion_detail: Optional[str] = None

    # ── Tool execution ──
    pending_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)

    # ── Agent orchestration ──
    delegate_requests: List[Dict[str, Any]] = field(default_factory=list)
    agent_results: List[Dict[str, Any]] = field(default_factory=list)

    # ── Evaluation ──
    evaluation_score: Optional[float] = None
    evaluation_feedback: Optional[str] = None

    # ── Output ──
    final_text: str = ""
    final_output: Optional[Any] = None

    # ── Raw API response (for debugging/passthrough) ──
    last_api_response: Optional[Any] = None

    # ── Metadata ──
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Event log ──
    events: List[Dict[str, Any]] = field(default_factory=list)

    # ── Event listener (set by pipeline for streaming) ──
    _event_listener: Optional[Any] = field(default=None, repr=False)

    def add_event(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Append an event to the log. If a listener is set, also notify it."""
        event_dict = {
            "type": event_type,
            "stage": self.current_stage,
            "iteration": self.iteration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data or {},
        }
        self.events.append(event_dict)
        self.updated_at = datetime.now(timezone.utc)

        # Forward to pipeline event listener (for streaming)
        if self._event_listener is not None:
            self._event_listener(event_dict)

    def add_message(self, role: str, content: Any) -> None:
        """Append a message in Anthropic API format."""
        self.messages.append({"role": role, "content": content})

    def add_tool_result(self, tool_use_id: str, content: Any, is_error: bool = False) -> None:
        """Append a tool result message."""
        result: Dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
        }
        if is_error:
            result["is_error"] = True
        self.tool_results.append(result)

    def accumulate_cost(self, cost_usd: float) -> None:
        """Add cost to the running total."""
        self.total_cost_usd += cost_usd

    @property
    def is_over_budget(self) -> bool:
        """Check if cost budget is exceeded."""
        if self.cost_budget_usd is None:
            return False
        return self.total_cost_usd >= self.cost_budget_usd

    @property
    def is_over_iterations(self) -> bool:
        """Check if max iterations is exceeded."""
        return self.iteration >= self.max_iterations
