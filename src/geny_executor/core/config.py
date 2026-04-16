"""Pipeline and model configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from geny_executor.core.state import PipelineState


@dataclass
class ModelConfig:
    """Anthropic model configuration.

    Mirrors the Anthropic Messages API parameter set.
    See: https://docs.anthropic.com/en/api/messages

    Sampling:
        Use EITHER temperature OR top_p, not both.
        top_k is an advanced option — prefer temperature in most cases.

    Extended Thinking:
        thinking_type="adaptive" is recommended for Claude 4.6+ models.
        budget_tokens must be < max_tokens when thinking_type="enabled".
    """

    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 8192
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None

    # Extended thinking
    thinking_enabled: bool = False
    thinking_budget_tokens: int = 10000
    thinking_type: str = "enabled"  # "enabled" | "disabled" | "adaptive"
    thinking_display: Optional[str] = None  # "summarized" | "omitted" | None


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    name: str = "default"

    # Model
    model: ModelConfig = field(default_factory=ModelConfig)

    # API
    api_key: str = ""
    base_url: Optional[str] = None

    # Limits
    max_iterations: int = 50
    cost_budget_usd: Optional[float] = None
    context_window_budget: int = 200_000

    # Behavior
    stream: bool = True
    single_turn: bool = False

    # Artifact selection — maps stage identifier to artifact name.
    # e.g. {"s06_api": "openai", "s15_memory": "vector"}
    # Unspecified stages use "default".
    artifacts: Dict[str, str] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def apply_to_state(self, state: PipelineState) -> None:
        """Apply config values to a PipelineState."""
        # Model / sampling
        state.model = self.model.model
        state.max_tokens = self.model.max_tokens
        state.temperature = self.model.temperature
        state.top_p = self.model.top_p
        state.top_k = self.model.top_k
        state.stop_sequences = self.model.stop_sequences

        # Extended thinking
        state.thinking_enabled = self.model.thinking_enabled
        state.thinking_budget_tokens = self.model.thinking_budget_tokens
        state.thinking_type = self.model.thinking_type
        state.thinking_display = self.model.thinking_display

        # Behavior
        state.stream = self.stream
        state.single_turn = self.single_turn

        # Limits
        state.max_iterations = self.max_iterations
        state.cost_budget_usd = self.cost_budget_usd
        state.context_window_budget = self.context_window_budget
