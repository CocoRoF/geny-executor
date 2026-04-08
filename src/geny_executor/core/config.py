"""Pipeline and model configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    """Anthropic model configuration."""

    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 8192
    temperature: float = 0.0
    top_p: Optional[float] = None
    stop_sequences: Optional[List[str]] = None

    # Extended thinking
    thinking_enabled: bool = False
    thinking_budget_tokens: int = 10000


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
    stream: bool = False
    single_turn: bool = False

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def apply_to_state(self, state: Any) -> None:
        """Apply config values to a PipelineState."""
        state.model = self.model.model
        state.max_tokens = self.model.max_tokens
        state.temperature = self.model.temperature
        state.stop_sequences = self.model.stop_sequences
        state.thinking_enabled = self.model.thinking_enabled
        state.thinking_budget_tokens = self.model.thinking_budget_tokens
        state.max_iterations = self.max_iterations
        state.cost_budget_usd = self.cost_budget_usd
        state.context_window_budget = self.context_window_budget
