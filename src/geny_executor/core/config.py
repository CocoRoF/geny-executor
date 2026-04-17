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

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready representation."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stop_sequences": list(self.stop_sequences) if self.stop_sequences else None,
            "thinking_enabled": self.thinking_enabled,
            "thinking_budget_tokens": self.thinking_budget_tokens,
            "thinking_type": self.thinking_type,
            "thinking_display": self.thinking_display,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Rehydrate a :class:`ModelConfig` from :meth:`to_dict` output.

        Unknown keys are ignored so forward-compatibility across minor versions
        is preserved. Missing keys fall back to dataclass defaults.
        """
        stop_raw = data.get("stop_sequences")
        kwargs: Dict[str, Any] = {}
        for key in (
            "model",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "thinking_enabled",
            "thinking_budget_tokens",
            "thinking_type",
            "thinking_display",
        ):
            if key in data:
                kwargs[key] = data[key]
        if stop_raw is not None:
            kwargs["stop_sequences"] = list(stop_raw)
        return cls(**kwargs)


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

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready representation of top-level pipeline settings.

        The nested :class:`ModelConfig` is serialized via its own ``to_dict``.
        """
        return {
            "name": self.name,
            "model": self.model.to_dict(),
            "api_key": self.api_key,
            "base_url": self.base_url,
            "max_iterations": self.max_iterations,
            "cost_budget_usd": self.cost_budget_usd,
            "context_window_budget": self.context_window_budget,
            "stream": self.stream,
            "single_turn": self.single_turn,
            "artifacts": dict(self.artifacts),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Rehydrate a :class:`PipelineConfig` from :meth:`to_dict` output.

        Unknown keys are ignored so forward-compatibility is preserved. The
        nested ``model`` key accepts either a dict (parsed via
        :meth:`ModelConfig.from_dict`) or a :class:`ModelConfig` instance.
        """
        kwargs: Dict[str, Any] = {}
        for key in (
            "name",
            "api_key",
            "base_url",
            "max_iterations",
            "cost_budget_usd",
            "context_window_budget",
            "stream",
            "single_turn",
        ):
            if key in data:
                kwargs[key] = data[key]
        if "artifacts" in data and data["artifacts"] is not None:
            kwargs["artifacts"] = dict(data["artifacts"])
        if "metadata" in data and data["metadata"] is not None:
            kwargs["metadata"] = dict(data["metadata"])
        model_raw = data.get("model")
        if isinstance(model_raw, ModelConfig):
            kwargs["model"] = model_raw
        elif isinstance(model_raw, dict):
            kwargs["model"] = ModelConfig.from_dict(model_raw)
        return cls(**kwargs)

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
