"""Stage 6: API — built-in :class:`ModelRouter` strategies.

These implementations cover the two ends of the spectrum:

* :class:`PassthroughRouter` — keeps whatever model the pipeline already
  resolved. Used as the default slot strategy so that wiring a router
  slot is always a no-op until a real router is plugged in.
* :class:`AdaptiveModelRouter` — picks between Opus / Sonnet / Haiku
  tiers based on simple, cheap-to-compute query characteristics: total
  message-character count, whether tools are advertised, whether the
  caller asked for extended thinking. Thresholds and tier model names
  are constructor-tunable so hosts can adapt to their own model
  generation without re-implementing the strategy.

The router decision only affects a single API call. The new config is
threaded through ``execute`` locally — ``state.model`` is not mutated.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional

from geny_executor.core.config import ModelConfig
from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.core.state import PipelineState
from geny_executor.stages.s06_api.interface import ModelRouter


class PassthroughRouter(ModelRouter):
    """Default no-op router: never overrides the resolved config."""

    @property
    def name(self) -> str:
        return "passthrough"

    def route(self, cfg: ModelConfig, state: PipelineState) -> Optional[ModelConfig]:
        return None


class AdaptiveModelRouter(ModelRouter):
    """Pick a model tier based on lightweight query heuristics.

    Decision order (first matching rule wins):

    1. ``thinking_enabled`` → ``heavy_model`` (Opus tier).
    2. Estimated message size ≥ ``heavy_threshold_chars`` → ``heavy_model``.
    3. Tools advertised on state → ``balanced_model`` (Sonnet tier).
    4. Estimated message size ≤ ``light_threshold_chars`` → ``light_model``
       (Haiku tier).
    5. Otherwise → ``balanced_model``.

    Thresholds are character counts rather than token counts to avoid a
    tokenizer dependency at routing time. The heuristic is intentionally
    coarse — the router's job is to *bias* selection, not to be optimal.

    If the elected tier model equals ``cfg.model`` the router returns
    ``None`` (no override) so downstream observers can distinguish
    "router ran and picked the same thing" from "router actively
    swapped the model" via the ``api.model_routed`` event.
    """

    DEFAULT_LIGHT = "claude-haiku-4-5-20251001"
    DEFAULT_BALANCED = "claude-sonnet-4-6"
    DEFAULT_HEAVY = "claude-opus-4-7"

    def __init__(
        self,
        *,
        light_model: str = DEFAULT_LIGHT,
        balanced_model: str = DEFAULT_BALANCED,
        heavy_model: str = DEFAULT_HEAVY,
        light_threshold_chars: int = 800,
        heavy_threshold_chars: int = 12_000,
        thinking_promotes_heavy: bool = True,
        tools_promote_balanced: bool = True,
    ) -> None:
        if light_threshold_chars < 0 or heavy_threshold_chars < 0:
            raise ValueError("threshold values must be non-negative")
        if heavy_threshold_chars < light_threshold_chars:
            raise ValueError("heavy_threshold_chars must be >= light_threshold_chars")
        self._light = light_model
        self._balanced = balanced_model
        self._heavy = heavy_model
        self._light_threshold = light_threshold_chars
        self._heavy_threshold = heavy_threshold_chars
        self._thinking_promotes_heavy = thinking_promotes_heavy
        self._tools_promote_balanced = tools_promote_balanced

    @property
    def name(self) -> str:
        return "adaptive"

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name="adaptive",
            fields=[
                ConfigField(
                    name="light_model",
                    type="string",
                    label="Light model",
                    description="Model elected for small, tool-free prompts (Haiku tier).",
                    default=cls.DEFAULT_LIGHT,
                ),
                ConfigField(
                    name="balanced_model",
                    type="string",
                    label="Balanced model",
                    description="Mid-tier model (Sonnet tier); also the fallback tier.",
                    default=cls.DEFAULT_BALANCED,
                ),
                ConfigField(
                    name="heavy_model",
                    type="string",
                    label="Heavy model",
                    description="Model elected for large prompts / extended thinking (Opus tier).",
                    default=cls.DEFAULT_HEAVY,
                ),
                ConfigField(
                    name="light_threshold_chars",
                    type="integer",
                    label="Light threshold (chars)",
                    description="Prompts at or below this size elect the light model.",
                    default=800,
                    min_value=0,
                ),
                ConfigField(
                    name="heavy_threshold_chars",
                    type="integer",
                    label="Heavy threshold (chars)",
                    description="Prompts at or above this size elect the heavy model.",
                    default=12_000,
                    min_value=0,
                ),
                ConfigField(
                    name="thinking_promotes_heavy",
                    type="boolean",
                    label="Thinking promotes heavy",
                    description="Extended thinking always elects the heavy model.",
                    default=True,
                    ui_widget="toggle",
                ),
                ConfigField(
                    name="tools_promote_balanced",
                    type="boolean",
                    label="Tools promote balanced",
                    description="Advertised tools elect at least the balanced model.",
                    default=True,
                    ui_widget="toggle",
                ),
            ],
        )

    def configure(self, config: Dict[str, Any]) -> None:
        """Apply routing knobs from a manifest ``strategy_configs`` entry.

        2026-06-09 environment-philosophy audit §1-2: every tuning knob of
        this router was constructor-only — unreachable from an environment
        edit, so hosts on a different model generation had to fork the
        strategy. The cross-field constraint (heavy >= light) is validated
        against the *merged* result so a partial update cannot sneak an
        inverted threshold pair past the constructor's own check.
        """
        for key in ("light_model", "balanced_model", "heavy_model"):
            if key in config:
                v = config[key]
                if not isinstance(v, str) or not v:
                    raise ValueError(f"adaptive: {key!r} must be a non-empty string, got {v!r}")
        for key in ("light_threshold_chars", "heavy_threshold_chars"):
            if key in config:
                v = config[key]
                if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                    raise ValueError(f"adaptive: {key!r} must be a non-negative integer, got {v!r}")
        for key in ("thinking_promotes_heavy", "tools_promote_balanced"):
            if key in config and not isinstance(config[key], bool):
                raise ValueError(f"adaptive: {key!r} must be a boolean, got {config[key]!r}")

        light_threshold = config.get("light_threshold_chars", self._light_threshold)
        heavy_threshold = config.get("heavy_threshold_chars", self._heavy_threshold)
        if heavy_threshold < light_threshold:
            raise ValueError(
                "adaptive: 'heavy_threshold_chars' must be >= 'light_threshold_chars' "
                f"(got {heavy_threshold} < {light_threshold})"
            )

        self._light = config.get("light_model", self._light)
        self._balanced = config.get("balanced_model", self._balanced)
        self._heavy = config.get("heavy_model", self._heavy)
        self._light_threshold = light_threshold
        self._heavy_threshold = heavy_threshold
        if "thinking_promotes_heavy" in config:
            self._thinking_promotes_heavy = config["thinking_promotes_heavy"]
        if "tools_promote_balanced" in config:
            self._tools_promote_balanced = config["tools_promote_balanced"]

    def get_config(self) -> Dict[str, Any]:
        return {
            "light_model": self._light,
            "balanced_model": self._balanced,
            "heavy_model": self._heavy,
            "light_threshold_chars": self._light_threshold,
            "heavy_threshold_chars": self._heavy_threshold,
            "thinking_promotes_heavy": self._thinking_promotes_heavy,
            "tools_promote_balanced": self._tools_promote_balanced,
        }

    @staticmethod
    def _estimate_chars(state: PipelineState) -> int:
        """Approximate the size of the outbound prompt in characters.

        Counts ``state.system`` plus all message content. Block-shaped
        messages contribute their text/JSON-payload length; bare strings
        contribute their own length.

        ``state.system`` may be either a string OR a list of content
        blocks — Stage 5's cache strategies convert it in place when
        applying cache markers, and s05 runs *before* this router (s06).
        Counting ``len()`` of the block list (the pre-2.2.0 bug, audit
        §1-2) collapsed any system prompt to ~1 char and skewed the
        size heuristic toward the light tier.
        """
        total = 0
        system = state.system
        if isinstance(system, str):
            total += len(system)
        elif isinstance(system, list):
            for block in system:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        total += len(text)
                elif isinstance(block, str):
                    total += len(block)
        for msg in state.messages:
            content = msg.get("content")
            if isinstance(content, str):
                total += len(content)
                continue
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text")
                        if isinstance(text, str):
                            total += len(text)
                            continue
                        # tool_use input / tool_result content: count repr
                        # length so heavy tool payloads still influence
                        # the routing decision.
                        for key in ("input", "content"):
                            if key in block:
                                total += len(str(block[key]))
                                break
        return total

    def _select_tier(self, cfg: ModelConfig, state: PipelineState) -> str:
        if self._thinking_promotes_heavy and cfg.thinking_enabled:
            return self._heavy
        size = self._estimate_chars(state)
        if size >= self._heavy_threshold:
            return self._heavy
        if self._tools_promote_balanced and bool(state.tools):
            return self._balanced
        if size <= self._light_threshold:
            return self._light
        return self._balanced

    def route(self, cfg: ModelConfig, state: PipelineState) -> Optional[ModelConfig]:
        target = self._select_tier(cfg, state)
        if target == cfg.model:
            return None
        return dataclasses.replace(cfg, model=target)


__all__ = ["PassthroughRouter", "AdaptiveModelRouter"]
