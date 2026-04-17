"""Stage 3: System — concrete stage implementation."""

from __future__ import annotations

from typing import Any, Dict, Optional

from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.core.slot import StrategySlot
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState
from geny_executor.stages.s03_system.interface import PromptBuilder
from geny_executor.stages.s03_system.artifact.default.builders import (
    ComposablePromptBuilder,
    StaticPromptBuilder,
)
from geny_executor.tools.registry import ToolRegistry


class SystemStage(Stage[Any, Any]):
    """Stage 3: System.

    Dual abstraction:
      - Level 2 builder: how to construct the system prompt
    """

    def __init__(
        self,
        builder: Optional[PromptBuilder] = None,
        *,
        prompt: str = "",
        template_vars: Optional[Dict[str, Any]] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ):
        if builder is None:
            builder = StaticPromptBuilder(prompt) if prompt else StaticPromptBuilder()

        self._slots: Dict[str, StrategySlot] = {
            "builder": StrategySlot(
                name="builder",
                strategy=builder,
                registry={
                    "static": StaticPromptBuilder,
                    "composable": ComposablePromptBuilder,
                },
                description="System prompt builder strategy",
            ),
        }
        self._tool_registry = tool_registry
        self._prompt = prompt
        self._template_vars: Dict[str, Any] = dict(template_vars or {})

    @property
    def _builder(self) -> PromptBuilder:
        return self._slots["builder"].strategy  # type: ignore[return-value]

    @property
    def name(self) -> str:
        return "system"

    @property
    def order(self) -> int:
        return 3

    @property
    def category(self) -> str:
        return "ingress"

    def get_strategy_slots(self) -> Dict[str, StrategySlot]:
        return self._slots

    def get_config_schema(self) -> ConfigSchema:
        return ConfigSchema(
            name="system",
            fields=[
                ConfigField(
                    name="prompt",
                    type="string",
                    label="System Prompt",
                    description="Static system prompt injected before the conversation.",
                    default="",
                    ui_widget="textarea",
                ),
                ConfigField(
                    name="template_vars",
                    type="object",
                    label="Template Variables",
                    description="Key-value pairs available to composable prompt builders.",
                    default={},
                ),
            ],
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "prompt": self._prompt,
            "template_vars": dict(self._template_vars),
        }

    def update_config(self, config: Dict[str, Any]) -> None:
        if "prompt" in config:
            prompt = str(config["prompt"])
            self._prompt = prompt
            builder = self._slots["builder"].strategy
            if isinstance(builder, StaticPromptBuilder):
                builder.configure({"prompt": prompt})
        if "template_vars" in config:
            tv = config["template_vars"] or {}
            self._template_vars = dict(tv)

    async def execute(self, input: Any, state: PipelineState) -> Any:
        # Build system prompt
        system = self._builder.build(state)
        state.system = system

        # Register tools in state if registry provided
        if self._tool_registry and not state.tools:
            state.tools = self._tool_registry.to_api_format()

        state.add_event(
            "system.built",
            {
                "prompt_type": "content_blocks" if isinstance(system, list) else "string",
                "prompt_length": (
                    sum(len(b.get("text", "")) for b in system)
                    if isinstance(system, list)
                    else len(str(system))
                ),
                "tools_count": len(state.tools),
            },
        )

        return input
