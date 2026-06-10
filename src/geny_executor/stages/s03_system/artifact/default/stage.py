"""Stage 3: System — concrete stage implementation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.core.slot import StrategySlot
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState
from geny_executor.stages.s03_system.interface import PromptBuilder
from geny_executor.stages.s03_system.artifact.default.builders import (
    ComposablePromptBuilder,
    StaticPromptBuilder,
)
from geny_executor.stages.s03_system.persona import DynamicPersonaPromptBuilder
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
                    # Phase 7 S7.1 — host-attached PersonaProvider
                    # drives this. Manifests can name it; the actual
                    # provider instance must arrive via
                    # ``Pipeline.attach_runtime(system_builder=...)``.
                    "dynamic_persona": DynamicPersonaPromptBuilder,
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
                    description=(
                        "Key-value pairs substituted into the built system "
                        "prompt: every {name} placeholder is replaced "
                        "post-build, whichever builder produced the prompt. "
                        "Placeholders without a matching key are left intact."
                    ),
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

    def _apply_template_vars(
        self, system: Union[str, List[Dict[str, Any]]]
    ) -> Union[str, List[Dict[str, Any]]]:
        """Substitute ``{name}`` placeholders into the built prompt.

        Why post-build instead of a builder kwarg (2.2.0 wave 4, config
        liveness): the :class:`PromptBuilder` contract is ``build(state)``
        — adding a ``template_vars`` parameter would break every custom
        builder hosts attach via ``Pipeline.attach_runtime``. Substituting
        on the *output* keeps the contract intact and works uniformly for
        static, composable and persona builders.

        Substitution is a literal ``{key}`` → ``str(value)`` replacement,
        NOT ``str.format``: prompts routinely contain literal braces (JSON
        examples, code snippets) that ``format`` would choke on. Unknown
        placeholders are left untouched.
        """

        def _substitute(text: str) -> str:
            for key, value in self._template_vars.items():
                text = text.replace("{" + str(key) + "}", str(value))
            return text

        if isinstance(system, str):
            return _substitute(system)
        if isinstance(system, list):
            blocks: List[Dict[str, Any]] = []
            for block in system:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    block = {**block, "text": _substitute(block["text"])}
                blocks.append(block)
            return blocks
        return system

    async def execute(self, input: Any, state: PipelineState) -> Any:
        # Build system prompt
        system = self._builder.build(state)
        if self._template_vars:
            system = self._apply_template_vars(system)
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
