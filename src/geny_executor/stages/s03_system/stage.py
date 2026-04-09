"""Stage 3: System — assembles system prompt."""

from __future__ import annotations

from typing import Any, List, Optional

from geny_executor.core.stage import Stage, StrategyInfo
from geny_executor.core.state import PipelineState
from geny_executor.stages.s03_system.builders import PromptBuilder, StaticPromptBuilder
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
        tool_registry: Optional[ToolRegistry] = None,
    ):
        if builder:
            self._builder = builder
        elif prompt:
            self._builder = StaticPromptBuilder(prompt)
        else:
            self._builder = StaticPromptBuilder("You are a helpful assistant.")

        self._tool_registry = tool_registry

    @property
    def name(self) -> str:
        return "system"

    @property
    def order(self) -> int:
        return 3

    @property
    def category(self) -> str:
        return "ingress"

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

    def list_strategies(self) -> List[StrategyInfo]:
        return [
            StrategyInfo(
                slot_name="builder",
                current_impl=type(self._builder).__name__,
                available_impls=[
                    "StaticPromptBuilder",
                    "ComposablePromptBuilder",
                ],
            ),
        ]
