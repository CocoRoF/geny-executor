"""Stage 10: Tool — executes tool calls from parsed response."""

from __future__ import annotations

from typing import Any, List, Optional

from geny_executor.core.stage import Stage, StrategyInfo
from geny_executor.core.state import PipelineState
from geny_executor.tools.base import ToolContext
from geny_executor.tools.registry import ToolRegistry
from geny_executor.stages.s10_tool.executors import SequentialExecutor, ToolExecutor
from geny_executor.stages.s10_tool.routers import RegistryRouter, ToolRouter


class ToolStage(Stage[Any, Any]):
    """Stage 10: Tool.

    Dual abstraction:
      - Level 2 executor: execution pattern (sequential/parallel)
      - Level 2 router: dispatches tool calls to implementations
    """

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        executor: Optional[ToolExecutor] = None,
        router: Optional[ToolRouter] = None,
        context: Optional[ToolContext] = None,
    ):
        self._registry = registry or ToolRegistry()
        self._executor = executor or SequentialExecutor()
        self._router = router or RegistryRouter(self._registry)
        self._context = context or ToolContext()

    @property
    def name(self) -> str:
        return "tool"

    @property
    def order(self) -> int:
        return 10

    @property
    def category(self) -> str:
        return "execution"

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    def should_bypass(self, state: PipelineState) -> bool:
        """Skip if no pending tool calls."""
        return not state.pending_tool_calls

    async def execute(self, input: Any, state: PipelineState) -> Any:
        if not state.pending_tool_calls:
            return input

        tool_calls = list(state.pending_tool_calls)

        state.add_event("tool.execute_start", {
            "count": len(tool_calls),
            "tools": [tc["tool_name"] for tc in tool_calls],
        })

        # Set context
        ctx = ToolContext(
            session_id=state.session_id,
            working_dir=self._context.working_dir,
            metadata=self._context.metadata,
        )

        # Execute all tool calls
        results = await self._executor.execute_all(tool_calls, self._router, ctx)

        # Add tool results to conversation messages
        state.add_message("user", results)

        # Store results and clear pending
        state.tool_results = results
        state.pending_tool_calls = []

        # Signal that we need to continue the loop (tool results need API call)
        state.loop_decision = "continue"

        state.add_event("tool.execute_complete", {
            "count": len(results),
            "errors": sum(1 for r in results if r.get("is_error")),
        })

        return input

    def list_strategies(self) -> List[StrategyInfo]:
        return [
            StrategyInfo(
                slot_name="executor",
                current_impl=type(self._executor).__name__,
                available_impls=["SequentialExecutor", "ParallelExecutor"],
            ),
            StrategyInfo(
                slot_name="router",
                current_impl=type(self._router).__name__,
                available_impls=["RegistryRouter"],
            ),
        ]
