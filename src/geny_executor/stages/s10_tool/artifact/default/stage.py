"""Default implementation of Stage 10: Tool."""

from __future__ import annotations

from typing import Any, Dict, Optional

from geny_executor.core.slot import StrategySlot
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState
from geny_executor.tools.base import ToolContext
from geny_executor.tools.registry import ToolRegistry
from geny_executor.tools.stage_binding import ToolAccessDenied
from geny_executor.stages.s10_tool.interface import ToolExecutor, ToolRouter
from geny_executor.stages.s10_tool.artifact.default.executors import (
    ParallelExecutor,
    PartitionExecutor,
    SequentialExecutor,
)
from geny_executor.stages.s10_tool.artifact.default.routers import RegistryRouter


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
        self._slots: Dict[str, StrategySlot] = {
            "executor": StrategySlot(
                name="executor",
                strategy=executor or SequentialExecutor(),
                registry={
                    "sequential": SequentialExecutor,
                    "parallel": ParallelExecutor,
                    # Phase 1 W3 Checkpoint 4 — capability-aware partition
                    # executor. Opt-in: set via `mutator.swap_strategy(
                    # stage_order=10, slot_name="executor",
                    # impl_name="partition")`.
                    "partition": PartitionExecutor,
                },
                description="Tool execution strategy",
            ),
            "router": StrategySlot(
                name="router",
                strategy=router or RegistryRouter(self._registry),
                registry={
                    "registry": RegistryRouter,
                },
                description="Tool dispatch strategy",
            ),
        }
        self._context = context or ToolContext()

    @property
    def _executor(self) -> ToolExecutor:
        return self._slots["executor"].strategy  # type: ignore[return-value]

    @property
    def _router(self) -> ToolRouter:
        return self._slots["router"].strategy  # type: ignore[return-value]

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

    def get_strategy_slots(self) -> Dict[str, StrategySlot]:
        return self._slots

    def should_bypass(self, state: PipelineState) -> bool:
        return not state.pending_tool_calls

    async def execute(self, input: Any, state: PipelineState) -> Any:
        if not state.pending_tool_calls:
            return input

        tool_calls = list(state.pending_tool_calls)

        binding = self.tool_binding
        for tc in tool_calls:
            tool_name = tc.get("tool_name", "")
            if not binding.is_allowed(tool_name):
                raise ToolAccessDenied(tool_name, self.order)

        state.add_event(
            "tool.execute_start",
            {
                "count": len(tool_calls),
                "tools": [tc["tool_name"] for tc in tool_calls],
            },
        )

        ctx = ToolContext(
            session_id=state.session_id,
            working_dir=self._context.working_dir,
            storage_path=self._context.storage_path,
            env_vars=self._context.env_vars,
            allowed_paths=self._context.allowed_paths,
            metadata=self._context.metadata,
            stage_order=self.order,
            stage_name=self.name,
        )

        router = self._router
        if isinstance(router, RegistryRouter):
            router.bind_registry(self._registry)

        # PartitionExecutor needs direct registry access to peek at each
        # tool's ``capabilities(input)`` before deciding parallel vs
        # serial. Other executors ignore the bind call.
        executor_strategy = self._executor
        bind_registry = getattr(executor_strategy, "bind_registry", None)
        if callable(bind_registry):
            bind_registry(self._registry)

        results = await executor_strategy.execute_all(
            tool_calls, router, ctx, on_event=state.add_event
        )

        state.add_message("user", results)
        state.tool_results = results
        state.pending_tool_calls = []
        state.loop_decision = "continue"

        state.add_event(
            "tool.execute_complete",
            {
                "count": len(results),
                "errors": sum(1 for r in results if r.get("is_error")),
            },
        )

        return input
