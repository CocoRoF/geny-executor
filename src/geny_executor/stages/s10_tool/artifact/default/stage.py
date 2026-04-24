"""Default implementation of Stage 10: Tool."""

from __future__ import annotations

from typing import Any, Dict, Optional

from geny_executor.core.schema import ConfigField, ConfigSchema
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


# Default parallel budget when a host doesn't specify one. Matches the
# ParallelExecutor default and the PartitionExecutor / StreamingToolExecutor
# defaults — keep these in sync when changing.
_DEFAULT_MAX_CONCURRENCY = 10


class ToolStage(Stage[Any, Any]):
    """Stage 10: Tool.

    Dual abstraction:
      - Level 2 executor: execution pattern (sequential/parallel/partition)
      - Level 2 router: dispatches tool calls to implementations

    Cycle 20260424 (Phase 2 Week 4 Checkpoint 4): exposes
    ``max_concurrency`` through the stage ConfigSchema so hosts can tune
    the parallel budget without swapping executors. Applied to the
    currently-active executor each time ``update_config`` runs.
    """

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        executor: Optional[ToolExecutor] = None,
        router: Optional[ToolRouter] = None,
        context: Optional[ToolContext] = None,
        *,
        max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
    ):
        self._registry = registry or ToolRegistry()
        self._max_concurrency = max(1, int(max_concurrency))
        default_executor = executor or SequentialExecutor()
        # Propagate onto the already-constructed executor if one was
        # passed in — callers may build a ParallelExecutor(5) directly
        # but still want the stage's knob to govern later updates.
        self._apply_max_concurrency(default_executor)
        self._slots: Dict[str, StrategySlot] = {
            "executor": StrategySlot(
                name="executor",
                strategy=default_executor,
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

    def _apply_max_concurrency(self, executor: ToolExecutor) -> None:
        """Push the current budget onto an executor that accepts one.

        SequentialExecutor ignores the knob. ParallelExecutor,
        PartitionExecutor, and StreamingToolExecutor all track
        ``_max_concurrency`` — we set the attribute directly so hosts can
        tune mid-session without reconstructing the executor.
        """
        if hasattr(executor, "_max_concurrency"):
            try:
                executor._max_concurrency = self._max_concurrency  # type: ignore[attr-defined]
            except AttributeError:
                # Some executors may expose the attribute as a
                # read-only descriptor — silently ignore.
                pass

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

    def get_config_schema(self) -> ConfigSchema:
        return ConfigSchema(
            name="tool",
            fields=[
                ConfigField(
                    name="max_concurrency",
                    type="integer",
                    label="Max Concurrency",
                    description=(
                        "Maximum number of tool calls that may execute in "
                        "parallel. Applies to ParallelExecutor, "
                        "PartitionExecutor, and StreamingToolExecutor. "
                        "SequentialExecutor ignores this knob."
                    ),
                    default=_DEFAULT_MAX_CONCURRENCY,
                    min_value=1,
                    max_value=64,
                    ui_widget="slider",
                ),
            ],
        )

    def get_config(self) -> Dict[str, Any]:
        return {"max_concurrency": self._max_concurrency}

    def update_config(self, config: Dict[str, Any]) -> None:
        if "max_concurrency" in config:
            value = int(config["max_concurrency"])
            self._max_concurrency = max(1, value)
            self._apply_max_concurrency(self._executor)

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

        # Bind a state-mutation sink onto the context so tools /
        # executors can apply ``ToolResult.state_mutations`` into
        # ``state.shared`` without plumbing PipelineState down through
        # every layer. The callback closes over the live ``state.shared``
        # dict, so mutations take effect immediately.
        from geny_executor.stages.s10_tool.state_mutation import (
            apply_state_mutations as _apply_raw,
        )
        from geny_executor.tools.base import ToolResult as _TR

        def _state_apply(mutations: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
            if not mutations:
                return {}
            return _apply_raw(
                _TR(content=None, state_mutations=mutations),
                state.shared,
                tool_name=tool_name,
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
            state_apply=_state_apply,
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

        # Re-apply the stage-level concurrency budget each turn — this
        # handles the case where an executor was swapped in via
        # ``StrategySlot.swap`` (which rebuilds with no args) and would
        # otherwise default to its class-level budget.
        self._apply_max_concurrency(executor_strategy)

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
