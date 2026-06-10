"""ToolDispatcher — Stage 10's dispatch path, callable from outside Stage 10.

Why this exists (2.3.0): the Stage 6 ``tool_loop="internal"`` strategy
resolves tool calls inside the API stage, CLI-style. Those dispatches
MUST be indistinguishable from Stage 10's own: same
:class:`~geny_executor.tools.registry.ToolRegistry` instance, same
permission ladder (matrix rules → posture → ASK→HITL → hooks — all owned
by :class:`~geny_executor.stages.s10_tool.artifact.default.routers.
RegistryRouter`), same large-result persistence and state-mutation
application, same canonical ``tool_result`` dict shape. Re-implementing
any of that inside Stage 6 would create the second permission-decision
path the 2.2.0 audit explicitly forbids.

So the dispatcher is deliberately thin: it holds the live
:class:`ToolStage` instance and, per call, drives the stage's OWN
machinery — ``build_dispatch_context`` (which reads permission rules /
posture / HITL requester off the stage's attached context at call time,
so ``refresh_runtime`` swaps are visible to internal-loop dispatches on
the very next turn) plus a stateless :class:`SequentialExecutor` for the
single-call route. The executor emits the same ``tool.call_start`` /
``tool.call_complete`` events Stage 10 emits, so per-call timing
attribution is identical across both execution shapes.

Lifetime: ``Pipeline._init_state`` installs a dispatcher onto
``state.tool_dispatcher`` whenever a Tool stage is registered — rebuilt
each run (construction is two attribute writes) so stage replacement via
``PipelineMutator`` is picked up at the next turn boundary.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from geny_executor.core.state import PipelineState
from geny_executor.stages.s10_tool.artifact.default.executors import SequentialExecutor
from geny_executor.stages.s10_tool.artifact.default.routers import RegistryRouter

logger = logging.getLogger(__name__)


__all__ = ["ToolDispatcher"]


class ToolDispatcher:
    """Dispatch one tool call through Stage 10's machinery.

    ``dispatch`` accepts the dict shape Stage 9 hands Stage 10
    (``{"tool_use_id", "tool_name", "tool_input"}``) and returns the
    canonical ``tool_result`` dict (``{"type": "tool_result",
    "tool_use_id", "content", "is_error"?}``) — the exact
    ``ToolResult.to_api_format`` output Stage 10 records into messages.

    Tool-level failures come back as ``is_error`` results (the router's
    contract); only dispatcher-machinery crashes escape as exceptions,
    and the internal loop wraps those too (containment is the turn's
    contract — a bad tool call must never kill the turn).
    """

    def __init__(self, tool_stage: Any) -> None:
        self._stage = tool_stage
        # Stateless single-call executor — execute_all uses only locals,
        # so one instance is safe across concurrent dispatches
        # (InternalAgenticLoop's parallel_tools gathers dispatch()
        # coroutines concurrently).
        self._executor = SequentialExecutor()

    async def dispatch(self, tool_call: Dict[str, Any], state: PipelineState) -> Dict[str, Any]:
        stage = self._stage
        # Built per call, not cached: build_dispatch_context reads the
        # permission rules / posture / hook runner / HITL requester off
        # the stage's live context object — the same object
        # attach_runtime/refresh_runtime mutate — so between-turn host
        # updates reach internal-loop dispatches without any extra
        # plumbing.
        ctx = stage.build_dispatch_context(state)

        router = stage._router
        if isinstance(router, RegistryRouter):
            router.bind_registry(stage._registry)

        results = await self._executor.execute_all(
            [dict(tool_call)], router, ctx, on_event=state.add_event
        )
        return results[0]
