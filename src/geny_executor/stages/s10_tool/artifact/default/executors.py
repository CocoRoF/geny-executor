"""Default artifact executors for Stage 10: Tool."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from geny_executor.tools.base import ToolCapabilities, ToolContext
from geny_executor.tools.registry import ToolRegistry
from geny_executor.stages.s10_tool.interface import (
    ToolEventCallback,
    ToolExecutor,
    ToolRouter,
)


def _emit_call_start(on_event: Optional[ToolEventCallback], tc: Dict[str, Any]) -> None:
    if on_event is None:
        return
    on_event(
        "tool.call_start",
        {
            "tool_use_id": tc.get("tool_use_id", ""),
            "name": tc.get("tool_name", ""),
            "input": tc.get("tool_input", {}),
        },
    )


def _emit_call_complete(
    on_event: Optional[ToolEventCallback],
    tc: Dict[str, Any],
    result_dict: Dict[str, Any],
    duration_ms: int,
) -> None:
    if on_event is None:
        return
    on_event(
        "tool.call_complete",
        {
            "tool_use_id": tc.get("tool_use_id", ""),
            "name": tc.get("tool_name", ""),
            "is_error": bool(result_dict.get("is_error")),
            "duration_ms": duration_ms,
        },
    )


class SequentialExecutor(ToolExecutor):
    """Executes tools one by one in order."""

    @property
    def name(self) -> str:
        return "sequential"

    @property
    def description(self) -> str:
        return "Execute tools sequentially"

    async def execute_all(
        self,
        tool_calls: List[Dict[str, Any]],
        router: ToolRouter,
        context: ToolContext,
        *,
        on_event: Optional[ToolEventCallback] = None,
    ) -> List[Dict[str, Any]]:
        results = []
        for tc in tool_calls:
            _emit_call_start(on_event, tc)
            t0 = time.monotonic()
            result = await router.route(
                tc["tool_name"],
                tc.get("tool_input", {}),
                context,
            )
            duration_ms = int((time.monotonic() - t0) * 1000)
            result_dict = result.to_api_format(tc["tool_use_id"])
            _emit_call_complete(on_event, tc, result_dict, duration_ms)
            results.append(result_dict)
        return results


class ParallelExecutor(ToolExecutor):
    """Executes independent tools concurrently."""

    def __init__(self, max_concurrency: int = 5):
        self._max_concurrency = max_concurrency

    @property
    def name(self) -> str:
        return "parallel"

    @property
    def description(self) -> str:
        return f"Execute tools in parallel (max {self._max_concurrency})"

    async def execute_all(
        self,
        tool_calls: List[Dict[str, Any]],
        router: ToolRouter,
        context: ToolContext,
        *,
        on_event: Optional[ToolEventCallback] = None,
    ) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def _execute_one(tc: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                _emit_call_start(on_event, tc)
                t0 = time.monotonic()
                result = await router.route(
                    tc["tool_name"],
                    tc.get("tool_input", {}),
                    context,
                )
                duration_ms = int((time.monotonic() - t0) * 1000)
                result_dict = result.to_api_format(tc["tool_use_id"])
                _emit_call_complete(on_event, tc, result_dict, duration_ms)
                return result_dict

        tasks = [_execute_one(tc) for tc in tool_calls]
        return list(await asyncio.gather(*tasks))


class PartitionExecutor(ToolExecutor):
    """Partition tool calls by ``ToolCapabilities.concurrency_safe``.

    Cycle 20260424 executor uplift — Phase 1 Week 3 Checkpoint 4.

    For each pending tool call, consults the tool's
    ``capabilities(input)`` to decide:
    - ``concurrency_safe=True`` → run in parallel batch (bounded by
      ``max_concurrency``)
    - ``concurrency_safe=False`` → run serially, after the parallel
      batch completes

    Result order matches the ``tool_calls`` input order — downstream
    stages (Tool Review, Agent, Loop) receive results in a deterministic
    sequence regardless of completion timing.

    When a tool is not found in the registry (shouldn't happen if
    routing is consistent with registration) it's treated as unsafe
    (fail-closed).

    This is the foundation for full partition + streaming orchestration
    (Phase 2 Week 4 — StreamingToolExecutor extension).
    """

    def __init__(
        self,
        registry: Optional[ToolRegistry] = None,
        *,
        max_concurrency: int = 10,
    ):
        self._registry = registry
        self._max_concurrency = max_concurrency

    @property
    def name(self) -> str:
        return "partition"

    @property
    def description(self) -> str:
        return (
            f"Partition tool calls by concurrency_safe capability "
            f"(max parallel: {self._max_concurrency})"
        )

    def bind_registry(self, registry: ToolRegistry) -> None:
        """Late-bind the tool registry — mirrors RegistryRouter pattern."""
        self._registry = registry

    def _lookup_capabilities(self, tc: Dict[str, Any]) -> ToolCapabilities:
        """Peek at a tool's capabilities for this invocation.

        Falls back to ``ToolCapabilities()`` (fail-closed: unsafe) when
        no registry is bound or the tool is unknown.
        """
        if self._registry is None:
            return ToolCapabilities()
        tool = self._registry.get(tc.get("tool_name", ""))
        if tool is None:
            return ToolCapabilities()
        try:
            return tool.capabilities(tc.get("tool_input", {}))
        except Exception:
            # Capability inspection must never crash orchestration.
            return ToolCapabilities()

    async def execute_all(
        self,
        tool_calls: List[Dict[str, Any]],
        router: ToolRouter,
        context: ToolContext,
        *,
        on_event: Optional[ToolEventCallback] = None,
    ) -> List[Dict[str, Any]]:
        # Late-bind registry from the router when needed (mirrors how
        # the default ToolStage calls bind_registry on the router).
        if self._registry is None:
            router_registry = getattr(router, "_registry", None)
            if isinstance(router_registry, ToolRegistry):
                self._registry = router_registry

        # Partition while preserving original positions so we can
        # reconstruct order in the final result list.
        safe_indexed: List[tuple[int, Dict[str, Any]]] = []
        unsafe_indexed: List[tuple[int, Dict[str, Any]]] = []
        for i, tc in enumerate(tool_calls):
            caps = self._lookup_capabilities(tc)
            (safe_indexed if caps.concurrency_safe else unsafe_indexed).append((i, tc))

        results: List[Optional[Dict[str, Any]]] = [None] * len(tool_calls)
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def _run_one(tc: Dict[str, Any]) -> Dict[str, Any]:
            _emit_call_start(on_event, tc)
            t0 = time.monotonic()
            result = await router.route(
                tc["tool_name"],
                tc.get("tool_input", {}),
                context,
            )
            duration_ms = int((time.monotonic() - t0) * 1000)
            result_dict = result.to_api_format(tc["tool_use_id"])
            _emit_call_complete(on_event, tc, result_dict, duration_ms)
            return result_dict

        async def _run_bounded(tc: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                return await _run_one(tc)

        # 1) Parallel batch — concurrency_safe tools
        if safe_indexed:
            parallel_results = await asyncio.gather(*(_run_bounded(tc) for _, tc in safe_indexed))
            for (pos, _), res in zip(safe_indexed, parallel_results):
                results[pos] = res

        # 2) Sequential batch — everything else (strict order)
        for pos, tc in unsafe_indexed:
            results[pos] = await _run_one(tc)

        # All positions filled (we iterated every input); narrow the type
        return [r for r in results if r is not None]
