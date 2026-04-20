"""Default artifact executors for Stage 10: Tool."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from geny_executor.tools.base import ToolContext
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
