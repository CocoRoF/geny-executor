"""Default artifact executors for Stage 10: Tool."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from geny_executor.tools.base import ToolContext
from geny_executor.stages.s10_tool.interface import ToolExecutor, ToolRouter


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
    ) -> List[Dict[str, Any]]:
        results = []
        for tc in tool_calls:
            result = await router.route(
                tc["tool_name"],
                tc.get("tool_input", {}),
                context,
            )
            results.append(result.to_api_format(tc["tool_use_id"]))
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
    ) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def _execute_one(tc: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                result = await router.route(
                    tc["tool_name"],
                    tc.get("tool_input", {}),
                    context,
                )
                return result.to_api_format(tc["tool_use_id"])

        tasks = [_execute_one(tc) for tc in tool_calls]
        return list(await asyncio.gather(*tasks))
