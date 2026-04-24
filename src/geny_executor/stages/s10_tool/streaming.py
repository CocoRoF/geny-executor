"""Streaming tool executor — online variant of PartitionExecutor.

Cycle 20260424 executor uplift — Phase 2 Week 4 Checkpoint 1.

``PartitionExecutor`` assumes all pending tool calls arrive in one
batch. ``StreamingToolExecutor`` is the online counterpart: tool calls
are added incrementally (as the LLM streams ``tool_use`` blocks) and
safe tools start executing immediately. When ``drain()`` is called,
the executor waits for all work to complete and returns results in
**received order** (not completion order).

Key invariants:

1. **Receive-order output.** ``drain()`` preserves the index at which
   each call was added. Downstream stages (Tool Review, Agent, Loop)
   see a deterministic sequence regardless of completion timing.
2. **Safe tools run immediately** when the pending queue contains no
   unsafe tools. This maximises overlap with the LLM's streaming
   response — by the time the stream ends, safe tools may already
   be finishing.
3. **Unsafe tools serialise.** A ``concurrency_safe=False`` call blocks
   subsequent calls (safe or unsafe) until it completes. This honours
   the fail-closed default: when ordering / atomicity matters, the
   tool opts out of parallelism and the executor respects the chain.
4. **Bounded parallelism.** ``max_concurrency`` caps the number of
   in-flight safe tools. Extra safe calls wait for a slot.
5. **Fail-closed on missing metadata.** Unknown tool names, missing
   registry, or ``capabilities()`` raising all fall back to unsafe
   treatment (serialize).

See ``executor_uplift/06_design_tool_system.md`` §6 and
``12_detailed_plan.md`` §2.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from geny_executor.stages.s10_tool.artifact.default.executors import (
    _emit_call_complete,
    _emit_call_start,
)
from geny_executor.stages.s10_tool.interface import ToolEventCallback, ToolRouter
from geny_executor.stages.s10_tool.persistence import maybe_persist_large_result
from geny_executor.tools.base import ToolCapabilities, ToolContext
from geny_executor.tools.registry import ToolRegistry


class StreamingToolExecutor:
    """Executes tool calls as they stream in, preserves receive order on drain.

    Unlike the batch executors that implement ``ToolExecutor.execute_all``,
    this class exposes an ``add(call)`` / ``drain()`` interface so hosts
    integrating with streaming LLM responses can kick off safe tools
    while the model is still emitting subsequent tool_use blocks.

    Typical usage::

        executor = StreamingToolExecutor(registry=reg, router=router)
        async for tool_use in llm_stream:
            await executor.add(_as_tool_call(tool_use))
        results = await executor.drain(context, on_event=state.add_event)

    When every tool in a turn is concurrency-safe, overlapping with
    the LLM stream can cut end-to-end latency roughly in half for
    read-only parallel searches. When any tool is unsafe, the executor
    degrades gracefully to the serial ordering the tool expected.
    """

    def __init__(
        self,
        *,
        registry: Optional[ToolRegistry] = None,
        router: Optional[ToolRouter] = None,
        max_concurrency: int = 10,
    ):
        self._registry = registry
        self._router = router
        self._max_concurrency = max_concurrency

        # Indexed queues — ``_order`` records insertion order so drain()
        # can return results in the same sequence.
        self._order: List[str] = []  # tool_use_id per call
        self._calls: Dict[str, Dict[str, Any]] = {}  # id → call dict
        self._caps: Dict[str, ToolCapabilities] = {}  # id → resolved caps

        # In-flight async tasks keyed by tool_use_id
        self._tasks: Dict[str, asyncio.Task[Dict[str, Any]]] = {}
        # Completed results keyed by tool_use_id (filled by task done)
        self._results: Dict[str, Dict[str, Any]] = {}
        # Pending safe calls awaiting a slot (FIFO)
        self._safe_queue: List[str] = []
        # Pending unsafe calls awaiting serial slot (FIFO)
        self._unsafe_queue: List[str] = []

        self._sem = asyncio.Semaphore(max_concurrency)
        # ``_chain_barrier`` is held by the currently-running unsafe tool;
        # safe tools added while it's held wait for release before starting
        # so the "unsafe serializes everything after it" invariant holds.
        self._chain_barrier: Optional[asyncio.Future[None]] = None

        # Keeps track of whether drain has been called — after drain,
        # add() becomes an error.
        self._drained: bool = False

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    @property
    def pending_count(self) -> int:
        """Calls that have been added but haven't finished yet."""
        return len(self._tasks) + len(self._safe_queue) + len(self._unsafe_queue)

    def bind_registry(self, registry: ToolRegistry) -> None:
        """Late-bind the registry — mirrors PartitionExecutor / RegistryRouter."""
        self._registry = registry

    def bind_router(self, router: ToolRouter) -> None:
        """Late-bind the router."""
        self._router = router

    async def add(self, call: Dict[str, Any], context: ToolContext) -> None:
        """Register a new tool call.

        The call's capabilities are resolved immediately (so
        ``pending_count`` and scheduling decisions are accurate) but
        execution may be deferred behind a chain barrier.

        Raises:
            RuntimeError: if called after ``drain()``.
            ValueError: if the call dict lacks a ``tool_use_id`` or
                duplicates an existing one.
        """
        if self._drained:
            raise RuntimeError("StreamingToolExecutor: add() after drain()")
        tuid = call.get("tool_use_id")
        if not tuid or not isinstance(tuid, str):
            raise ValueError("StreamingToolExecutor: call dict must carry a 'tool_use_id' string")
        if tuid in self._calls:
            raise ValueError(f"StreamingToolExecutor: duplicate tool_use_id {tuid!r}")

        self._order.append(tuid)
        self._calls[tuid] = call
        caps = self._lookup_capabilities(call)
        self._caps[tuid] = caps

        if caps.concurrency_safe and self._chain_barrier is None:
            # Safe and no active unsafe chain → start immediately
            # (semaphore will still bound simultaneous in-flight).
            self._tasks[tuid] = asyncio.create_task(self._run_safe(tuid, context))
        elif caps.concurrency_safe:
            # Safe but blocked by an active unsafe chain — queue until release
            self._safe_queue.append(tuid)
        else:
            # Unsafe — raise the chain barrier immediately so any safe
            # call added *after* this one queues behind it, even before
            # we start executing. Barrier is released only when the
            # unsafe queue fully drains (see ``_run_unsafe``).
            if self._chain_barrier is None:
                self._chain_barrier = asyncio.get_running_loop().create_future()
            self._unsafe_queue.append(tuid)
            # If no in-flight tasks, start this unsafe now.
            if not self._tasks:
                await self._start_next_unsafe(context)

    async def drain(
        self,
        context: ToolContext,
        *,
        on_event: Optional[ToolEventCallback] = None,
    ) -> List[Dict[str, Any]]:
        """Wait for every added call to finish; return results in receive order.

        Emits ``tool.call_start`` / ``tool.call_complete`` events through
        ``on_event`` if provided — identical shape to the batch executors.
        Events fire from inside the execution coroutine, so they interleave
        naturally with actual execution timing.
        """
        self._drained = True
        # Store on_event so _run_safe / _run_unsafe can use it.
        self._on_event = on_event  # type: ignore[attr-defined]

        # Drain loop: wait for in-flight tasks, then try to advance queues.
        while self._tasks or self._safe_queue or self._unsafe_queue:
            if self._tasks:
                done, _ = await asyncio.wait(
                    self._tasks.values(), return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    # Find and remove the completed task entry
                    tuid = next((k for k, v in self._tasks.items() if v is task), None)
                    if tuid is None:
                        continue
                    del self._tasks[tuid]
                    # Result is recorded by the task body itself via
                    # self._results[tuid]; propagate exceptions if any.
                    exc = task.exception()
                    if exc is not None and tuid not in self._results:
                        # The task crashed before writing a result — surface
                        # as a synthetic error entry.
                        self._results[tuid] = {
                            "type": "tool_result",
                            "tool_use_id": tuid,
                            "content": f"error: {exc}",
                            "is_error": True,
                        }

            # If the chain barrier was released, start waiting unsafe/safe.
            await self._advance_queues(context)

        return [self._results[tuid] for tuid in self._order]

    # ─────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────

    def _lookup_capabilities(self, call: Dict[str, Any]) -> ToolCapabilities:
        """Same fail-closed policy as PartitionExecutor."""
        if self._registry is None:
            return ToolCapabilities()
        tool = self._registry.get(call.get("tool_name", ""))
        if tool is None:
            return ToolCapabilities()
        try:
            return tool.capabilities(call.get("tool_input", {}))
        except Exception:
            return ToolCapabilities()

    async def _advance_queues(self, context: ToolContext) -> None:
        """Attempt to promote queued calls once slots open up.

        Ordering rules:
        - ``unsafe`` has priority over ``safe`` whenever nothing is
          in flight. The barrier is held for the entire unsafe queue
          and released only when ``_run_unsafe`` sees an empty queue.
        - ``safe`` drains only when the barrier is down (no unsafe
          work pending *and* no unsafe work running).
        """
        # Unsafe first: if nothing is running and unsafe queued, start it.
        if not self._tasks and self._unsafe_queue:
            await self._start_next_unsafe(context)
            return

        # Otherwise, if the chain barrier is down, safe calls may run.
        if self._chain_barrier is None:
            while self._safe_queue:
                tuid = self._safe_queue.pop(0)
                self._tasks[tuid] = asyncio.create_task(self._run_safe(tuid, context))

    async def _start_next_unsafe(self, context: ToolContext) -> None:
        """Pop the head of the unsafe queue and kick off its task.

        The barrier is already up (set by ``add`` the moment the first
        unsafe call entered the pipeline). ``_run_unsafe`` releases it
        only when the queue is fully drained.
        """
        if not self._unsafe_queue:
            return
        tuid = self._unsafe_queue.pop(0)
        if self._chain_barrier is None:
            # Defensive — should have been set by add()
            self._chain_barrier = asyncio.get_running_loop().create_future()
        self._tasks[tuid] = asyncio.create_task(self._run_unsafe(tuid, context))

    async def _run_safe(self, tuid: str, context: ToolContext) -> Dict[str, Any]:
        """Execute a concurrency-safe call under the semaphore."""
        async with self._sem:
            return await self._dispatch(tuid, context)

    async def _run_unsafe(self, tuid: str, context: ToolContext) -> Dict[str, Any]:
        """Execute an unsafe call.

        The chain barrier is released **only when the entire unsafe
        queue has drained** — if more unsafe calls are queued, they
        run serially before safe calls are allowed through. This keeps
        the "unsafe serializes everything after it" invariant across
        consecutive unsafe additions.
        """
        try:
            return await self._dispatch(tuid, context)
        finally:
            if not self._unsafe_queue:
                barrier = self._chain_barrier
                self._chain_barrier = None
                if barrier is not None and not barrier.done():
                    barrier.set_result(None)

    async def _dispatch(self, tuid: str, context: ToolContext) -> Dict[str, Any]:
        """Shared execution body — routes, times, emits events, stores result."""
        call = self._calls[tuid]
        on_event = getattr(self, "_on_event", None)
        _emit_call_start(on_event, call)
        t0 = time.monotonic()
        assert self._router is not None, "router must be bound before dispatch"
        result = await self._router.route(
            call["tool_name"],
            call.get("tool_input", {}),
            context,
        )
        duration_ms = int((time.monotonic() - t0) * 1000)
        result = maybe_persist_large_result(
            result,
            tool_use_id=tuid,
            tool_name=call["tool_name"],
            capabilities=self._caps.get(tuid, ToolCapabilities()),
            context=context,
        )
        result_dict = result.to_api_format(tuid)
        _emit_call_complete(on_event, call, result_dict, duration_ms)
        self._results[tuid] = result_dict
        return result_dict
