"""Execution replay and debug support."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Set

from geny_executor.history.models import ReplayEvent
from geny_executor.history.service import HistoryService


class ExecutionReplayer:
    """Replay a recorded execution from its event stream."""

    def __init__(self, history: HistoryService) -> None:
        self._history = history

    async def replay(
        self,
        exec_id: str,
        speed: float = 1.0,
        event_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        breakpoints: Optional[Set[int]] = None,
    ) -> AsyncGenerator[ReplayEvent, Optional[str]]:
        """
        Replay event stream in time order.

        Args:
            exec_id: execution id to replay
            speed: 1.0 = realtime, 2.0 = 2x, 0 = instant
            event_callback: optional async callback per event
            breakpoints: stage orders to pause at
        """
        events = self._history.load_event_stream(exec_id)
        if not events:
            raise ValueError(f"No events found for execution {exec_id}")

        prev_ts: Optional[float] = None
        bp = set(breakpoints) if breakpoints else set()

        for event in events:
            # Time delay
            ts_str = event.get("timestamp")
            if ts_str and speed > 0:
                try:
                    from datetime import datetime

                    ts = datetime.fromisoformat(ts_str).timestamp()
                    if prev_ts is not None:
                        delay = (ts - prev_ts) / speed
                        if delay > 0:
                            await asyncio.sleep(min(delay, 5.0))  # cap at 5s
                    prev_ts = ts
                except (ValueError, TypeError):
                    pass

            # Breakpoint check
            stage_order = event.get("data", {}).get("stage_order")
            if bp and stage_order in bp and event.get("type") == "stage_start":
                command = yield ReplayEvent(type="breakpoint", event=event, stage_order=stage_order)
                while command != "continue":
                    if command == "step":
                        bp = {stage_order + 1}
                        break
                    command = yield ReplayEvent(type="waiting", event=None)

            # Yield event
            yield ReplayEvent(type="event", event=event)

            if event_callback:
                await event_callback(event)

    def get_stage_snapshot(
        self, exec_id: str, stage_order: int, iteration: int = 0
    ) -> Optional[Dict[str, Any]]:
        """Get state snapshot at a specific stage."""
        events = self._history.load_event_stream(exec_id)
        if not events:
            return None

        snapshot_events: List[Dict[str, Any]] = []
        for event in events:
            snapshot_events.append(event)
            if (
                event.get("type") == "stage_complete"
                and event.get("data", {}).get("stage_order") == stage_order
                and event.get("data", {}).get("iteration", 0) == iteration
            ):
                break

        return {
            "events_count": len(snapshot_events),
            "stage_order": stage_order,
            "iteration": iteration,
            "events": snapshot_events[-20:],
        }

    def get_events_summary(self, exec_id: str) -> Dict[str, Any]:
        """Get summary of event stream without loading all events."""
        events = self._history.load_event_stream(exec_id)
        if not events:
            return {"total_events": 0, "stages": [], "iterations": 0}

        stage_enters: List[Dict[str, Any]] = []
        max_iteration = 0
        event_types: Dict[str, int] = {}

        for event in events:
            etype = event.get("type", "unknown")
            event_types[etype] = event_types.get(etype, 0) + 1

            it = event.get("iteration", 0)
            if it > max_iteration:
                max_iteration = it

            if etype == "stage_start":
                stage_enters.append(
                    {
                        "stage_order": event.get("data", {}).get("stage_order"),
                        "stage_name": event.get("stage", ""),
                        "iteration": it,
                    }
                )

        return {
            "total_events": len(events),
            "event_types": event_types,
            "stages": stage_enters,
            "iterations": max_iteration,
        }


class DebugExecutor:
    """Debug mode executor — supports breakpoints during live execution."""

    def __init__(self) -> None:
        self._breakpoints: Set[int] = set()
        self._paused = False
        self._continue_event = asyncio.Event()
        self._step_mode = False

    def set_breakpoints(self, stage_orders: Set[int]) -> None:
        """Set breakpoints at specific stage orders."""
        self._breakpoints = set(stage_orders)

    def clear_breakpoints(self) -> None:
        """Clear all breakpoints."""
        self._breakpoints.clear()

    def continue_execution(self) -> None:
        """Resume execution after breakpoint."""
        self._step_mode = False
        self._continue_event.set()

    def step_next(self) -> None:
        """Execute until next stage."""
        self._step_mode = True
        self._continue_event.set()

    @property
    def is_paused(self) -> bool:
        return self._paused

    async def check_breakpoint(
        self,
        stage_order: int,
        stage_name: str,
        event_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ) -> None:
        """Check if we should pause at this stage. Called by pipeline."""
        if stage_order not in self._breakpoints and not self._step_mode:
            return

        self._paused = True
        if event_callback:
            await event_callback(
                {
                    "type": "debug_pause",
                    "stage_order": stage_order,
                    "stage_name": stage_name,
                }
            )

        self._continue_event.clear()
        await self._continue_event.wait()
        self._paused = False
