"""EventBus — pub/sub for pipeline events."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Callable, Dict, List

from geny_executor.events.types import PipelineEvent

# Handler can be sync or async
EventHandler = Callable[[PipelineEvent], None]


class EventBus:
    """Pipeline event bus — all stage transitions and API events flow through here.

    Supports:
      - Exact type matching: bus.on("stage.enter", handler)
      - Wildcard matching: bus.on("*", handler) — receives all events
      - Prefix matching: bus.on("stage.*", handler) — matches stage.enter, stage.exit, etc.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, List[EventHandler]] = defaultdict(list)

    def on(self, event_type: str, handler: EventHandler) -> Callable[[], None]:
        """Register a handler. Returns an unsubscribe function."""
        self._handlers[event_type].append(handler)

        def unsubscribe() -> None:
            self.off(event_type, handler)

        return unsubscribe

    def off(self, event_type: str, handler: EventHandler) -> None:
        """Remove a handler."""
        handlers = self._handlers.get(event_type)
        if handlers and handler in handlers:
            handlers.remove(handler)

    async def emit(self, event: PipelineEvent) -> None:
        """Emit an event to all matching handlers."""
        matched_handlers: List[EventHandler] = []

        # Exact match
        matched_handlers.extend(self._handlers.get(event.type, []))

        # Wildcard match
        matched_handlers.extend(self._handlers.get("*", []))

        # Prefix match (e.g., "stage.*" matches "stage.enter")
        if "." in event.type:
            prefix = event.type.rsplit(".", 1)[0] + ".*"
            matched_handlers.extend(self._handlers.get(prefix, []))

        for handler in matched_handlers:
            try:
                result = handler(event)
                # Support both sync and async handlers
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                # Event handlers should not crash the pipeline
                pass

    def clear(self) -> None:
        """Remove all handlers."""
        self._handlers.clear()
