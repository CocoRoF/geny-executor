"""EventBus — pub/sub for pipeline events."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Callable, Coroutine, Dict, List, Union

from geny_executor.events.types import PipelineEvent

logger = logging.getLogger(__name__)

# Handler can be sync or async
EventHandler = Union[
    Callable[[PipelineEvent], None],
    Callable[[PipelineEvent], Coroutine[Any, Any, None]],
]


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
        """Emit an event to all matching handlers (deduplicated)."""
        seen_ids: set = set()
        matched_handlers: List[EventHandler] = []

        def _collect(handler_list: List[EventHandler]) -> None:
            for h in handler_list:
                h_id = id(h)
                if h_id not in seen_ids:
                    seen_ids.add(h_id)
                    matched_handlers.append(h)

        # Exact match
        _collect(self._handlers.get(event.type, []))

        # Wildcard match
        _collect(self._handlers.get("*", []))

        # Prefix match (e.g., "stage.*" matches "stage.enter")
        if "." in event.type:
            prefix = event.type.rsplit(".", 1)[0] + ".*"
            _collect(self._handlers.get(prefix, []))

        for handler in matched_handlers:
            try:
                result = handler(event)
                # Support both sync and async handlers
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                # Event handlers should not crash the pipeline, but log the error
                logger.warning("Event handler %r failed on %s: %s", handler, event.type, e)

    def clear(self) -> None:
        """Remove all handlers."""
        self._handlers.clear()
