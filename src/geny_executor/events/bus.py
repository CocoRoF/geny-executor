"""EventBus — pub/sub for pipeline events."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Callable, Coroutine, Dict, List, Set, Union

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

    Two dispatch paths exist since 2.2.0:

    - :meth:`emit` (async) — the original path; async handlers are
      awaited inline. Used by the pipeline for its own bus-native
      events (stage transitions, pipeline lifecycle).
    - :meth:`emit_sync` (sync) — for producers that cannot await,
      chiefly the ``state.add_event`` → bus bridge (audit 2026-06-09
      §3.2: state events were invisible to ``pipeline.on()``
      subscribers because the only dispatch path was async and
      ``add_event`` is synchronous). Sync handlers run inline,
      preserving the put_nowait ordering streaming consumers rely on;
      async handlers are scheduled as tasks (fire-and-forget,
      reference-pinned so the loop cannot GC them mid-flight).
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        # Strong refs to fire-and-forget handler tasks created by
        # emit_sync — asyncio only keeps weak references to tasks, so
        # without this set a scheduled async handler could be garbage
        # collected before it runs.
        self._background_tasks: Set[asyncio.Task] = set()

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

    def _matched_handlers(self, event: PipelineEvent) -> List[EventHandler]:
        """Collect exact + wildcard + prefix matches, deduplicated.

        Shared by :meth:`emit` and :meth:`emit_sync` so the two
        dispatch paths can never drift on matching semantics.
        """
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

        return matched_handlers

    async def emit(self, event: PipelineEvent) -> None:
        """Emit an event to all matching handlers (deduplicated)."""
        for handler in self._matched_handlers(event):
            try:
                result = handler(event)
                # Support both sync and async handlers
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                # Event handlers should not crash the pipeline, but log the error
                logger.warning("Event handler %r failed on %s: %s", handler, event.type, e)

    def emit_sync(self, event: PipelineEvent) -> None:
        """Synchronous dispatch for producers that cannot await.

        Sync handlers run inline (synchronous delivery — a streaming
        collector's ``queue.put_nowait`` happens before this returns,
        exactly like the legacy ``state._event_listener`` path it
        replaces). Async handlers cannot be awaited from a sync frame;
        when an event loop is running they are scheduled as tasks
        (delivery happens on a later loop tick — ordering relative to
        sync handlers is NOT guaranteed for async handlers on this
        path), otherwise the coroutine is closed and a warning logged.
        Handler failures are contained, matching :meth:`emit`.
        """
        for handler in self._matched_handlers(event):
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        result.close()
                        logger.warning(
                            "EventBus.emit_sync: async handler %r for %s dropped — "
                            "no running event loop to schedule it on",
                            handler,
                            event.type,
                        )
                    else:
                        task = loop.create_task(result)
                        self._background_tasks.add(task)
                        task.add_done_callback(self._background_tasks.discard)
            except Exception as e:
                logger.warning("Event handler %r failed on %s: %s", handler, event.type, e)

    def clear(self) -> None:
        """Remove all handlers."""
        self._handlers.clear()
