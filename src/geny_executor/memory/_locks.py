"""Loop-agnostic lock for memory stores.

Why this exists
---------------
``asyncio.Lock`` instances bind to whichever event loop calls
``acquire()`` first. Subsequent acquires from a different loop raise
``RuntimeError: Future attached to a different loop`` — silently
swallowed in many code paths and producing empty snapshots / missing
writes that look like data corruption.

Hosts that drive the executor's stores from a sync context (e.g.
Geny's legacy ``run_coro_sync`` worker-thread bridge, or any code
that does ``asyncio.run(...)`` inside a thread pool) routinely create
short-lived event loops per call. Each new loop attempting to acquire
the same ``asyncio.Lock`` triggers the cross-loop error.

This wrapper backs the lock with a ``threading.Lock`` (loop-agnostic)
while preserving the ``async with`` surface so existing call sites
need no changes. The acquire is synchronous — the memory subsystem
holds the lock only for short disk reads/writes, so blocking the
event loop briefly is acceptable. The pattern is identical to what
ContextVar-aware async libraries already do for fast critical
sections.

Use ``LoopAgnosticLock()`` everywhere ``asyncio.Lock()`` was used in
provider stores.
"""

from __future__ import annotations

import asyncio
import threading
from types import TracebackType
from typing import Optional, Type


class LoopAgnosticLock:
    """``asyncio.Lock``-compatible mutex backed by ``threading.Lock``.

    Loop-agnostic: safe under cross-loop access (host code calling the
    executor's async store methods from a worker-thread loop, then
    again from the main pipeline loop). The underlying mutex never
    binds to a specific event loop.

    Not reentrant — same as ``asyncio.Lock``. A coroutine that already
    holds the lock and re-acquires from the same task will deadlock,
    so callers must never nest ``async with self._lock:`` for the
    same lock instance.

    **Acquire must never block the event loop.** Store critical sections
    hold this lock across ``await`` points (e.g. ``vector_store.index``
    awaits the embedding HTTP call while holding it). If a second
    coroutine on the SAME loop then acquired synchronously, it would
    freeze the loop thread — the holder could never resume to release,
    deadlocking the whole process. So the async acquire below yields to
    the loop instead of blocking it: a fast non-blocking try first, and
    only on contention does it offload the blocking wait to a worker
    thread (keeping the loop free so the current holder can finish).
    """

    __slots__ = ("_lock",)

    def __init__(self) -> None:
        self._lock = threading.Lock()

    async def _acquire_without_blocking_loop(self) -> None:
        # Uncontended fast path — no thread hop.
        if self._lock.acquire(blocking=False):
            return
        # Contended: wait in a worker thread so the event loop keeps running
        # (and the current holder — possibly a coroutine on THIS loop that
        # yielded mid-critical-section — can make progress and release).
        await asyncio.to_thread(self._lock.acquire)

    async def __aenter__(self) -> "LoopAgnosticLock":
        await self._acquire_without_blocking_loop()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self._lock.release()

    async def acquire(self) -> bool:
        await self._acquire_without_blocking_loop()
        return True

    def release(self) -> None:
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()


__all__ = ["LoopAgnosticLock"]
