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
    """

    __slots__ = ("_lock",)

    def __init__(self) -> None:
        self._lock = threading.Lock()

    async def __aenter__(self) -> "LoopAgnosticLock":
        # Synchronous acquire. Memory writes inside are short-lived
        # disk ops; blocking the event loop here matches the cost of
        # the I/O itself. If a future caller needs non-blocking
        # acquire, swap to ``await asyncio.to_thread(self._lock.acquire)``.
        self._lock.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self._lock.release()

    async def acquire(self) -> bool:
        self._lock.acquire()
        return True

    def release(self) -> None:
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()


__all__ = ["LoopAgnosticLock"]
