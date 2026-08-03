"""Dedicated worker pool for the memory subsystem's internal offloads.

Why not ``asyncio.to_thread``
-----------------------------
``LoopAgnosticLock`` parks CONTENDED acquirers in the loop's default
thread pool (``to_thread(lock.acquire)``) so the loop stays free. If a
lock/gate HOLDER then runs its own work via ``to_thread`` too, holder and
waiters compete for the same bounded pool — and on small machines
(GitHub CI runners: 2 vCPUs → default pool ≈ 6 workers) a burst of
waiters can fill every slot while the holder's work item sits queued
behind them forever: a starvation deadlock. Observed live: the 2.64.3
release pipeline hung twice in ``Verify tests pass`` on exactly this
shape (10 concurrent deletes → 9 pooled waiters + 1 queued build),
while 32-core dev machines sailed through.

Running the subsystem's own offloads on this small dedicated pool makes
holder progress independent of waiter pressure: the holder finishes,
releases, and the pooled waiters drain. Two workers are plenty — these
are serialized-by-lock jobs (vault scan, sidecar build), not a fan-out.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, TypeVar

T = TypeVar("T")

_MEM_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mem-offload")


async def run_offloaded(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run ``fn`` on the memory subsystem's dedicated worker pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_MEM_EXECUTOR, partial(fn, *args, **kwargs))


__all__ = ["run_offloaded"]
