"""A cancelled acquirer must not walk off with the lock.

Production, 2026-08-11. Turns stopped answering and stayed that way
until a restart. Three separate stacks — the turn's own memory write,
the compaction kick, the conversation archiver — were all parked in
``LoopAgnosticLock._acquire_without_blocking_loop``, and no task in the
process held the lock.

A worker thread blocked in ``lock.acquire()`` cannot be cancelled.
Cancelling the *await* abandoned the waiter while the thread went on to
take the mutex on behalf of a caller that no longer existed:
``__aenter__`` never returned, so ``__aexit__`` never ran, and the lock
was held forever by nobody. Every later acquirer parked behind it.

The trigger was mundane and frequent — a host-side stall guard
abandoning a slow turn. One stall leaked the lock; from then on every
turn stalled.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor.memory._locks import LoopAgnosticLock


async def _acquire_and_release(lock: LoopAgnosticLock) -> None:
    async with lock:
        pass


@pytest.mark.asyncio
async def test_cancelling_a_contended_acquire_does_not_strand_the_lock():
    """The exact production shape, minimised."""
    lock = LoopAgnosticLock()
    holder_may_release = asyncio.Event()
    holder_has_lock = asyncio.Event()

    async def holder():
        async with lock:
            holder_has_lock.set()
            await holder_may_release.wait()

    h = asyncio.create_task(holder())
    await holder_has_lock.wait()

    c = asyncio.create_task(_acquire_and_release(lock))
    await asyncio.sleep(0.05)          # let it reach the thread hop
    c.cancel()                          # ← the stall guard, in production
    with pytest.raises(asyncio.CancelledError):
        await c

    holder_may_release.set()
    await h

    # The lock must be free. Before the fix the worker thread took it on
    # behalf of the cancelled contender and nothing ever released it.
    await asyncio.wait_for(_acquire_and_release(lock), timeout=5.0)
    assert not lock.locked()


@pytest.mark.asyncio
async def test_repeated_cancellations_do_not_accumulate():
    """One leak is already fatal; ten makes sure the release path itself
    does not go wrong under repetition."""
    lock = LoopAgnosticLock()
    release = asyncio.Event()
    got_it = asyncio.Event()

    async def holder():
        async with lock:
            got_it.set()
            await release.wait()

    h = asyncio.create_task(holder())
    await got_it.wait()

    for _ in range(10):
        c = asyncio.create_task(_acquire_and_release(lock))
        await asyncio.sleep(0.01)
        c.cancel()
        with pytest.raises(asyncio.CancelledError):
            await c

    release.set()
    await h
    await asyncio.wait_for(_acquire_and_release(lock), timeout=5.0)
    assert not lock.locked()


@pytest.mark.asyncio
async def test_uncancelled_contention_still_works():
    """The fix must not break what the lock is for: a contended acquirer
    waits, then gets it, and not before."""
    lock = LoopAgnosticLock()
    order = []
    release = asyncio.Event()
    got_it = asyncio.Event()

    async def holder():
        async with lock:
            got_it.set()
            order.append("holder-in")
            await release.wait()
            order.append("holder-out")

    async def waiter():
        async with lock:
            order.append("waiter-in")

    h = asyncio.create_task(holder())
    await got_it.wait()
    w = asyncio.create_task(waiter())
    await asyncio.sleep(0.05)
    assert order == ["holder-in"], "the waiter took a held lock"
    release.set()
    await asyncio.wait_for(asyncio.gather(h, w), timeout=5.0)
    assert order == ["holder-in", "holder-out", "waiter-in"]


@pytest.mark.asyncio
async def test_the_loop_keeps_running_while_contended():
    """The reason for the thread hop in the first place — a contended
    acquire must not freeze the event loop."""
    lock = LoopAgnosticLock()
    release = asyncio.Event()
    got_it = asyncio.Event()
    ticks = 0

    async def ticker():
        nonlocal ticks
        for _ in range(5):
            await asyncio.sleep(0.01)
            ticks += 1

    async def holder():
        async with lock:
            got_it.set()
            await release.wait()

    h = asyncio.create_task(holder())
    await got_it.wait()
    w = asyncio.create_task(_acquire_and_release(lock))
    await ticker()
    assert ticks == 5, "the loop stopped while an acquire was contended"
    release.set()
    await asyncio.wait_for(asyncio.gather(h, w), timeout=5.0)
