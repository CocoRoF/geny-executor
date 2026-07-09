"""Regression: embedding clients must be loop-safe.

An httpx-backed SDK client (AsyncOpenAI / genai.Client) binds its transport
to the event loop that first drives it and cannot be reused from another
loop. Hosts that drive memory writes through a sync->async bridge spin a
fresh event loop PER CALL, so a naively-cached client (a) fails cross-loop
on every bridged embed with "Event loop is closed" and (b), never being
closed, leaks its socket pool once per session.

``_LoopBoundClientMixin`` fixes this: it caches one client on the stable
loop (pooled/reused), hands any other live loop a short-lived client it
closes within the call, and drops a dead-loop cache so an all-bridge caller
never accumulates references. These tests pin that behaviour without needing
the real SDKs or a network — a fake SDK client asserts it is only ever
driven on the loop it was built on.
"""

from __future__ import annotations

import asyncio
import threading

from geny_executor.memory.embedding.client import _LoopBoundClientMixin


class _FakeSDKClient:
    """Stand-in for AsyncOpenAI: records its build loop and asserts it is
    never driven from a different loop (the exact cross-loop failure the
    mixin exists to prevent)."""

    def __init__(self, registry: dict) -> None:
        self.build_loop = asyncio.get_running_loop()
        self.closed = False
        registry["built"] += 1

    async def do_call(self) -> None:
        assert asyncio.get_running_loop() is self.build_loop, (
            "embedding client driven on a different loop than it was built on"
        )

    async def close(self) -> None:
        self.closed = True


class _FakeLoopBoundClient(_LoopBoundClientMixin):
    """Minimal client using the mixin, with a counting fake SDK client."""

    def __init__(self, injected: object | None = None) -> None:
        self._client = injected
        self._client_loop = None
        self._injected_client = injected is not None
        self._registry = {"built": 0}

    def _build_client(self) -> object:
        return _FakeSDKClient(self._registry)

    async def embed(self, texts):
        client, ephemeral = self._acquire_client()
        try:
            await client.do_call()  # asserts same-loop use
            return [[0.0]] * len(texts)
        finally:
            if ephemeral:
                await self._aclose_client(client)

    async def close(self) -> None:
        # Same delegation the real OpenAI/Google clients use.
        await self._close_cached_client()

    @property
    def build_count(self) -> int:
        return self._registry["built"]


async def test_stable_loop_reuses_cached_client():
    """Repeated embeds on the same (stable) loop reuse one pooled client."""
    c = _FakeLoopBoundClient()
    await c.embed(["a"])
    await c.embed(["b"])
    await c.embed(["c"])
    assert c.build_count == 1  # built once, reused — connection pooling kept


async def test_other_live_loop_gets_ephemeral_closed_client():
    """A call on a DIFFERENT still-live loop must get its own short-lived
    client, closed within the call, without disturbing the cache."""
    c = _FakeLoopBoundClient()
    await c.embed(["main"])              # caches on this (test) loop
    assert c.build_count == 1
    cached = c._client

    captured: dict = {}

    def worker() -> None:
        loop = asyncio.new_event_loop()
        try:
            # This loop != the cached loop, and the cached loop is still
            # alive (test thread is blocked in join), so → ephemeral.
            loop.run_until_complete(c.embed(["worker"]))
            captured["ok"] = True
        except Exception as exc:  # noqa: BLE001
            captured["err"] = repr(exc)
        finally:
            loop.close()

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert captured.get("ok"), captured.get("err")
    assert c.build_count == 2          # cached + one ephemeral
    assert c._client is cached         # cache untouched
    # The cached (test-loop) client keeps being reused afterwards:
    await c.embed(["main2"])
    assert c.build_count == 2


def test_ephemeral_loops_never_cross_loop_and_do_not_accumulate():
    """All-bridge caller: every call on a fresh, then-dead loop. The mixin
    rebinds each round (dead-loop cache dropped, not reused → no cross-loop
    error) and never retains more than the single current client."""
    c = _FakeLoopBoundClient()
    for _ in range(4):
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(c.embed(["x"]))  # do_call asserts same-loop
        finally:
            loop.close()
    # Rebuilt each round (previous loop was dead) rather than reusing a
    # dead-loop client, and only ONE reference is held now (old ones were
    # dereferenced → GC finalizes their transports; no growing retention).
    assert c.build_count == 4
    held = c._client
    assert held is not None
    # A fifth call on yet another fresh loop still holds exactly one.
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(c.embed(["y"]))
    finally:
        loop.close()
    assert c.build_count == 5
    assert c._client is not held  # rebound, previous reference released


async def test_injected_client_is_used_verbatim_and_not_auto_closed():
    """A test-injected client is used as-is, never rebuilt, and close()
    leaves it to the caller."""
    reg = {"built": 0}
    injected = _FakeSDKClient(reg)  # built on this loop
    c = _FakeLoopBoundClient(injected=injected)
    await c.embed(["z"])
    assert c.build_count == 0          # never called _build_client
    assert c._client is injected
    await c.close()
    assert injected.closed is False    # caller owns injected client's lifetime


async def test_close_releases_cached_client():
    """close() closes the lazily-built cached client (the per-session leak
    fix) and clears the cache."""
    c = _FakeLoopBoundClient()
    await c.embed(["a"])
    cached = c._client
    assert cached is not None
    await c.close()
    assert cached.closed is True
    assert c._client is None
