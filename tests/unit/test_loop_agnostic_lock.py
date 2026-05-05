"""Regression: memory store locks must survive cross-loop access.

Hosts that drive the executor from a sync bridge (e.g. Geny's legacy
``run_coro_sync`` worker-thread pattern) create a fresh asyncio event
loop per call. Before 1.19.0 the stores used ``asyncio.Lock`` which
bind to the first loop and raise ``RuntimeError`` ("Future attached to
a different loop") on subsequent acquires from any other loop. The
errors were silently swallowed in many code paths and produced empty
snapshots / missing writes.

These tests reproduce the failure mode by simulating the worker-thread
bridge: every operation runs on a fresh thread + fresh loop. With
``LoopAgnosticLock`` (threading.Lock-backed) the operations all
succeed; if the lock regresses to ``asyncio.Lock``, the second
operation raises and the test fails.
"""

from __future__ import annotations

import asyncio
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

import pytest

from geny_executor.memory.provider import (
    Importance,
    NoteDraft,
    Scope,
    Turn,
)
from geny_executor.memory.providers.ephemeral import EphemeralMemoryProvider
from geny_executor.memory.providers.file.provider import FileMemoryProvider

_T = TypeVar("_T")


def _run_in_fresh_loop(coro: Awaitable[_T]) -> _T:
    """Execute ``coro`` on a brand-new event loop in a worker thread.

    Mirrors Geny's ``run_coro_sync`` worker-thread pattern exactly —
    each call gets its own thread + its own loop, so memory store
    locks must NOT bind to a single loop.
    """
    box: list[Any] = []

    def _runner() -> None:
        loop = asyncio.new_event_loop()
        try:
            box.append(loop.run_until_complete(coro))
        except BaseException as exc:  # noqa: BLE001
            box.append(exc)
        finally:
            loop.close()

    t = threading.Thread(target=_runner)
    t.start()
    t.join()
    result = box[0] if box else None
    if isinstance(result, BaseException):
        raise result
    return result


def test_file_provider_stm_append_across_loops() -> None:
    with tempfile.TemporaryDirectory() as td:
        provider = FileMemoryProvider(
            root=Path(td), scope=Scope.SESSION, timezone_name="UTC"
        )
        _run_in_fresh_loop(provider.initialize())

        # Append from three different loops — pre-fix this would raise
        # RuntimeError on the second call (lock bound to loop #1).
        for i in range(3):
            _run_in_fresh_loop(
                provider.stm().append(
                    Turn(
                        role="user",
                        content=f"msg-{i}",
                        timestamp=datetime.now(timezone.utc),
                    )
                )
            )

        jsonl = Path(td) / "transcripts" / "session.jsonl"
        assert jsonl.exists(), "session.jsonl should be created"
        lines = jsonl.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3, f"expected 3 jsonl lines, got {len(lines)}"


def test_file_provider_notes_and_index_across_loops() -> None:
    with tempfile.TemporaryDirectory() as td:
        provider = FileMemoryProvider(
            root=Path(td), scope=Scope.SESSION, timezone_name="UTC"
        )
        _run_in_fresh_loop(provider.initialize())

        # Note write on one loop
        _run_in_fresh_loop(
            provider.notes().write(
                NoteDraft(
                    title="T1",
                    body="body 1",
                    category="topics",
                    filename="t1.md",
                )
            )
        )
        # Snapshot on another loop — pre-fix raises here
        snap = _run_in_fresh_loop(provider.index().snapshot())
        assert "t1.md" in snap.get("files", {}), (
            "snapshot must include note written on a different loop"
        )

        # Second note on yet another loop
        _run_in_fresh_loop(
            provider.notes().write(
                NoteDraft(
                    title="T2",
                    body="body 2",
                    category="critical",
                    filename="t2.md",
                    importance=Importance.CRITICAL,
                )
            )
        )
        snap2 = _run_in_fresh_loop(provider.index().snapshot())
        files = snap2.get("files", {})
        assert "t1.md" in files and "t2.md" in files


def test_file_provider_list_categories_and_vault_map_across_loops() -> None:
    with tempfile.TemporaryDirectory() as td:
        provider = FileMemoryProvider(
            root=Path(td), scope=Scope.SESSION, timezone_name="UTC"
        )
        _run_in_fresh_loop(provider.initialize())
        _run_in_fresh_loop(
            provider.notes().write(
                NoteDraft(
                    title="K", body="b", category="insights", filename="k.md"
                )
            )
        )
        cats = _run_in_fresh_loop(provider.index().list_categories())
        names = {c["name"] for c in cats}
        assert "insights" in names

        vmap = _run_in_fresh_loop(provider.index().build_vault_map())
        assert "insights" in (vmap.get("categories") or {})


def test_ephemeral_provider_across_loops() -> None:
    """Ephemeral provider has no disk lock but the smoke must pass to
    confirm nothing else in the provider chain breaks under
    cross-loop drive.
    """
    p = EphemeralMemoryProvider()
    _run_in_fresh_loop(p.initialize())
    _run_in_fresh_loop(
        p.stm().append(
            Turn(role="user", content="hi", timestamp=datetime.now(timezone.utc))
        )
    )
    turns = _run_in_fresh_loop(p.stm().recent(n=10))
    assert len(turns) == 1


def test_concurrent_writes_serialize_correctly() -> None:
    """Multiple worker threads writing simultaneously must serialize
    through the threading-backed lock. No lost writes, no corruption.
    """
    import concurrent.futures as _cf

    with tempfile.TemporaryDirectory() as td:
        provider = FileMemoryProvider(
            root=Path(td), scope=Scope.SESSION, timezone_name="UTC"
        )
        _run_in_fresh_loop(provider.initialize())

        N = 20

        def _do_append(i: int) -> None:
            _run_in_fresh_loop(
                provider.stm().append(
                    Turn(
                        role="user",
                        content=f"concurrent-{i}",
                        timestamp=datetime.now(timezone.utc),
                    )
                )
            )

        with _cf.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(_do_append, range(N)))

        jsonl = Path(td) / "transcripts" / "session.jsonl"
        lines = jsonl.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == N, (
            f"all {N} concurrent writes must be persisted, got {len(lines)}"
        )
