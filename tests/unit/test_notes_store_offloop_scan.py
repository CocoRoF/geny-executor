"""The full-vault scan must not block the event loop.

Field incident (2026-08-03): resuming a session with a 6.2k-note vault ran
``_ensure_loaded`` inline on the host's event loop for ~19s — health checks
stopped answering and the process was watchdog-restarted mid-load, so large
sessions could never come back up. ``_ensure_loaded`` now performs the scan
in a worker thread (``asyncio.to_thread``); these tests lock that in.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

import pytest

from geny_executor.memory.provider import NoteDraft, Scope
from geny_executor.memory.providers.file.provider import FileMemoryProvider


def _make_provider(td: str) -> FileMemoryProvider:
    return FileMemoryProvider(root=Path(td), scope=Scope.SESSION)


@pytest.mark.asyncio
async def test_slow_scan_keeps_loop_responsive(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        seed = _make_provider(td)
        await seed.initialize()
        for i in range(10):
            await seed.notes().write(
                NoteDraft(title=f"n{i}", body=f"body {i}", category="topics", filename=f"n{i}.md")
            )

        # Fresh provider → cold cache → list() triggers the full scan.
        p = _make_provider(td)
        await p.initialize()
        store = p.notes()
        real_load = store._load_note

        def slow_load(path):
            time.sleep(0.05)  # 10 notes ≈ 0.5s of blocking work if run on-loop
            return real_load(path)

        monkeypatch.setattr(store, "_load_note", slow_load)

        beats = 0

        async def heartbeat():
            nonlocal beats
            while True:
                beats += 1
                await asyncio.sleep(0.01)

        hb = asyncio.create_task(heartbeat())
        try:
            metas = await store.list()
        finally:
            hb.cancel()
        assert len(metas) == 10
        # On-loop scan would freeze the heartbeat for the full ~0.5s (≈0
        # beats); a threaded scan lets it tick throughout.
        assert beats >= 20, f"event loop starved during scan (beats={beats})"


@pytest.mark.asyncio
async def test_scan_results_identical_after_offload():
    with tempfile.TemporaryDirectory() as td:
        seed = _make_provider(td)
        await seed.initialize()
        await seed.notes().write(
            NoteDraft(title="Target", body="t", category="topics", filename="target.md")
        )
        await seed.notes().write(
            NoteDraft(title="Source", body="see [[target]]", category="topics", filename="source.md")
        )

        p = _make_provider(td)
        await p.initialize()
        metas = await p.notes().list()
        assert {m.ref.filename for m in metas} == {"target.md", "source.md"}
        target = await p.notes().read("target.md")
        assert target is not None and target.links_in == ["source.md"]
