"""Sidecar refresh must be coalesced and off-loop; batch re-index must skip
unchanged notes.

Field incident (2026-08-03, prod): an observation prune sweep deleted notes
one-by-one and every delete ran a full-vault sidecar payload build inline on
the event loop — a 6k-note vault blocked the loop for tens of seconds, the
host was watchdog-restarted mid-sweep, and every resume re-embedded the whole
vault (minutes of embedding HTTP per 30-min idle-evict cycle). These tests
lock in the fixes.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

import pytest

from geny_executor.memory.embedding import LocalHashEmbeddingClient
from geny_executor.memory.provider import NoteDraft, Scope
from geny_executor.memory.providers.file.provider import FileMemoryProvider


def _provider(td: str) -> FileMemoryProvider:
    return FileMemoryProvider(root=Path(td), scope=Scope.SESSION)


@pytest.mark.asyncio
async def test_burst_of_deletes_coalesces_payload_builds(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        p = _provider(td)
        await p.initialize()
        for i in range(12):
            await p.notes().write(
                NoteDraft(title=f"n{i}", body=f"b{i}", category="observations", filename=f"n{i}.md")
            )

        index = p.index()
        builds = {"n": 0}
        real_build = index._payload_from_notes

        def counting_build(notes):
            builds["n"] += 1
            time.sleep(0.02)  # make the build observable
            return real_build(notes)

        monkeypatch.setattr(index, "_payload_from_notes", counting_build)

        # Concurrent burst — the prune-sweep shape.
        await asyncio.gather(*(p.notes().delete(f"n{i}.md") for i in range(10)))

        # Coalescing: a 10-delete burst must not run 10 builds. (≤ half is
        # the contract; typically 2-3 depending on gate timing.)
        assert builds["n"] <= 5, f"sidecar builds not coalesced: {builds['n']} for 10 deletes"


@pytest.mark.asyncio
async def test_sidecar_refresh_keeps_loop_responsive(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        p = _provider(td)
        await p.initialize()
        for i in range(5):
            await p.notes().write(
                NoteDraft(title=f"m{i}", body=f"b{i}", category="topics", filename=f"m{i}.md")
            )
        index = p.index()
        real_build = index._payload_from_notes

        def slow_build(notes):
            time.sleep(0.3)  # heavy vault stand-in — MUST run off-loop
            return real_build(notes)

        monkeypatch.setattr(index, "_payload_from_notes", slow_build)

        beats = 0

        async def heartbeat():
            nonlocal beats
            while True:
                beats += 1
                await asyncio.sleep(0.01)

        hb = asyncio.create_task(heartbeat())
        try:
            await p.notes().delete("m0.md")  # triggers refresh (slow build)
        finally:
            hb.cancel()
        assert beats >= 10, f"event loop starved during sidecar refresh (beats={beats})"


@pytest.mark.asyncio
async def test_index_batch_skips_unchanged_notes():
    from geny_executor.memory.provider import NoteRef

    with tempfile.TemporaryDirectory() as td:
        p = FileMemoryProvider(
            root=Path(td), scope=Scope.SESSION,
            embedding_client=LocalHashEmbeddingClient(),
        )
        await p.initialize()
        vec = p.vector()
        assert vec is not None

        embeds = {"n": 0}
        real_embed = vec._embed_guarded

        async def counting_embed(texts):
            embeds["n"] += len(texts)
            return await real_embed(texts)

        vec._embed_guarded = counting_embed  # type: ignore[method-assign]

        items = [
            (NoteRef(filename=f"k{i}.md", scope=Scope.SESSION, backend="filesystem"), f"body {i}")
            for i in range(6)
        ]
        await vec.index_batch(items)
        first = embeds["n"]
        assert first == 6

        # Same content again — the resume-warm-up shape: zero embeds.
        await vec.index_batch(items)
        assert embeds["n"] == first, "unchanged notes were re-embedded"

        # One changed note — exactly one embed.
        items[3] = (items[3][0], "body 3 CHANGED")
        await vec.index_batch(items)
        assert embeds["n"] == first + 1
