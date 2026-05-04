"""Test that wikilink target normalization correctly populates `links_in`.

EXEC-3 — `_refresh_backlinks` previously keyed `link_map` by the raw
wikilink target ("target") while the cache keyed notes by their on-disk
filename ("target.md"). The mismatch left `target.links_in` empty for
every note regardless of how many other notes linked to it. This test
locks in the post-fix behaviour: wikilinks with or without the `.md`
suffix correctly produce backlinks.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from geny_executor.memory.provider import NoteDraft, Scope
from geny_executor.memory.providers.file.provider import FileMemoryProvider


@pytest.mark.asyncio
async def test_bare_wikilink_produces_backlink():
    with tempfile.TemporaryDirectory() as td:
        p = FileMemoryProvider(root=Path(td), scope=Scope.SESSION)
        await p.initialize()
        await p.notes().write(
            NoteDraft(title="Target", body="target body", category="topics", filename="target.md")
        )
        await p.notes().write(
            NoteDraft(title="Source", body="link [[target]]", category="topics", filename="source.md")
        )
        target = await p.notes().read("target.md")
        assert target is not None
        assert target.links_in == ["source.md"]


@pytest.mark.asyncio
async def test_full_filename_wikilink_produces_backlink():
    with tempfile.TemporaryDirectory() as td:
        p = FileMemoryProvider(root=Path(td), scope=Scope.SESSION)
        await p.initialize()
        await p.notes().write(
            NoteDraft(title="Target", body="t", category="topics", filename="target.md")
        )
        await p.notes().write(
            NoteDraft(title="Source", body="link [[target.md]]", category="topics", filename="source.md")
        )
        target = await p.notes().read("target.md")
        assert target is not None
        assert "source.md" in target.links_in


@pytest.mark.asyncio
async def test_unresolved_wikilink_does_not_break_others():
    """A link to a missing note keeps the link_map intact for resolvable ones."""
    with tempfile.TemporaryDirectory() as td:
        p = FileMemoryProvider(root=Path(td), scope=Scope.SESSION)
        await p.initialize()
        await p.notes().write(
            NoteDraft(title="A", body="a", category="topics", filename="a.md")
        )
        await p.notes().write(
            NoteDraft(
                title="B",
                body="links [[a]] and [[ghost]]",
                category="topics",
                filename="b.md",
            )
        )
        a = await p.notes().read("a.md")
        assert a is not None
        assert "b.md" in a.links_in


@pytest.mark.asyncio
async def test_backlinks_survive_disk_reload():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        p = FileMemoryProvider(root=root, scope=Scope.SESSION)
        await p.initialize()
        await p.notes().write(
            NoteDraft(title="T", body="t", category="topics", filename="t.md")
        )
        await p.notes().write(
            NoteDraft(title="S", body="[[t]]", category="topics", filename="s.md")
        )
        # Fresh provider on the same root
        p2 = FileMemoryProvider(root=root, scope=Scope.SESSION)
        await p2.initialize()
        t2 = await p2.notes().read("t.md")
        assert t2 is not None
        assert t2.links_in == ["s.md"]
