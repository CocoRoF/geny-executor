"""EXEC — `IndexHandle.list_categories` for category-tree enumeration.

Hosts (Geny's Opsidian sidebar) need to render every category folder
including empty ones. The protocol's `snapshot()["files"]` only
covers categories with at least one note; this method also includes
empty folders so the sidebar isn't surprised by `topics` or
`insights` "vanishing" the moment they have 0 notes.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from geny_executor.memory.provider import (
    Importance,
    NoteDraft,
    Scope,
)
from geny_executor.memory.providers.ephemeral import EphemeralMemoryProvider
from geny_executor.memory.providers.file.provider import FileMemoryProvider


def _run(coro):
    return asyncio.run(coro)


def test_file_list_categories_includes_empty_canonical_folders():
    """Every NOTE_CATEGORIES entry appears with file_count=0 even
    before any note is written, because `initialize()` ensures the
    canonical category dirs exist."""
    with tempfile.TemporaryDirectory() as td:
        provider = FileMemoryProvider(root=Path(td), scope=Scope.SESSION)

        async def go():
            await provider.initialize()
            return await provider.index().list_categories()

        cats = _run(go())
        names = {c["name"] for c in cats}
        # Canonical categories from NOTE_CATEGORIES
        for name in ("daily", "topics", "projects", "insights",
                     "dms", "conversations", "compactions", "root"):
            assert name in names, f"missing canonical category: {name}"
        # All start with 0 files
        for entry in cats:
            assert entry["file_count"] == 0
            assert entry["exists"] is True


def test_file_list_categories_counts_files_per_category():
    with tempfile.TemporaryDirectory() as td:
        provider = FileMemoryProvider(root=Path(td), scope=Scope.SESSION)

        async def go():
            await provider.initialize()
            await provider.notes().write(
                NoteDraft(title="T", body="b", category="topics", filename="t.md")
            )
            await provider.notes().write(
                NoteDraft(title="C", body="b", category="critical",
                          filename="c.md", importance=Importance.CRITICAL)
            )
            await provider.notes().write(
                NoteDraft(title="C2", body="b", category="critical", filename="c2.md")
            )
            return await provider.index().list_categories()

        cats = _run(go())
        by_name = {c["name"]: c for c in cats}
        assert by_name["topics"]["file_count"] == 1
        assert by_name["critical"]["file_count"] == 2


def test_file_list_categories_includes_host_defined_categories():
    """Host-created categories (Geny's `critical`, `executions`)
    appear once they exist on disk, even before NOTE_CATEGORIES is
    updated."""
    with tempfile.TemporaryDirectory() as td:
        provider = FileMemoryProvider(root=Path(td), scope=Scope.SESSION)

        async def go():
            await provider.initialize()
            await provider.notes().write(
                NoteDraft(title="X", body="b", category="critical",
                          filename="x.md", importance=Importance.CRITICAL)
            )
            return await provider.index().list_categories()

        cats = _run(go())
        names = {c["name"] for c in cats}
        assert "critical" in names


def test_ephemeral_list_categories():
    p = EphemeralMemoryProvider()

    async def go():
        await p.initialize()
        await p.notes().write(
            NoteDraft(title="T", body="b", category="topics", filename="t.md")
        )
        return await p.index().list_categories()

    cats = _run(go())
    by_name = {c["name"]: c for c in cats}
    assert by_name["topics"]["file_count"] == 1
