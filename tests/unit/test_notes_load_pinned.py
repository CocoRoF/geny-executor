"""EXEC-4 — `NotesHandle.load_pinned` helper.

Concatenates notes in a category (default `critical`) into a
prompt-injectable string, sorted by importance then recency, with
char-budget cutoff. Hosts use this for the system prompt's pinned-
facts section.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from geny_executor.memory.provider import Importance, NoteDraft, Scope
from geny_executor.memory.providers.ephemeral import EphemeralMemoryProvider
from geny_executor.memory.providers.file.provider import FileMemoryProvider


def _run(coro):
    return asyncio.run(coro)


def test_load_pinned_empty_when_no_critical_notes():
    with tempfile.TemporaryDirectory() as td:
        provider = FileMemoryProvider(root=Path(td), scope=Scope.SESSION)

        async def go():
            await provider.initialize()
            await provider.notes().write(
                NoteDraft(title="T", body="not pinned", category="topics", filename="t.md")
            )
            return await provider.notes().load_pinned()

        assert _run(go()) == ""


def test_load_pinned_concatenates_notes_with_titles():
    with tempfile.TemporaryDirectory() as td:
        provider = FileMemoryProvider(root=Path(td), scope=Scope.SESSION)

        async def go():
            await provider.initialize()
            await provider.notes().write(
                NoteDraft(
                    title="User likes coffee",
                    body="Always offer coffee at start.",
                    category="critical",
                    filename="coffee.md",
                    importance=Importance.HIGH,
                )
            )
            await provider.notes().write(
                NoteDraft(
                    title="Address user as 'sir'",
                    body="Default address form is 'sir'.",
                    category="critical",
                    filename="address.md",
                    importance=Importance.CRITICAL,
                )
            )
            return await provider.notes().load_pinned(max_chars=10000)

        out = _run(go())
        # CRITICAL importance sorts before HIGH
        assert out.startswith("## Address user as 'sir'")
        assert "User likes coffee" in out
        assert "Always offer coffee" in out
        assert "Default address form" in out


def test_load_pinned_respects_max_chars():
    with tempfile.TemporaryDirectory() as td:
        provider = FileMemoryProvider(root=Path(td), scope=Scope.SESSION)

        async def go():
            await provider.initialize()
            for i in range(5):
                await provider.notes().write(
                    NoteDraft(
                        title=f"Fact {i}",
                        body="x" * 200,
                        category="critical",
                        filename=f"fact{i}.md",
                        importance=Importance.HIGH,
                    )
                )
            return await provider.notes().load_pinned(max_chars=400)

        out = _run(go())
        # Char budget enforcement — at most 1-2 notes fit
        assert len(out) <= 800  # generous upper bound including separators


def test_load_pinned_custom_category():
    with tempfile.TemporaryDirectory() as td:
        provider = FileMemoryProvider(root=Path(td), scope=Scope.SESSION)

        async def go():
            await provider.initialize()
            await provider.notes().write(
                NoteDraft(
                    title="Cycle plan",
                    body="ship today",
                    category="projects",
                    filename="plan.md",
                    importance=Importance.HIGH,
                )
            )
            return await provider.notes().load_pinned(
                category="projects", max_chars=1000,
            )

        out = _run(go())
        assert "ship today" in out


def test_ephemeral_provider_load_pinned():
    provider = EphemeralMemoryProvider()

    async def go():
        await provider.initialize()
        await provider.notes().write(
            NoteDraft(
                title="P",
                body="pinned body",
                category="critical",
                filename="p.md",
                importance=Importance.CRITICAL,
            )
        )
        return await provider.notes().load_pinned()

    assert "pinned body" in _run(go())


def test_load_pinned_zero_max_chars_returns_empty():
    with tempfile.TemporaryDirectory() as td:
        provider = FileMemoryProvider(root=Path(td), scope=Scope.SESSION)

        async def go():
            await provider.initialize()
            await provider.notes().write(
                NoteDraft(
                    title="P",
                    body="pinned",
                    category="critical",
                    filename="p.md",
                    importance=Importance.CRITICAL,
                )
            )
            return await provider.notes().load_pinned(max_chars=0)

        assert _run(go()) == ""
