"""``IndexHandle`` progressive-disclosure surface (EXEC-3).

Verifies the 4-step read chain across every provider implementation:

1. ``list_categories`` → folder + count
2. ``list_notes(category)`` → ``NoteSummary`` per note
3. ``read_outline(filename)`` → markdown heading tree
4. ``read_section(filename, heading)`` → body slice
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from geny_executor.memory.provider import (
    Importance,
    NoteDraft,
    NoteOutline,
    NoteSummary,
    Scope,
)
from geny_executor.memory.providers.ephemeral import EphemeralMemoryProvider
from geny_executor.memory.providers.file.provider import FileMemoryProvider


def _run(coro):
    return asyncio.run(coro)


_BODY = """\
# Project Plan

This is the introduction paragraph that explains
what the document is about.

## Goals

We want to:

- ship X
- track Y

## Risks

Risks include:

- vendor lock
- timeline slip

### Mitigations

We mitigate by:

- spike spike
- spike

## Decisions

Final decisions go here.
"""


@pytest.fixture
def file_provider_with_notes(tmp_path):
    p = FileMemoryProvider(root=tmp_path, scope=Scope.SESSION, timezone_name="UTC")

    async def setup():
        await p.initialize()
        await p.notes().write(
            NoteDraft(
                title="Project Plan",
                body=_BODY,
                category="projects",
                filename="plan.md",
                importance=Importance.HIGH,
                tags=["roadmap", "Q3"],
            )
        )
        await p.notes().write(
            NoteDraft(
                title="Tiny note",
                body="just one paragraph",
                category="projects",
                filename="tiny.md",
                importance=Importance.LOW,
                tags=["q3"],
            )
        )

    _run(setup())
    return p


# ── list_notes ───────────────────────────────────────────────────────


def test_list_notes_filters_by_category(file_provider_with_notes):
    p = file_provider_with_notes
    notes = _run(p.index().list_notes(category="projects"))
    assert {n.filename for n in notes} == {"plan.md", "tiny.md"}
    plan = next(n for n in notes if n.filename == "plan.md")
    assert plan.title == "Project Plan"
    assert plan.category == "projects"
    assert plan.tags == ["roadmap", "Q3"]
    assert plan.first_paragraph.startswith("This is the introduction")
    assert plan.char_count > 0


def test_list_notes_filters_by_tag(file_provider_with_notes):
    p = file_provider_with_notes
    notes = _run(p.index().list_notes(tag="roadmap"))
    assert [n.filename for n in notes] == ["plan.md"]


def test_list_notes_pagination(file_provider_with_notes):
    p = file_provider_with_notes
    page1 = _run(p.index().list_notes(category="projects", limit=1, offset=0))
    page2 = _run(p.index().list_notes(category="projects", limit=1, offset=1))
    assert len(page1) == 1
    assert len(page2) == 1
    assert page1[0].filename != page2[0].filename


# ── read_outline ─────────────────────────────────────────────────────


def test_read_outline_builds_heading_tree(file_provider_with_notes):
    p = file_provider_with_notes
    outline = _run(p.index().read_outline("plan.md"))
    assert isinstance(outline, NoteOutline)
    assert outline.filename == "plan.md"
    headings = outline.headings
    h1 = [h for h in headings if h.level == 1]
    assert len(h1) == 1
    assert h1[0].heading == "Project Plan"
    # Top-level h1 has 3 h2 children: Goals, Risks, Decisions
    children_h2 = [c.heading for c in h1[0].children if c.level == 2]
    assert children_h2 == ["Goals", "Risks", "Decisions"]
    # Risks has Mitigations as h3 child
    risks_node = next(c for c in h1[0].children if c.heading == "Risks")
    assert any(c.heading == "Mitigations" and c.level == 3 for c in risks_node.children)


def test_read_outline_missing_file_returns_none(file_provider_with_notes):
    p = file_provider_with_notes
    assert _run(p.index().read_outline("nope.md")) is None


# ── read_section ─────────────────────────────────────────────────────


def test_read_section_extracts_h2_body(file_provider_with_notes):
    p = file_provider_with_notes
    body = _run(p.index().read_section("plan.md", "Goals"))
    assert body is not None
    assert "ship X" in body
    assert "track Y" in body
    # Should NOT include the next h2 ("## Risks") body
    assert "vendor lock" not in body


def test_read_section_extracts_h3_body(file_provider_with_notes):
    p = file_provider_with_notes
    body = _run(p.index().read_section("plan.md", "Mitigations"))
    assert body is not None
    assert "spike" in body
    assert "vendor lock" not in body


def test_read_section_case_insensitive(file_provider_with_notes):
    p = file_provider_with_notes
    body = _run(p.index().read_section("plan.md", "goals"))
    assert body is not None and "ship X" in body


def test_read_section_missing_heading_returns_none(file_provider_with_notes):
    p = file_provider_with_notes
    assert _run(p.index().read_section("plan.md", "Schedule")) is None


# ── ephemeral parity ────────────────────────────────────────────────


def test_ephemeral_progressive_disclosure_chain():
    p = EphemeralMemoryProvider()

    async def go():
        await p.initialize()
        await p.notes().write(
            NoteDraft(
                title="Plan",
                body=_BODY,
                category="topics",
                filename="plan.md",
                importance=Importance.HIGH,
                tags=["a"],
            )
        )
        cats = await p.index().list_categories()
        names = {c["name"] for c in cats}
        assert "topics" in names
        notes = await p.index().list_notes(category="topics")
        assert isinstance(notes[0], NoteSummary)
        outline = await p.index().read_outline("plan.md")
        assert outline is not None and outline.headings
        body = await p.index().read_section("plan.md", "Goals")
        assert body is not None and "ship X" in body

    _run(go())
