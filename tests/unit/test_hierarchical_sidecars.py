"""Hierarchical sidecar incremental refresh (EXEC-5).

Verifies that the executor's ``_FileIndexStore`` keeps the per-category
``<cat>/_index.json`` shards and the root ``_summary.json`` overview
in lockstep with note writes/updates/deletes — the host no longer has
to operate a parallel sidecar writer.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from geny_executor.memory.provider import (
    Importance,
    NoteDraft,
    NotePatch,
    Scope,
)
from geny_executor.memory.providers.file.provider import FileMemoryProvider


def _run(coro):
    return asyncio.run(coro)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _make_provider(tmp: Path, *, descriptions=None) -> FileMemoryProvider:
    return FileMemoryProvider(
        root=tmp,
        scope=Scope.SESSION,
        timezone_name="UTC",
        category_descriptions=descriptions,
    )


# ── per-category shard ─────────────────────────────────────────────


def test_shard_created_after_first_write():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        p = _make_provider(root)

        async def go():
            await p.initialize()
            await p.notes().write(
                NoteDraft(
                    title="T1",
                    body="body",
                    category="topics",
                    filename="t1.md",
                    importance=Importance.HIGH,
                    tags=["alpha"],
                )
            )

        _run(go())
        shard = root / "memory" / "topics" / "_index.json"
        assert shard.exists(), "topics shard should be created on write"
        payload = _read_json(shard)
        assert payload["category"] == "topics"
        assert payload["file_count"] == 1
        assert "t1.md" in payload["files"]
        assert payload["tag_counts"].get("alpha") == 1


def test_shard_updates_incrementally_on_subsequent_write():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        p = _make_provider(root)

        async def go():
            await p.initialize()
            await p.notes().write(
                NoteDraft(title="T1", body="b", category="topics", filename="t1.md")
            )
            await p.notes().write(
                NoteDraft(title="T2", body="b", category="topics", filename="t2.md")
            )
            await p.notes().write(
                NoteDraft(title="C1", body="b", category="critical", filename="c1.md")
            )

        _run(go())
        topics_shard = _read_json(root / "memory" / "topics" / "_index.json")
        critical_shard = _read_json(root / "memory" / "critical" / "_index.json")
        assert topics_shard["file_count"] == 2
        assert critical_shard["file_count"] == 1
        # Critical shard must NOT contain topics' notes (incremental — no cross-talk)
        assert "t1.md" not in critical_shard["files"]
        assert "c1.md" not in topics_shard["files"]


def test_shard_drops_note_on_delete():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        p = _make_provider(root)

        async def go():
            await p.initialize()
            await p.notes().write(
                NoteDraft(title="T1", body="b", category="topics", filename="t1.md")
            )
            await p.notes().write(
                NoteDraft(title="T2", body="b", category="topics", filename="t2.md")
            )
            assert await p.notes().delete("t1.md") is True

        _run(go())
        shard = _read_json(root / "memory" / "topics" / "_index.json")
        assert shard["file_count"] == 1
        assert "t1.md" not in shard["files"]
        assert "t2.md" in shard["files"]


def test_shard_reflects_update_metadata_change():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        p = _make_provider(root)

        async def go():
            await p.initialize()
            await p.notes().write(
                NoteDraft(
                    title="T1",
                    body="b",
                    category="topics",
                    filename="t1.md",
                    tags=["initial"],
                )
            )
            await p.notes().update("t1.md", NotePatch(tags=["updated"]))

        _run(go())
        shard = _read_json(root / "memory" / "topics" / "_index.json")
        assert shard["tag_counts"].get("updated") == 1
        assert "initial" not in shard["tag_counts"]


# ── root summary ───────────────────────────────────────────────────


def test_root_summary_lists_every_category():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        p = _make_provider(
            root,
            descriptions={"topics": "subject pages", "critical": "always-pinned facts"},
        )

        async def go():
            await p.initialize()
            await p.notes().write(
                NoteDraft(title="T1", body="b", category="topics", filename="t1.md")
            )

        _run(go())
        summary_path = root / "memory" / "_summary.json"
        assert summary_path.exists()
        summary = _read_json(summary_path)
        names = {c["name"] for c in summary["categories"]}
        # Canonical empty categories from NOTE_CATEGORIES appear (so
        # the sidebar always shows them — Geny's UX requirement).
        # ``critical`` is host-defined (not canonical) and only
        # surfaces once the host writes a note into it.
        for canonical in ("topics", "insights", "projects",
                          "daily", "dms", "conversations", "compactions"):
            assert canonical in names, f"missing canonical category: {canonical}"
        topics = next(c for c in summary["categories"] if c["name"] == "topics")
        assert topics["file_count"] == 1
        assert topics["description"] == "subject pages"


def test_root_summary_updates_count_on_delete():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        p = _make_provider(root)

        async def go():
            await p.initialize()
            await p.notes().write(
                NoteDraft(title="T1", body="b", category="topics", filename="t1.md")
            )
            await p.notes().write(
                NoteDraft(title="T2", body="b", category="topics", filename="t2.md")
            )
            await p.notes().delete("t1.md")

        _run(go())
        summary = _read_json(root / "memory" / "_summary.json")
        topics = next(c for c in summary["categories"] if c["name"] == "topics")
        assert topics["file_count"] == 1


# ── root flat index unaffected by hierarchical writes ──────────────


def test_root_flat_index_still_contains_all_files():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        p = _make_provider(root)

        async def go():
            await p.initialize()
            await p.notes().write(
                NoteDraft(title="T1", body="b", category="topics", filename="t1.md")
            )
            await p.notes().write(
                NoteDraft(title="C1", body="b", category="critical", filename="c1.md")
            )

        _run(go())
        flat = _read_json(root / "memory" / "_index.json")
        assert "t1.md" in flat["files"]
        assert "c1.md" in flat["files"]
        # Flat index is NOT a shard — must keep ``files`` (not just a
        # category-scoped slice).
        assert flat["files"]["t1.md"]["category"] == "topics"


# ── set_hooks updates descriptions live ─────────────────────────────


def test_set_hooks_updates_descriptions_on_next_refresh():
    from geny_executor.memory.provider import MemoryHooks

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        p = _make_provider(root)

        async def go():
            await p.initialize()
            await p.notes().write(
                NoteDraft(title="T1", body="b", category="topics", filename="t1.md")
            )
            p.set_hooks(MemoryHooks(vault_descriptions={"topics": "from-hook"}))
            # Trigger a refresh by writing again
            await p.notes().write(
                NoteDraft(title="T2", body="b", category="topics", filename="t2.md")
            )

        _run(go())
        shard = _read_json(root / "memory" / "topics" / "_index.json")
        assert shard["description"] == "from-hook"
