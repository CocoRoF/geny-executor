"""Hierarchical sidecar incremental refresh (EXEC-5 / 1.21.0).

Verifies that the executor's ``_FileIndexStore`` keeps:

- ``<root>/_index.json`` — bounded folder-tree summary (no per-note metadata)
- ``<cat>/_index.json``  — per-category shard with note-level detail

in lockstep with note writes/updates/deletes. The host no longer has
to operate a parallel sidecar writer; pre-1.21.0 the root file was a
flat dump that grew unbounded plus a separate ``_summary.json`` that
duplicated the folder summary at smaller scale.
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


# ── root folder-tree summary (1.21.0) ──────────────────────────────


def test_root_index_is_folder_summary_not_flat_dump():
    """Root ``<memory>/_index.json`` is a folder-tree summary —
    bounded by category count, NOT by note count. Per-note metadata
    lives only inside category shards.
    """
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
        root_index = _read_json(root / "memory" / "_index.json")
        # Top-level keys are summary-shaped, NOT flat-dump-shaped.
        assert "categories" in root_index
        assert "category_descriptions" in root_index
        assert "files" not in root_index, (
            "root _index.json must NOT contain a per-note `files` dict"
        )
        assert "tag_map" not in root_index
        assert "link_graph" not in root_index
        # Category list contains the canonical entries with file counts.
        names = {c["name"] for c in root_index["categories"]}
        for canonical in ("topics", "insights", "projects",
                          "daily", "dms", "conversations", "compactions"):
            assert canonical in names, f"missing canonical category: {canonical}"
        topics = next(c for c in root_index["categories"] if c["name"] == "topics")
        assert topics["file_count"] == 1
        assert topics["description"] == "subject pages"
        assert topics["path"] == "memory/topics"


def test_root_index_updates_count_on_delete():
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
        root_index = _read_json(root / "memory" / "_index.json")
        topics = next(c for c in root_index["categories"] if c["name"] == "topics")
        assert topics["file_count"] == 1


def test_summary_json_not_written_in_1_21():
    """Pre-1.21.0 the executor wrote ``<root>/_summary.json`` alongside
    the flat root index. 1.21.0 collapses both into ``<root>/_index.json``.
    The legacy summary file must NOT be created.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        p = _make_provider(root)

        async def go():
            await p.initialize()
            await p.notes().write(
                NoteDraft(title="T1", body="b", category="topics", filename="t1.md")
            )

        _run(go())
        legacy = root / "memory" / "_summary.json"
        assert not legacy.exists(), (
            "_summary.json was retired in 1.21.0; the folder summary "
            "is now at <root>/_index.json"
        )


def test_snapshot_returns_full_in_memory_view_without_disk_dump():
    """``snapshot()`` returns the per-note view in memory (callers that
    need it use this) — but the disk root index stays bounded.
    """
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
            return await p.index().snapshot()

        snap = _run(go())
        # In-memory snapshot has per-note `files` dict.
        assert "t1.md" in snap["files"]
        assert "c1.md" in snap["files"]
        assert snap["files"]["t1.md"]["category"] == "topics"
        # But the disk root index stays bounded.
        root_index = _read_json(root / "memory" / "_index.json")
        assert "files" not in root_index


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
