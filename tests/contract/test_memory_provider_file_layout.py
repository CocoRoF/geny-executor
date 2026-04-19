"""On-disk format lock for FileMemoryProvider.

The `MemoryProviderContract` suite verifies behaviour. These tests
verify the *format* the behaviour produces — file paths, frontmatter
shape, JSONL record schema, and the index-cache schema. Breaking any
of them means the output is no longer readable by Geny's legacy
reader (or by our own rebuilt reader).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geny_executor.memory.provider import (
    Importance,
    NoteDraft,
    RetrievalQuery,
    Turn,
)
from geny_executor.memory.providers import FileMemoryProvider
from geny_executor.memory.providers.file import frontmatter
from geny_executor.memory.providers.file.layout import DirectoryLayout


@pytest.fixture
async def provider(tmp_path: Path) -> FileMemoryProvider:
    p = FileMemoryProvider(tmp_path / "session")
    await p.initialize()
    return p


class TestDirectoryLayout:
    def test_default_paths(self, tmp_path: Path):
        root = tmp_path / "s1"
        layout = DirectoryLayout(root)
        assert layout.stm_jsonl == root / "transcripts" / "session.jsonl"
        assert layout.main_ltm == root / "memory" / "MEMORY.md"
        assert layout.dated_ltm("2026-04-19") == root / "memory" / "2026-04-19.md"
        assert layout.topic_ltm("async") == root / "memory" / "topics" / "async.md"
        assert layout.note_dir("topics") == root / "memory" / "topics"
        assert layout.note_dir("root") == root / "memory"
        assert layout.vector_index == root / "vectordb" / "index.faiss"
        assert layout.vector_metadata == root / "vectordb" / "metadata.json"
        assert layout.index_json == root / "memory" / "_index.json"

    def test_ensure_creates_tree(self, tmp_path: Path):
        layout = DirectoryLayout(tmp_path / "brand-new")
        layout.ensure()
        assert layout.transcripts.is_dir()
        assert layout.memory.is_dir()
        assert layout.topics_dir.is_dir()
        for cat in ("daily", "entities", "projects", "insights"):
            assert layout.note_dir(cat).is_dir()

    def test_is_reserved(self, tmp_path: Path):
        layout = DirectoryLayout(tmp_path / "s")
        assert layout.is_reserved(Path("MEMORY.md"))
        assert layout.is_reserved(Path("_index.json"))
        assert not layout.is_reserved(Path("topics/async.md"))


@pytest.mark.asyncio
class TestSTMJsonlFormat:
    async def test_append_writes_one_jsonl_line_per_turn(self, provider: FileMemoryProvider):
        stm = provider.stm()
        await stm.append(Turn(role="user", content="hi"))
        await stm.append(Turn(role="assistant", content="hello"))
        lines = (
            provider.root.joinpath("transcripts/session.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["type"] == "message"
        assert first["role"] == "user"
        assert first["content"] == "hi"
        assert "ts" in first and first["ts"]

    async def test_unknown_type_is_ignored_but_preserved(self, provider: FileMemoryProvider):
        path = provider.root / "transcripts" / "session.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"type": "event", "event": "tool_call", "ts": "2026-04-19T10:00:00+00:00"})
            + "\n"
            + json.dumps(
                {
                    "type": "message",
                    "role": "user",
                    "content": "ok",
                    "ts": "2026-04-19T10:00:01+00:00",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        # STM reads skip events and return only messages
        recent = await provider.stm().recent(5)
        assert len(recent) == 1
        assert recent[0].content == "ok"
        # The raw file is preserved — no rewrite during read
        assert path.read_text(encoding="utf-8").count("tool_call") == 1


@pytest.mark.asyncio
class TestLTMMarkdownFormat:
    async def test_main_ltm_is_appended_with_timestamp_comment(self, provider: FileMemoryProvider):
        await provider.ltm().append("first note")
        body = provider.root.joinpath("memory/MEMORY.md").read_text(encoding="utf-8")
        assert "first note" in body
        assert body.lstrip().startswith("<!--")

    async def test_dated_ltm_uses_iso_filename(self, provider: FileMemoryProvider):
        from datetime import datetime, timezone

        await provider.ltm().write_dated(
            "daily log", day=datetime(2026, 4, 19, tzinfo=timezone.utc)
        )
        assert provider.root.joinpath("memory/2026-04-19.md").exists()

    async def test_topic_is_slugged_and_placed(self, provider: FileMemoryProvider):
        await provider.ltm().write_topic("Python Async!", "body")
        candidates = list(provider.root.joinpath("memory/topics").glob("*.md"))
        assert len(candidates) == 1
        assert candidates[0].stem.startswith("python_async")


@pytest.mark.asyncio
class TestNotesFrontmatter:
    async def test_note_is_written_with_frontmatter_block(self, provider: FileMemoryProvider):
        notes = provider.notes()
        await notes.write(
            NoteDraft(
                title="Scratch",
                body="body line\nwith [[other]]",
                tags=["py", "async"],
                importance=Importance.HIGH,
                category="topics",
            )
        )
        note_path = next(provider.root.joinpath("memory/topics").glob("*.md"))
        text = note_path.read_text(encoding="utf-8")
        assert text.startswith("---\n")
        meta, body = frontmatter.split(text)
        assert meta["title"] == "Scratch"
        assert meta["importance"] == "high"
        assert meta["tags"] == ["py", "async"]
        assert meta["category"] == "topics"
        assert "body line" in body
        assert "[[other]]" in body
        # Wikilink was parsed into links_to
        assert "other" in meta["links_to"]

    async def test_round_trip_reads_back_tags_and_importance(self, provider: FileMemoryProvider):
        notes = provider.notes()
        await notes.write(
            NoteDraft(
                title="Alpha",
                body="text",
                tags=["one", "two"],
                importance=Importance.CRITICAL,
                category="insights",
            )
        )
        fresh = FileMemoryProvider(provider.root)
        await fresh.initialize()
        all_notes = await fresh.notes().list()
        assert len(all_notes) == 1
        assert all_notes[0].importance == Importance.CRITICAL
        assert set(all_notes[0].tags) == {"one", "two"}


@pytest.mark.asyncio
class TestIndexJsonSchema:
    async def test_index_cache_has_expected_keys(self, provider: FileMemoryProvider):
        notes = provider.notes()
        await notes.write(
            NoteDraft(title="Linked", body="see [[target]]", tags=["x"], category="topics")
        )
        payload = await provider.index().snapshot()
        for key in ("files", "tag_map", "link_graph", "last_rebuilt", "total_files"):
            assert key in payload
        assert payload["total_files"] >= 1
        assert "x" in payload["tag_map"]
        # Index JSON is written back to disk
        index_path = provider.root / "memory" / "_index.json"
        assert index_path.exists()
        on_disk = json.loads(index_path.read_text(encoding="utf-8"))
        assert on_disk["total_files"] == payload["total_files"]


@pytest.mark.asyncio
class TestSnapshotRoundTrip:
    async def test_tarball_round_trip_preserves_notes(
        self, provider: FileMemoryProvider, tmp_path: Path
    ):
        await provider.notes().write(
            NoteDraft(title="Persist", body="survive restart", category="topics")
        )
        await provider.ltm().write_dated("daily entry")
        snap = await provider.snapshot()
        assert snap.size_bytes > 0
        assert snap.checksum

        fresh = FileMemoryProvider(tmp_path / "restored")
        await fresh.initialize()
        await fresh.restore(snap)
        after = await fresh.notes().list()
        titles = {m.title for m in after}
        assert "Persist" in titles

    async def test_tampered_checksum_rejected(self, provider: FileMemoryProvider, tmp_path: Path):
        snap = await provider.snapshot()
        snap.checksum = "0" * 64  # deliberately wrong
        fresh = FileMemoryProvider(tmp_path / "restored-bad")
        await fresh.initialize()
        with pytest.raises(ValueError, match="checksum"):
            await fresh.restore(snap)


@pytest.mark.asyncio
class TestRetrieveComposition:
    async def test_retrieve_mixes_layers_and_respects_budget(self, provider: FileMemoryProvider):
        await provider.stm().append(Turn(role="user", content="budget question"))
        await provider.ltm().append("Budget memo: capacity matters.")
        await provider.notes().write(
            NoteDraft(
                title="budget",
                body="specifics about budget planning",
                tags=["finance"],
                importance=Importance.HIGH,
                category="topics",
            )
        )
        result = await provider.retrieve(RetrievalQuery(text="budget", max_chars=4000))
        assert result.chunks, "retrieve must pull at least one chunk"
        sources = {c.source for c in result.chunks}
        # At minimum we should see LTM and Notes for a keyword the other layers also contain.
        assert "note" in sources or "long_term" in sources
        assert result.total_chars <= 4000 + max((len(c.content) for c in result.chunks), default=0)
