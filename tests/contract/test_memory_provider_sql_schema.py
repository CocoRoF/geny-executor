"""Schema-lock suite for SQLMemoryProvider.

Mirrors what `test_memory_provider_file_layout.py` does for the file
provider: nails down the *format* (table set, column set, link origin
discriminator, snapshot payload shape) so a refactor that drifts the
schema fails here loudly instead of silently in some downstream
adapter.

These tests run against a real SQLite database in a tmp_path; they
do not stub out the connection.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from geny_executor.memory.provider import (
    Importance,
    Layer,
    NoteDraft,
    RetrievalQuery,
    Scope,
    Turn,
)
from geny_executor.memory.providers import SQLMemoryProvider
from geny_executor.memory.providers.sql.schema import SQLITE_TABLES


@pytest.fixture
async def provider(tmp_path: Path) -> SQLMemoryProvider:
    p = SQLMemoryProvider(tmp_path / "session.db")
    await p.initialize()
    return p


@pytest.mark.asyncio
class TestSchemaTables:
    async def test_owned_tables_exist_after_initialize(self, provider, tmp_path):
        # Open a sibling sync connection to read sqlite_master directly.
        path = tmp_path / "session.db"
        with sqlite3.connect(path) as conn:
            cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            present = {row[0] for row in cur.fetchall()}
        for table in SQLITE_TABLES:
            assert table in present, f"missing table {table!r} after initialize"

    async def test_notes_columns_match_contract(self, provider, tmp_path):
        path = tmp_path / "session.db"
        with sqlite3.connect(path) as conn:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(notes)")}
        expected = {
            "filename",
            "title",
            "body",
            "importance",
            "category",
            "scope",
            "backend",
            "frontmatter_json",
            "created_at",
            "updated_at",
        }
        assert expected <= cols, f"notes table missing columns: {expected - cols}"

    async def test_vector_rows_columns_include_blob(self, provider, tmp_path):
        path = tmp_path / "session.db"
        with sqlite3.connect(path) as conn:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(vector_rows)")}
        assert "vector_blob" in cols
        assert "dimension" in cols
        assert "filename" in cols


@pytest.mark.asyncio
class TestSTMRows:
    async def test_append_writes_jsonl_compatible_record(self, provider, tmp_path):
        await provider.stm().append(Turn(role="user", content="hello"))
        # Open sibling read-only handle to see the row directly
        with sqlite3.connect(tmp_path / "session.db") as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM stm_turns").fetchone()
        assert row["role"] == "user"
        assert row["content"] == "hello"
        assert row["content_kind"] == "string"
        assert row["type"] == "message"
        # ts is ISO-8601
        assert "T" in row["ts"]

    async def test_append_handles_structured_content(self, provider, tmp_path):
        await provider.stm().append(Turn(role="user", content=[{"type": "text", "text": "hi"}]))
        with sqlite3.connect(tmp_path / "session.db") as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM stm_turns").fetchone()
        assert row["content_kind"] == "json"
        decoded = json.loads(row["content"])
        assert decoded == [{"type": "text", "text": "hi"}]


@pytest.mark.asyncio
class TestNoteLinks:
    async def test_wikilink_in_body_writes_wikilink_origin(self, provider, tmp_path):
        await provider.notes().write(NoteDraft(title="src", body="see [[target.md]]", tags=["x"]))
        with sqlite3.connect(tmp_path / "session.db") as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM note_links").fetchall()
        assert any(r["origin"] == "wikilink" and r["target"] == "target.md" for r in rows)

    async def test_explicit_link_writes_explicit_origin(self, provider, tmp_path):
        a = await provider.notes().write(NoteDraft(title="a", body="x"))
        b = await provider.notes().write(NoteDraft(title="b", body="y"))
        await provider.notes().link(a.ref.filename, b.ref.filename)
        with sqlite3.connect(tmp_path / "session.db") as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM note_links WHERE origin = 'explicit'").fetchall()
        assert any(r["source"] == a.ref.filename and r["target"] == b.ref.filename for r in rows)


@pytest.mark.asyncio
class TestNoteTags:
    async def test_tags_normalised_into_note_tags_table(self, provider, tmp_path):
        await provider.notes().write(NoteDraft(title="x", body="y", tags=["red", "blue", "red"]))
        with sqlite3.connect(tmp_path / "session.db") as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT tag FROM note_tags").fetchall()
        tags = sorted(r["tag"] for r in rows)
        assert tags == ["blue", "red"]


@pytest.mark.asyncio
class TestSnapshotRoundTrip:
    async def test_snapshot_payload_is_json_with_tables_key(self, provider):
        await provider.notes().write(NoteDraft(title="x", body="y"))
        snap = await provider.snapshot()
        document = json.loads(snap.payload.decode("utf-8"))
        assert "tables" in document
        assert "format" in document
        for table in SQLITE_TABLES:
            assert table in document["tables"]

    async def test_tampered_checksum_raises(self, provider):
        await provider.notes().write(NoteDraft(title="x", body="y"))
        snap = await provider.snapshot()

        # Build a fresh provider with an independent DSN
        from pathlib import Path

        original = Path(provider.dsn)
        fresh_path = original.with_name(original.stem + "-tampered.db")
        fresh = SQLMemoryProvider(fresh_path)
        await fresh.initialize()
        snap.checksum = "deadbeef" * 8  # wrong but well-formed
        with pytest.raises(ValueError, match="checksum"):
            await fresh.restore(snap)


@pytest.mark.asyncio
class TestRetrieveComposition:
    async def test_retrieve_includes_notes_and_ltm(self, provider):
        await provider.notes().write(
            NoteDraft(
                title="alpha", body="quick brown fox", tags=["fox"], importance=Importance.HIGH
            )
        )
        await provider.ltm().append("the fox is quick")
        result = await provider.retrieve(
            RetrievalQuery(
                text="fox",
                layers={Layer.NOTES, Layer.LTM},
                max_chars=4000,
            )
        )
        assert result.chunks
        sources = {c.source for c in result.chunks}
        # Both planes should contribute at least one chunk
        assert any(
            s in {"note", "long_term", "long_term_dated", "long_term_topic"} for s in sources
        )


@pytest.mark.asyncio
class TestPromoteUpdatesScope:
    async def test_promote_persists_new_scope(self, provider, tmp_path):
        meta = await provider.notes().write(NoteDraft(title="prom", body="x", scope=Scope.SESSION))
        new_ref = await provider.promote(meta.ref, Scope.USER)
        assert new_ref.scope == Scope.USER
        with sqlite3.connect(tmp_path / "session.db") as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT scope FROM notes WHERE filename = ?", (meta.ref.filename,)
            ).fetchone()
        assert row["scope"] == Scope.USER.value
