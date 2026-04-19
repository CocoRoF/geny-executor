"""Vector-layer tests for SQLMemoryProvider.

Mirrors `test_memory_provider_file_vector.py` against the SQL backend.
The point is to confirm that the same `EmbeddingClient` Protocol gives
identical surface behaviour regardless of the underlying store.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from geny_executor.memory.embedding import LocalHashEmbeddingClient
from geny_executor.memory.provider import (
    Capability,
    ExecutionSummary,
    Layer,
    NoteDraft,
    NoteRef,
    RetrievalQuery,
    Scope,
)
from geny_executor.memory.providers import SQLMemoryProvider


@pytest.fixture
async def provider(tmp_path: Path) -> SQLMemoryProvider:
    client = LocalHashEmbeddingClient(model="test", dimension=64)
    p = SQLMemoryProvider(tmp_path / "session.db", embedding_client=client)
    await p.initialize()
    return p


@pytest.fixture
async def provider_no_vector(tmp_path: Path) -> SQLMemoryProvider:
    p = SQLMemoryProvider(tmp_path / "session.db")
    await p.initialize()
    return p


@pytest.mark.asyncio
class TestVectorWiring:
    async def test_vector_handle_present_when_client_supplied(self, provider):
        assert provider.vector() is not None

    async def test_vector_handle_absent_without_client(self, provider_no_vector):
        assert provider_no_vector.vector() is None

    async def test_descriptor_surfaces_vector_layer(self, provider):
        d = provider.descriptor
        assert Layer.VECTOR in d.layers
        assert Capability.REINDEX in d.capabilities


@pytest.mark.asyncio
class TestIndexAndSearch:
    async def test_index_and_search_returns_inserted_row(self, provider, tmp_path):
        ref = NoteRef(filename="test.md", scope=Scope.SESSION, backend="sqlite")
        added = await provider.vector().index(ref, "the quick brown fox jumps")
        assert added == 1
        # Same filename re-indexed → row replaced, no new row added
        again = await provider.vector().index(ref, "the quick brown fox jumps over the lazy dog")
        assert again == 0

        # Direct table inspection — only one row for that filename
        with sqlite3.connect(tmp_path / "session.db") as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM vector_rows WHERE filename = ?", (ref.filename,)
            ).fetchall()
        assert len(rows) == 1
        assert rows[0]["dimension"] == 64

        results = await provider.vector().search("the quick brown fox", top_k=3)
        assert results
        assert results[0].key == "test.md"
        assert results[0].source == "vector"

    async def test_dimension_mismatch_raises(self, tmp_path):
        client = LocalHashEmbeddingClient(model="test", dimension=64)
        provider = SQLMemoryProvider(tmp_path / "session.db", embedding_client=client)
        await provider.initialize()
        # Force a vector with the wrong dimension
        original_embed = client.embed

        async def _bad_embed(texts):  # type: ignore[no-untyped-def]
            return [[0.0] * 32 for _ in texts]

        client.embed = _bad_embed  # type: ignore[assignment]
        with pytest.raises(ValueError, match="dimension mismatch"):
            await provider.vector().index(
                NoteRef(filename="x.md", scope=Scope.SESSION, backend="sqlite"),
                "body",
            )
        client.embed = original_embed  # type: ignore[assignment]


@pytest.mark.asyncio
class TestRecordExecutionAutoIndexes:
    async def test_record_execution_indexes_into_vector_table(self, provider, tmp_path):
        await provider.record_execution(
            ExecutionSummary(
                session_id="s1",
                user_input="what is the meaning of fox?",
                final_text="A fox is a small wild dog-like mammal.",
                tags=["fox"],
            )
        )
        with sqlite3.connect(tmp_path / "session.db") as conn:
            n = conn.execute("SELECT COUNT(*) FROM vector_rows").fetchone()[0]
        assert n >= 1


@pytest.mark.asyncio
class TestRetrieveVectorBranch:
    async def test_retrieve_includes_vector_chunks(self, provider):
        await provider.notes().write(NoteDraft(title="alpha", body="the quick brown fox"))
        await provider.vector().index(
            NoteRef(filename="alpha.md", scope=Scope.SESSION, backend="sqlite"),
            "the quick brown fox",
        )
        result = await provider.retrieve(
            RetrievalQuery(
                text="fox",
                layers={Layer.STM, Layer.LTM, Layer.NOTES, Layer.VECTOR},
                max_chars=4000,
            )
        )
        # Vector-source chunk must appear in the breakdown
        assert result.layer_breakdown.get(Layer.VECTOR, 0) >= 1


@pytest.mark.asyncio
class TestReindex:
    async def test_reindex_returns_plan_with_vector_layer(self, provider):
        # Write two notes, index them, then reindex
        for i in range(2):
            ref = NoteRef(filename=f"n{i}.md", scope=Scope.SESSION, backend="sqlite")
            await provider.notes().write(
                NoteDraft(title=f"n{i}", body=f"body {i}", filename=f"n{i}.md")
            )
            await provider.vector().index(ref, f"body {i}")
        plan = await provider.vector().reindex()
        assert plan.layer == Layer.VECTOR
        assert plan.chunks_to_reindex == 2


@pytest.mark.asyncio
class TestSnapshotPreservesVectors:
    async def test_round_trip_carries_vector_rows(self, provider, tmp_path):
        ref = NoteRef(filename="alpha.md", scope=Scope.SESSION, backend="sqlite")
        await provider.notes().write(NoteDraft(title="alpha", body="the fox", filename="alpha.md"))
        await provider.vector().index(ref, "the fox")
        snap = await provider.snapshot()

        client = LocalHashEmbeddingClient(model="test", dimension=64)
        fresh_path = tmp_path / "restored.db"
        fresh = SQLMemoryProvider(fresh_path, embedding_client=client)
        await fresh.initialize()
        await fresh.restore(snap)

        with sqlite3.connect(fresh_path) as conn:
            n = conn.execute("SELECT COUNT(*) FROM vector_rows").fetchone()[0]
        assert n == 1
        results = await fresh.vector().search("fox", top_k=3)
        assert results and results[0].key == "alpha.md"
