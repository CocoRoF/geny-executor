"""FileMemoryProvider × VectorHandle behavioural tests.

When a `FileMemoryProvider` is constructed with an
`embedding_client=`, its `vector()` handle must:

- report the client's `EmbeddingDescriptor`,
- index notes on `record_execution` automatically,
- be searchable alongside Notes / LTM in `retrieve()`,
- reject dimension mismatches,
- replace an existing vector when the same note is re-indexed,
- produce a reindex plan via `VectorHandle.reindex`,
- survive snapshot round-trip with the vector payload intact.

Local (deterministic) embeddings are used so the tests are
reproducible and require no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from geny_executor.memory.embedding import LocalHashEmbeddingClient
from geny_executor.memory.provider import (
    ExecutionSummary,
    Importance,
    Layer,
    NoteDraft,
    NoteRef,
    RetrievalQuery,
    Scope,
)
from geny_executor.memory.providers import FileMemoryProvider


pytestmark = pytest.mark.asyncio


@pytest.fixture
async def provider(tmp_path: Path) -> FileMemoryProvider:
    client = LocalHashEmbeddingClient(dimension=128)
    p = FileMemoryProvider(tmp_path / "s", embedding_client=client)
    await p.initialize()
    return p


class TestVectorHandleWiring:
    async def test_vector_returns_non_none_when_client_configured(
        self, provider: FileMemoryProvider
    ):
        v = provider.vector()
        assert v is not None
        d = v.descriptor
        assert d.provider == "local"
        assert d.dimension == 128

    async def test_vector_none_without_client(self, tmp_path: Path):
        p = FileMemoryProvider(tmp_path / "no-emb")
        await p.initialize()
        assert p.vector() is None

    async def test_descriptor_includes_vector_layer(self, provider: FileMemoryProvider):
        d = provider.descriptor
        assert Layer.VECTOR in d.layers
        vector_backend = next(b for b in d.backends if b.layer == Layer.VECTOR)
        assert vector_backend.metadata["embedding_provider"] == "local"
        assert vector_backend.metadata["dimension"] == 128


class TestVectorIndexAndSearch:
    async def test_index_batch_inserts_rows_in_order(self, provider: FileMemoryProvider):
        v = provider.vector()
        refs = [
            NoteRef(filename="a.md", scope=Scope.SESSION, category="topics"),
            NoteRef(filename="b.md", scope=Scope.SESSION, category="topics"),
        ]
        added = await v.index_batch([(refs[0], "alpha body"), (refs[1], "beta body")])
        assert added == 2

    async def test_search_returns_nearest_chunk(self, provider: FileMemoryProvider):
        v = provider.vector()
        await v.index(NoteRef(filename="match.md", scope=Scope.SESSION), "python async await")
        await v.index(NoteRef(filename="other.md", scope=Scope.SESSION), "guitar amplifier tube")
        hits = await v.search("python async", top_k=1)
        assert len(hits) == 1
        assert hits[0].key == "match.md"
        assert hits[0].source == "vector"

    async def test_reindexing_same_note_replaces_row(self, provider: FileMemoryProvider):
        v = provider.vector()
        ref = NoteRef(filename="r.md", scope=Scope.SESSION)
        await v.index(ref, "original body")
        await v.index(ref, "new body text")
        # Only one row for r.md
        hits = await v.search("body", top_k=5)
        assert sum(1 for h in hits if h.key == "r.md") == 1

    async def test_remove_deletes_row(self, provider: FileMemoryProvider):
        v = provider.vector()
        ref = NoteRef(filename="gone.md", scope=Scope.SESSION)
        await v.index(ref, "some text")
        assert await v.remove(ref) is True
        assert await v.search("some text", top_k=5) == []

    async def test_dimension_mismatch_is_rejected(self, provider: FileMemoryProvider):
        """Embedding a value with the wrong dim must raise — this is
        the invariant C5 relies on."""
        v = provider.vector()
        client = v._client  # type: ignore[attr-defined]
        # Monkey-patch the client to emit wrong-dim vectors
        original = client.embed

        async def bad_embed(texts):  # type: ignore
            return [[0.0] * 5 for _ in texts]

        client.embed = bad_embed  # type: ignore[assignment]
        try:
            with pytest.raises(ValueError, match="dimension"):
                await v.index(NoteRef(filename="x.md", scope=Scope.SESSION), "text")
        finally:
            client.embed = original  # type: ignore[assignment]


class TestRecordExecutionAutoIndexes:
    async def test_record_execution_indexes_the_new_note(self, provider: FileMemoryProvider):
        summary = ExecutionSummary(
            session_id="s1",
            user_input="how do we deploy",
            final_text="Use the green-blue strategy with traffic shifting.",
            tags=["deploy"],
        )
        await provider.record_execution(summary)
        v = provider.vector()
        hits = await v.search("deploy traffic", top_k=3)
        assert hits, "vector index should contain the execution note"


class TestRetrieveMixesVectorLayer:
    async def test_vector_contributes_chunks_when_declared_in_query(
        self, provider: FileMemoryProvider
    ):
        await provider.notes().write(
            NoteDraft(
                title="deploy strategy",
                body="green-blue deploy with traffic shifting and canary rollout",
                tags=["deploy"],
                importance=Importance.HIGH,
                category="topics",
            )
        )
        # Ensure the vector also has the row (notes.write doesn't auto-index;
        # record_execution does. Here we index manually via the handle.)
        v = provider.vector()
        await v.index_batch(
            [
                (
                    NoteRef(filename="deploy-strategy.md", scope=Scope.SESSION),
                    "green-blue deploy with traffic shifting and canary rollout",
                )
            ]
        )

        q = RetrievalQuery(
            text="canary deploy",
            layers={Layer.STM, Layer.LTM, Layer.NOTES, Layer.VECTOR},
            max_chars=4000,
        )
        result = await provider.retrieve(q)
        sources = {c.source for c in result.chunks}
        assert "vector" in sources


class TestReindex:
    async def test_reindex_rebuilds_rows_from_source(self, provider: FileMemoryProvider):
        # Write two notes, then index them, then reindex.
        await provider.notes().write(
            NoteDraft(title="alpha", body="the first body text", category="topics", tags=["x"])
        )
        await provider.notes().write(
            NoteDraft(title="beta", body="another body of prose", category="topics", tags=["y"])
        )
        v = provider.vector()
        await v.index_batch(
            [
                (NoteRef(filename="alpha.md", scope=Scope.SESSION), "the first body text"),
                (
                    NoteRef(filename="beta.md", scope=Scope.SESSION),
                    "another body of prose",
                ),
            ]
        )
        before = await v.search("body", top_k=5)
        plan = await v.reindex()
        after = await v.search("body", top_k=5)
        assert {h.key for h in after} == {h.key for h in before}
        assert plan.chunks_to_reindex == 2
        assert plan.layer == Layer.VECTOR


class TestSnapshotPreservesVectors:
    async def test_round_trip_keeps_vector_rows(self, provider: FileMemoryProvider, tmp_path: Path):
        v = provider.vector()
        await v.index(NoteRef(filename="persist.md", scope=Scope.SESSION), "unique query phrase A")
        snap = await provider.snapshot()
        assert Layer.VECTOR in snap.layers

        fresh = FileMemoryProvider(
            tmp_path / "fresh", embedding_client=LocalHashEmbeddingClient(dimension=128)
        )
        await fresh.initialize()
        await fresh.restore(snap)
        hits = await fresh.vector().search("unique query phrase A", top_k=1)
        assert hits
        assert hits[0].key == "persist.md"
