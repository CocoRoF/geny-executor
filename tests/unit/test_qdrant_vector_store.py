"""QdrantVectorStore (2.47.0) — knowledge-vault vector backend.

Contract under test with a fake qdrant client: chunked document indexing
(one point per chunk, payload carries page/source metadata), idempotent
deterministic ids, document replacement, search → MemoryChunk shape, and
dimension-mismatch bootstrap protection. Also: the retriever's curated
layer consumes the vector plane (hybrid) and the file provider accepts an
injected vector store.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from geny_executor.memory.provider import EmbeddingDescriptor, NoteRef, Scope


# ── fake qdrant SDK (installed into sys.modules before import) ───────


@dataclass
class _FakePoint:
    id: str
    vector: List[float]
    payload: Dict[str, Any]
    score: float = 0.9


class _FakeModels(types.ModuleType):
    def __init__(self):
        super().__init__("qdrant_client.models")

        class VectorParams:
            def __init__(self, size, distance):
                self.size = size
                self.distance = distance

        class Distance:
            COSINE = "cosine"

        class MatchValue:
            def __init__(self, value):
                self.value = value

        class FieldCondition:
            def __init__(self, key, match):
                self.key = key
                self.match = match

        class Filter:
            def __init__(self, must):
                self.must = must

        class PointStruct:
            def __init__(self, id, vector, payload):
                self.id = id
                self.vector = vector
                self.payload = payload

        self.VectorParams = VectorParams
        self.Distance = Distance
        self.MatchValue = MatchValue
        self.FieldCondition = FieldCondition
        self.Filter = Filter
        self.PointStruct = PointStruct


class _FakeAsyncQdrant:
    instances: List["_FakeAsyncQdrant"] = []

    def __init__(self, *, url, api_key=None, timeout=None):
        self.url = url
        self.points: Dict[str, _FakePoint] = {}
        self.collection_dim: Optional[int] = None
        _FakeAsyncQdrant.instances.append(self)

    async def collection_exists(self, name):
        return self.collection_dim is not None

    async def get_collection(self, name):
        dim = self.collection_dim

        class _Info:
            class config:
                class params:
                    class vectors:
                        size = dim

        return _Info()

    async def create_collection(self, collection_name, vectors_config):
        self.collection_dim = vectors_config.size

    async def delete(self, collection_name, points_selector):
        target = points_selector.must[0].match.value
        self.points = {
            pid: p for pid, p in self.points.items()
            if p.payload.get("filename") != target
        }

    async def upsert(self, collection_name, points):
        for p in points:
            self.points[p.id] = _FakePoint(p.id, p.vector, p.payload)

    async def query_points(self, collection_name, query, limit, score_threshold,
                           with_payload):
        pts = list(self.points.values())[:limit]

        class _R:
            points = pts

        return _R()

    async def count(self, collection_name):
        n = len(self.points)

        class _C:
            count = n

        return _C()

    async def close(self):
        pass


@pytest.fixture()
def qdrant_store(monkeypatch):
    _FakeAsyncQdrant.instances.clear()
    pkg = types.ModuleType("qdrant_client")
    pkg.AsyncQdrantClient = _FakeAsyncQdrant
    pkg.models = _FakeModels()
    monkeypatch.setitem(sys.modules, "qdrant_client", pkg)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", pkg.models)

    class _Embed:
        descriptor = EmbeddingDescriptor(
            provider="openai", model="text-embedding-3-large", dimension=4,
        )

        async def embed(self, texts):
            return [[float(len(t) % 7), 1.0, 0.0, 0.5] for t in texts]

    from geny_executor.memory.vector import DocumentChunk, QdrantVectorStore

    store = QdrantVectorStore(
        url="http://fake:6333", collection="kb_test", client=_Embed(),
    )
    return store, DocumentChunk


@pytest.mark.asyncio
async def test_index_document_one_point_per_chunk_with_payload(qdrant_store):
    store, DocumentChunk = qdrant_store
    ref = NoteRef(filename="doc-abc.md", scope=Scope.USER, category="knowledge")
    n = await store.index_document(ref, [
        DocumentChunk(text="1장 내용", metadata={"page": 1, "source_url": "u"}),
        DocumentChunk(text="2장 내용", metadata={"page": 2}),
    ])
    assert n == 2
    backend = _FakeAsyncQdrant.instances[0]
    payloads = sorted(
        (p.payload for p in backend.points.values()),
        key=lambda p: p["chunk_index"],
    )
    assert payloads[0]["filename"] == "doc-abc.md"
    assert payloads[0]["page"] == 1 and payloads[0]["source_url"] == "u"
    assert payloads[1]["chunk_count"] == 2
    assert backend.collection_dim == 4  # bootstrapped from descriptor


@pytest.mark.asyncio
async def test_reindex_replaces_document_points(qdrant_store):
    store, DocumentChunk = qdrant_store
    ref = NoteRef(filename="doc.md", scope=Scope.USER)
    await store.index_document(ref, [DocumentChunk(text=f"c{i}") for i in range(3)])
    await store.index_document(ref, [DocumentChunk(text="only one now")])
    backend = _FakeAsyncQdrant.instances[0]
    assert len(backend.points) == 1  # shrink drops stale chunk points


@pytest.mark.asyncio
async def test_note_write_seam_and_search_shape(qdrant_store):
    store, _ = qdrant_store
    ref = NoteRef(filename="note.md", scope=Scope.USER, category="topics")
    assert await store.index(ref, "노트 본문") == 1

    hits = await store.search("질의", top_k=5)
    assert len(hits) == 1
    chunk = hits[0]
    assert chunk.source == "vector"
    assert chunk.key == "vector:note.md#0"
    assert chunk.metadata["category"] == "topics"
    assert chunk.content  # preview text


@pytest.mark.asyncio
async def test_dimension_mismatch_raises(qdrant_store):
    store, DocumentChunk = qdrant_store
    backend_cls = _FakeAsyncQdrant
    # Pre-create the collection with a different dimension.
    ref = NoteRef(filename="x.md", scope=Scope.USER)
    await store.index_document(ref, [DocumentChunk(text="t")])
    backend_cls.instances[0].collection_dim = 1536
    store._collection_ready = False
    with pytest.raises(RuntimeError):
        await store._ensure_collection()


@pytest.mark.asyncio
async def test_remove_deletes_by_ref(qdrant_store):
    store, DocumentChunk = qdrant_store
    ref = NoteRef(filename="gone.md", scope=Scope.USER)
    await store.index_document(ref, [DocumentChunk(text="a"), DocumentChunk(text="b")])
    assert await store.remove(ref)
    assert len(_FakeAsyncQdrant.instances[0].points) == 0


# ── file provider accepts injected store ─────────────────────────────


@pytest.mark.asyncio
async def test_file_provider_uses_injected_vector_store(tmp_path, qdrant_store):
    store, _ = qdrant_store
    from geny_executor.memory.providers.file.provider import FileMemoryProvider

    provider = FileMemoryProvider(tmp_path, vector_store=store)
    assert provider.vector() is store

    # Note writes flow through the injected store via the indexer seam.
    from geny_executor.memory.provider import Importance, NoteDraft

    await provider.notes().write(
        NoteDraft(title="지식 노트", body="qdrant로 색인될 본문",
                  importance=Importance.MEDIUM, category="topics"),
    )
    backend = _FakeAsyncQdrant.instances[0]
    assert any(
        p.payload.get("preview", "").startswith("qdrant로")
        for p in backend.points.values()
    )


# ── retriever hybrid curated layer ───────────────────────────────────


@pytest.mark.asyncio
async def test_retriever_curated_layer_uses_vector_plane(qdrant_store):
    store, DocumentChunk = qdrant_store
    ref = NoteRef(filename="kb-doc.md", scope=Scope.USER, category="knowledge")
    await store.index_document(ref, [DocumentChunk(text="지식 청크", metadata={"page": 3})])

    from geny_executor.memory.provider import MemoryHooks
    from geny_executor.memory.retriever import MemoryAwareRetriever

    class _Notes:
        async def search(self, query, limit=5):
            return []

    class _Curated:
        def notes(self):
            return _Notes()

        def vector(self):
            return store

    class _Provider:
        def curated(self):
            return _Curated()

    hooks = MemoryHooks()
    retriever = MemoryAwareRetriever(_Provider(), hooks=hooks)
    chunks: list = []
    total = await retriever._load_curated(chunks, "질의", 0, 5000, hooks)
    assert total > 0
    assert chunks and chunks[0].metadata["layer"] == "curated"
    assert chunks[0].metadata["page"] == 3
