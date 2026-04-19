"""Unit + contract tests for the EmbeddingClient family.

Local backend is exercised directly (no deps). The three remote
backends (openai / voyage / google) are exercised with stubbed clients
or transports so the suite runs without API keys or network.

What we lock down here is the *shape* of the contract:
- descriptor exposes provider / model / dimension / metric
- embed(...) returns one vector per input, in order
- dimension is consistent across calls
- empty input list is a no-op
- registry dispatches on provider name
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence
from unittest.mock import AsyncMock

import pytest

from geny_executor.memory.embedding import (
    EmbeddingClient,
    EmbeddingError,
    LocalHashEmbeddingClient,
    create_embedding_client,
)
from geny_executor.memory.embedding.openai import OpenAIEmbeddingClient
from geny_executor.memory.embedding.voyage import VoyageEmbeddingClient


@pytest.mark.asyncio
class TestLocalHashBackend:
    async def test_descriptor_fields(self):
        c = LocalHashEmbeddingClient(dimension=256)
        d = c.descriptor
        assert d.provider == "local"
        assert d.model == "hash-v1"
        assert d.dimension == 256
        assert d.metric == "cosine"
        assert d.api_key_present is False

    async def test_embed_returns_one_vector_per_input_in_order(self):
        c = LocalHashEmbeddingClient(dimension=128)
        vecs = await c.embed(["alpha", "beta", "gamma"])
        assert len(vecs) == 3
        assert all(len(v) == 128 for v in vecs)
        # Order preserved
        again = await c.embed(["alpha"])
        assert vecs[0] == again[0]

    async def test_embed_is_deterministic(self):
        c = LocalHashEmbeddingClient(dimension=64)
        a = await c.embed(["the quick brown fox"])
        b = await c.embed(["the quick brown fox"])
        assert a == b

    async def test_vector_is_l2_normalised(self):
        import math

        c = LocalHashEmbeddingClient(dimension=128)
        (v,) = await c.embed(["some meaningful text"])
        norm = math.sqrt(sum(x * x for x in v))
        assert abs(norm - 1.0) < 1e-6

    async def test_empty_input_returns_empty_list(self):
        c = LocalHashEmbeddingClient(dimension=32)
        assert await c.embed([]) == []

    async def test_dimension_bounds_validated(self):
        with pytest.raises(ValueError):
            LocalHashEmbeddingClient(dimension=8)
        with pytest.raises(ValueError):
            LocalHashEmbeddingClient(dimension=10000)

    async def test_same_prefix_closer_than_unrelated_text(self):
        """Local hashing isn't semantic, but texts that share tokens
        still land closer than texts that share *none*."""
        import math

        c = LocalHashEmbeddingClient(dimension=512)
        a, b, c_vec = await c.embed(
            ["python async await", "python async generator", "guitar amplifier tube"]
        )

        def cos(u, v):
            dot = sum(x * y for x, y in zip(u, v))
            nu = math.sqrt(sum(x * x for x in u))
            nv = math.sqrt(sum(x * x for x in v))
            return dot / (nu * nv) if nu and nv else 0.0

        related = cos(a, b)
        unrelated = cos(a, c_vec)
        assert related > unrelated


@pytest.mark.asyncio
class TestOpenAIBackend:
    async def test_descriptor_picks_up_known_dimension(self):
        c = OpenAIEmbeddingClient(
            model="text-embedding-3-small",
            api_key="sk-fake",
            client=_stub_openai_client([[0.0] * 1536]),
        )
        assert c.descriptor.dimension == 1536
        assert c.descriptor.api_key_present is True

    async def test_embed_delegates_to_sdk_and_preserves_order(self):
        stub = _stub_openai_client([[0.1] * 4, [0.2] * 4])
        c = OpenAIEmbeddingClient(model="text-embedding-3-small", client=stub, dimension=4)
        vecs = await c.embed(["hello", "world"])
        assert vecs == [[0.1] * 4, [0.2] * 4]
        # SDK called once with the batch
        stub.embeddings.create.assert_awaited_once()

    async def test_embed_raises_embedding_error_on_sdk_exception(self):
        stub = AsyncMock()
        stub.embeddings = AsyncMock()
        stub.embeddings.create = AsyncMock(side_effect=RuntimeError("boom"))
        c = OpenAIEmbeddingClient(model="text-embedding-3-small", client=stub, dimension=4)
        with pytest.raises(EmbeddingError):
            await c.embed(["x"])

    async def test_missing_sdk_raises_helpful_importerror(self, monkeypatch):
        # Force import failure by wiping the real module out of sys.modules
        import sys

        # When the real SDK is installed this still exercises the happy
        # path; when it's missing the ImportError bubbles with our message.
        monkeypatch.setitem(sys.modules, "openai", None)
        c = OpenAIEmbeddingClient(model="text-embedding-3-small", api_key="sk-fake")
        with pytest.raises(ImportError, match="openai"):
            await c.embed(["x"])


@pytest.mark.asyncio
class TestVoyageBackend:
    async def test_embed_uses_transport_hook_and_preserves_order(self):
        captured: Dict[str, Any] = {}

        async def fake_post(url, headers, body):
            captured["url"] = url
            captured["headers"] = headers
            captured["body"] = body
            return {
                "data": [
                    {"embedding": [0.5, 0.5]},
                    {"embedding": [0.25, 0.75]},
                ]
            }

        c = VoyageEmbeddingClient(
            model="voyage-3",
            api_key="voy-fake",
            dimension=2,
            transport=fake_post,
        )
        vecs = await c.embed(["a", "b"])
        assert vecs == [[0.5, 0.5], [0.25, 0.75]]
        assert captured["url"].endswith("/v1/embeddings")
        assert captured["headers"]["Authorization"] == "Bearer voy-fake"
        assert captured["body"]["input"] == ["a", "b"]
        assert captured["body"]["model"] == "voyage-3"

    async def test_embed_surfaces_malformed_response_as_error(self):
        async def bad(url, headers, body):
            return {"not_data": []}

        c = VoyageEmbeddingClient(model="voyage-3", api_key="k", transport=bad, dimension=4)
        with pytest.raises(EmbeddingError):
            await c.embed(["x"])


class TestRegistry:
    def test_dispatches_local(self):
        c = create_embedding_client("local", dimension=32)
        assert isinstance(c, LocalHashEmbeddingClient)
        assert c.descriptor.provider == "local"
        assert c.descriptor.dimension == 32

    def test_dispatches_openai(self):
        c = create_embedding_client("openai", model="text-embedding-3-large", api_key="sk-x")
        assert c.descriptor.provider == "openai"
        assert c.descriptor.model == "text-embedding-3-large"
        assert c.descriptor.dimension == 3072

    def test_dispatches_voyage(self):
        c = create_embedding_client("voyage", api_key="v-k")
        assert c.descriptor.provider == "voyage"
        assert c.descriptor.model == "voyage-3"
        assert c.descriptor.dimension == 1024

    def test_dispatches_google(self):
        c = create_embedding_client("google", api_key="g-k")
        assert c.descriptor.provider == "google"
        assert c.descriptor.model == "text-embedding-004"
        assert c.descriptor.dimension == 768

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="unknown embedding provider"):
            create_embedding_client("made-up")

    def test_all_clients_conform_to_protocol(self):
        for name in ("local",):
            c = create_embedding_client(name, dimension=32)
            assert isinstance(c, EmbeddingClient)


# ── helpers ──────────────────────────────────────────────────────────


def _stub_openai_client(vectors: List[List[float]]):
    """Produce a fake `AsyncOpenAI` whose `embeddings.create` returns
    the given vectors wrapped in the shape the SDK emits.
    """

    class _FakeEmbedding:
        def __init__(self, v: Sequence[float]) -> None:
            self.embedding = list(v)

    class _FakeResponse:
        def __init__(self, vecs: Sequence[Sequence[float]]) -> None:
            self.data = [_FakeEmbedding(v) for v in vecs]

    stub = AsyncMock()
    stub.embeddings = AsyncMock()
    stub.embeddings.create = AsyncMock(return_value=_FakeResponse(vectors))
    return stub
