"""OpenAI-compatible embedding backend + runtime provider registry (2.52.0).

First-class local-embedding support: any self-hosted `/v1/embeddings`
endpoint (vLLM / Ollama / LM Studio / TEI) plugs in via
`provider="openai_compatible"` with a `base_url`, and hosts with their
own embedding services register custom backends by name through
`register_embedding_provider` — both resolving through the ordinary
serializable config path (`MemoryProviderFactory` included).

These tests pin:
  - the wire shape (payload/model/bearer handling, response parsing,
    index-ordered rows, row-count mismatch);
  - base_url normalization (API root vs full endpoint);
  - optional-auth semantics (no Authorization header without a key,
    no env-ladder fallback);
  - dimension self-heal from the first response;
  - HTTP status → EmbeddingError category via the shared classifier;
  - the registry: registration, factory routing (direct + through
    MemoryProviderFactory config), builtin-shadowing rejection,
    replace/unregister semantics.
"""

from __future__ import annotations

import asyncio

import pytest

from geny_executor.memory.embedding.client import (
    EmbeddingError,
    category_for_http_status,
)
from geny_executor.memory.embedding.openai_compatible import (
    OpenAICompatibleEmbeddingClient,
    _normalize_endpoint,
)
from geny_executor.memory.embedding.registry import (
    create_embedding_client,
    register_embedding_provider,
    registered_embedding_providers,
    unregister_embedding_provider,
)


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


# ── base_url normalization ───────────────────────────────────────────


def test_normalize_endpoint_api_root():
    assert _normalize_endpoint("http://h:8000/v1") == "http://h:8000/v1/embeddings"


def test_normalize_endpoint_full_endpoint_unchanged():
    assert _normalize_endpoint("http://h:8000/v1/embeddings") == "http://h:8000/v1/embeddings"


def test_normalize_endpoint_trailing_slash():
    assert _normalize_endpoint("http://h:8000/v1/") == "http://h:8000/v1/embeddings"


def test_empty_base_url_rejected():
    with pytest.raises(ValueError):
        OpenAICompatibleEmbeddingClient(base_url="", model="m")


def test_empty_model_rejected():
    with pytest.raises(ValueError):
        OpenAICompatibleEmbeddingClient(base_url="http://h/v1", model="  ")


# ── wire shape ───────────────────────────────────────────────────────


def _client(captured, response=None, **kwargs):
    async def transport(url, headers, body):
        captured.append((url, headers, body))
        if response is not None:
            return response
        return {
            "data": [
                {"index": i, "embedding": [float(i), 1.0, 2.0]}
                for i in range(len(body["input"]))
            ]
        }

    return OpenAICompatibleEmbeddingClient(
        base_url="http://vllm:8000/v1", model="bge-m3", transport=transport, **kwargs
    )


def test_embed_payload_and_parsing():
    captured = []
    client = _client(captured)
    vectors = _run(client.embed(["hello", "world"]))
    assert len(vectors) == 2
    url, headers, body = captured[0]
    assert url == "http://vllm:8000/v1/embeddings"
    assert body == {"input": ["hello", "world"], "model": "bge-m3"}
    # authless local endpoint → no Authorization header at all
    assert "Authorization" not in headers


def test_bearer_sent_when_key_present():
    captured = []
    client = _client(captured, api_key="tok-123")
    _run(client.embed(["x"]))
    assert captured[0][1]["Authorization"] == "Bearer tok-123"


def test_rows_sorted_by_index():
    captured = []
    client = _client(
        captured,
        response={
            "data": [
                {"index": 1, "embedding": [1.0]},
                {"index": 0, "embedding": [0.0]},
            ]
        },
    )
    vectors = _run(client.embed(["a", "b"]))
    assert vectors == [[0.0], [1.0]]


def test_row_count_mismatch_is_invalid():
    captured = []
    client = _client(captured, response={"data": [{"index": 0, "embedding": [1.0]}]})
    with pytest.raises(EmbeddingError) as exc:
        _run(client.embed(["a", "b"]))
    assert exc.value.category == "invalid"


def test_malformed_response_is_invalid():
    captured = []
    client = _client(captured, response={"nope": True})
    with pytest.raises(EmbeddingError) as exc:
        _run(client.embed(["a"]))
    assert exc.value.category == "invalid"


def test_dimension_self_heals_from_first_response():
    captured = []
    client = _client(captured)
    assert client.descriptor.dimension == 0
    _run(client.embed(["hello"]))
    assert client.descriptor.dimension == 3
    assert client.descriptor.provider == "openai_compatible"
    assert client.descriptor.model == "bge-m3"


def test_explicit_dimension_kept():
    captured = []
    client = _client(captured, dimension=1024)
    assert client.descriptor.dimension == 1024


def test_empty_texts_no_call():
    captured = []
    client = _client(captured)
    assert _run(client.embed([])) == []
    assert captured == []


# ── status classification (shared classifier) ────────────────────────


@pytest.mark.parametrize(
    "status,expected",
    [
        (401, "auth"),
        (403, "auth"),
        (429, "quota"),
        (408, "transient"),
        (500, "transient"),
        (503, "transient"),
        (400, "invalid"),
        (404, "invalid"),
        (422, "invalid"),
        (302, "unknown"),
    ],
)
def test_category_for_http_status(status, expected):
    assert category_for_http_status(status) == expected


def test_voyage_alias_still_matches_shared_classifier():
    from geny_executor.memory.embedding.voyage import _category_for_status

    for status in (401, 429, 500, 404, 302):
        assert _category_for_status(status) == category_for_http_status(status)


# ── factory routing ──────────────────────────────────────────────────


def test_factory_builds_openai_compatible():
    client = create_embedding_client(
        "openai_compatible",
        model="bge-m3",
        dimension=768,
        options={"base_url": "http://vllm:8000/v1"},
    )
    assert isinstance(client, OpenAICompatibleEmbeddingClient)
    assert client.descriptor.dimension == 768


def test_factory_openai_compatible_requires_model():
    with pytest.raises(ValueError):
        create_embedding_client(
            "openai_compatible", options={"base_url": "http://vllm:8000/v1"}
        )


# ── runtime provider registry ────────────────────────────────────────


class _StubClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @property
    def descriptor(self):  # pragma: no cover - shape only
        from geny_executor.memory.provider import EmbeddingDescriptor

        return EmbeddingDescriptor(provider="stub", model="m", dimension=3)

    async def embed(self, texts):
        return [[0.0, 0.0, 0.0] for _ in texts]

    async def close(self):
        return None


@pytest.fixture(autouse=True)
def _clean_registry():
    yield
    unregister_embedding_provider("host-svc")


def test_register_and_create_by_name():
    seen = {}

    def builder(**kwargs):
        seen.update(kwargs)
        return _StubClient(**kwargs)

    register_embedding_provider("host-svc", builder)
    assert "host-svc" in registered_embedding_providers()
    client = create_embedding_client(
        "host-svc", model="m1", api_key="k", dimension=7, options={"endpoint": "http://x"}
    )
    assert isinstance(client, _StubClient)
    # builder receives normalized kwargs + expanded options — the exact
    # surface built-in backends get.
    assert seen == {"model": "m1", "api_key": "k", "dimension": 7, "endpoint": "http://x"}


def test_register_routes_through_memory_provider_factory(tmp_path):
    register_embedding_provider("host-svc", lambda **kw: _StubClient(**kw))
    from geny_executor.memory.factory import MemoryProviderFactory

    provider = MemoryProviderFactory().build(
        {
            "provider": "file",
            "root": str(tmp_path),
            "embedding": {"provider": "host-svc", "model": "m1"},
        }
    )
    try:
        assert provider.vector() is not None  # embedding wired → vector layer alive
    finally:
        _run(provider.close())


def test_builtin_shadowing_rejected():
    for builtin in ("openai", "voyage", "google", "local", "openai_compatible"):
        with pytest.raises(ValueError):
            register_embedding_provider(builtin, lambda **kw: _StubClient(**kw))


def test_double_registration_needs_replace():
    register_embedding_provider("host-svc", lambda **kw: _StubClient(**kw))
    with pytest.raises(ValueError):
        register_embedding_provider("host-svc", lambda **kw: _StubClient(**kw))
    # replace=True is the deliberate idempotent-boot path
    register_embedding_provider("host-svc", lambda **kw: _StubClient(**kw), replace=True)


def test_unregister():
    register_embedding_provider("host-svc", lambda **kw: _StubClient(**kw))
    assert unregister_embedding_provider("host-svc") is True
    assert unregister_embedding_provider("host-svc") is False
    with pytest.raises(ValueError):
        create_embedding_client("host-svc")


def test_unknown_provider_error_mentions_registered():
    register_embedding_provider("host-svc", lambda **kw: _StubClient(**kw))
    with pytest.raises(ValueError) as exc:
        create_embedding_client("nope")
    assert "host-svc" in str(exc.value)
