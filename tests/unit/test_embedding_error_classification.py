"""EmbeddingError category classification (audit §2.6).

The embedding boundary previously raised a single generic exception,
which made retry/breaker policy impossible — the live 401-spam
incident retried a dead key on every note write because nothing could
distinguish "key revoked" from "network blip". These tests pin:

  - `EmbeddingError.category` defaults to 'unknown' and rejects
    out-of-vocabulary values (typos degrade safely instead of
    becoming a fifth category nobody handles);
  - the openai backend maps typed SDK exceptions (not message text);
  - the voyage backend maps HTTP status codes and transport errors.
"""

from __future__ import annotations

import pytest

from geny_executor.memory.embedding.client import (
    EMBEDDING_ERROR_CATEGORIES,
    EmbeddingError,
)
from geny_executor.memory.embedding.openai import (
    OpenAIEmbeddingClient,
    _classify_openai_error,
)
from geny_executor.memory.embedding.voyage import (
    VoyageEmbeddingClient,
    _category_for_status,
)


# ── EmbeddingError itself ───────────────────────────────────────────


def test_default_category_is_unknown() -> None:
    assert EmbeddingError("boom").category == "unknown"


def test_explicit_categories_round_trip() -> None:
    for category in EMBEDDING_ERROR_CATEGORIES:
        assert EmbeddingError("boom", category=category).category == category


def test_invalid_category_normalizes_to_unknown() -> None:
    assert EmbeddingError("boom", category="banana").category == "unknown"


def test_back_compat_cost_kwarg_still_works() -> None:
    err = EmbeddingError("boom", cost=None)
    assert err.cost is None
    assert err.category == "unknown"


# ── openai: typed SDK exception mapping ─────────────────────────────


def _sdk_exc(name: str) -> Exception:
    """Instantiate an openai SDK exception type without running its
    __init__ (the real constructors demand httpx Response plumbing
    that adds nothing to an isinstance-based classifier test)."""
    openai = pytest.importorskip("openai")
    cls = getattr(openai, name)
    return cls.__new__(cls)


@pytest.mark.parametrize(
    ("exc_name", "expected"),
    [
        ("AuthenticationError", "auth"),
        ("PermissionDeniedError", "auth"),
        ("RateLimitError", "quota"),
        ("APIConnectionError", "transient"),
        ("APITimeoutError", "transient"),
        ("InternalServerError", "transient"),
        ("BadRequestError", "unknown"),
    ],
)
def test_classify_openai_error(exc_name: str, expected: str) -> None:
    assert _classify_openai_error(_sdk_exc(exc_name)) == expected


def test_classify_openai_error_generic_exception_is_unknown() -> None:
    assert _classify_openai_error(ValueError("nope")) == "unknown"


async def test_openai_client_attaches_category_on_failure() -> None:
    """End-to-end through the client: a typed SDK failure surfaces as
    a classified EmbeddingError."""

    class _FailingEmbeddings:
        async def create(self, **_kwargs):
            raise _sdk_exc("AuthenticationError")

    class _FakeSDKClient:
        embeddings = _FailingEmbeddings()

    client = OpenAIEmbeddingClient(api_key="k", client=_FakeSDKClient())
    with pytest.raises(EmbeddingError) as excinfo:
        await client.embed(["hello"])
    assert excinfo.value.category == "auth"


async def test_openai_client_unclassified_failure_is_unknown() -> None:
    class _FailingEmbeddings:
        async def create(self, **_kwargs):
            raise RuntimeError("something else entirely")

    class _FakeSDKClient:
        embeddings = _FailingEmbeddings()

    client = OpenAIEmbeddingClient(api_key="k", client=_FakeSDKClient())
    with pytest.raises(EmbeddingError) as excinfo:
        await client.embed(["hello"])
    assert excinfo.value.category == "unknown"


# ── voyage: HTTP status mapping ─────────────────────────────────────


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, "auth"),
        (403, "auth"),
        (429, "quota"),
        (408, "transient"),
        (500, "transient"),
        (503, "transient"),
        (400, "unknown"),
        (404, "unknown"),
    ],
)
def test_voyage_status_classification(status: int, expected: str) -> None:
    assert _category_for_status(status) == expected


async def test_voyage_transport_stub_can_raise_classified_error() -> None:
    """The injectable transport hook propagates classified errors
    untouched — what the vector layer's breaker will consume."""

    async def transport(_url, _headers, _body):
        raise EmbeddingError("voyage embed HTTP 401: nope", category="auth")

    client = VoyageEmbeddingClient(api_key="k", transport=transport)
    with pytest.raises(EmbeddingError) as excinfo:
        await client.embed(["hello"])
    assert excinfo.value.category == "auth"


# ── input token-budget guard (crash-safety net) ──────────────────────

import asyncio  # noqa: E402

from geny_executor.memory.embedding import openai as _openai_mod  # noqa: E402


def test_bound_input_passes_short_text():
    from geny_executor.memory.embedding.openai import _bound_input, _MAX_EMBED_BYTES

    t = "짧은 한글 노트 " * 10
    assert len(t.encode("utf-8")) <= _MAX_EMBED_BYTES
    assert _bound_input(t) == t  # untouched


def test_bound_input_truncates_over_budget_on_utf8_boundary():
    from geny_executor.memory.embedding.openai import (
        _bound_input, _MAX_EMBED_BYTES, _TRUNCATE_TO_BYTES,
    )

    # A CJK note far over the byte budget (each char is 3 UTF-8 bytes).
    huge = "가" * 10000  # 30000 bytes
    assert len(huge.encode("utf-8")) > _MAX_EMBED_BYTES
    out = _bound_input(huge)
    encoded = out.encode("utf-8")
    assert len(encoded) <= _TRUNCATE_TO_BYTES  # bounded → tokens ≤ bytes ≤ budget
    assert "�" not in out  # clean UTF-8 boundary, no mojibake


def test_embed_never_sends_over_budget_input():
    """The whole-note memory path embeds un-chunked text; the client must
    bound it so OpenAI never 400s ('maximum input length is 8192 tokens')."""
    seen = {}

    class _FakeEmbeddings:
        async def create(self, *, input, model):
            seen["input"] = input

            class _R:
                data = [type("E", (), {"embedding": [0.0, 0.1]})() for _ in input]

            return _R()

    class _FakeClient:
        embeddings = _FakeEmbeddings()

    client = OpenAIEmbeddingClient(model="text-embedding-3-large", api_key="k", client=_FakeClient())
    asyncio.run(client.embed(["나" * 20000]))
    assert len(seen["input"][0].encode("utf-8")) <= 8192
