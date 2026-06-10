"""OpenAI boundary hardening (2.2.0 — audit §2.5 + Tier 2).

Two prod-grade fixes under test:

  * The $0 bug — ``create_message_stream`` never sent
    ``stream_options={"include_usage": True}``, so the Chat Completions
    stream sent no usage chunk, the harvesting branch never fired, and
    every streamed call priced at $0 (CostBudgetGuard and both hosts'
    cost displays neutralized for months). The fix is one line; these
    tests exist so it can never silently regress.

  * ``max_tokens → max_completion_tokens`` — OpenAI's reasoning families
    reject the classic kwarg. Proactive (static prefix table) + reactive
    (heal-once on the 400 that names the rename), mirroring the
    Anthropic 2.1.2/2.1.3 pattern through the BaseClient hook.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

openai_sdk = pytest.importorskip("openai")

from geny_executor.core.config import ModelConfig  # noqa: E402
from geny_executor.core.errors import ErrorCategory  # noqa: E402
from geny_executor.llm_client.openai import (  # noqa: E402
    OpenAIClient,
    _model_requires_max_completion_tokens,
)
from geny_executor.llm_client.types import APIRequest  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _req(**overrides: Any) -> APIRequest:
    base: Dict[str, Any] = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1024,
    }
    base.update(overrides)
    return APIRequest(**base)


def _stream_chunks(*, usage_details: Any) -> List[SimpleNamespace]:
    return [
        SimpleNamespace(
            model="gpt-mock",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="hello", tool_calls=None),
                    finish_reason=None,
                )
            ],
        ),
        SimpleNamespace(
            model="gpt-mock",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, tool_calls=None),
                    finish_reason="stop",
                )
            ],
        ),
        SimpleNamespace(
            model="gpt-mock",
            choices=[],
            usage=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=5,
                prompt_tokens_details=usage_details,
            ),
        ),
    ]


def _client_with_stream(
    chunks: List[SimpleNamespace], captured: Dict[str, Any]
) -> OpenAIClient:
    async def create(**kwargs: Any):
        captured.clear()
        captured.update(kwargs)

        async def _gen():
            for chunk in chunks:
                yield chunk

        return _gen()

    client = OpenAIClient(api_key="sk-mock")
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return client


async def _drain(client: OpenAIClient, **mc_overrides: Any):
    events = []
    async for evt in client.create_message_stream(
        model_config=ModelConfig(model="gpt-4o", max_tokens=64, **mc_overrides),
        messages=[{"role": "user", "content": "x"}],
    ):
        events.append(evt)
    return events


# ---------------------------------------------------------------------------
# §2.5 — streaming usage
# ---------------------------------------------------------------------------


async def test_streaming_requests_include_usage() -> None:
    """THE one-line fix: without this flag the API sends no usage chunk
    at all and the harvest branch below is dead code."""
    captured: Dict[str, Any] = {}
    client = _client_with_stream(
        _stream_chunks(usage_details=None), captured
    )
    await _drain(client)
    assert captured["stream_options"] == {"include_usage": True}
    assert captured["stream"] is True


async def test_streaming_usage_lands_in_response() -> None:
    captured: Dict[str, Any] = {}
    client = _client_with_stream(
        _stream_chunks(usage_details=SimpleNamespace(cached_tokens=3)), captured
    )
    events = await _drain(client)
    resp = [e for e in events if e["type"] == "message_complete"][0]["response"]
    assert resp.usage.input_tokens == 11
    assert resp.usage.output_tokens == 5
    assert resp.usage.total_tokens == 16
    assert resp.text == "hello"


async def test_streaming_extracts_cached_tokens() -> None:
    captured: Dict[str, Any] = {}
    client = _client_with_stream(
        _stream_chunks(usage_details=SimpleNamespace(cached_tokens=7)), captured
    )
    events = await _drain(client)
    resp = [e for e in events if e["type"] == "message_complete"][0]["response"]
    assert resp.usage.cache_read_input_tokens == 7


async def test_streaming_usage_details_as_dict() -> None:
    """vLLM-style OpenAI-compatible servers ship the details block as a
    plain dict, not a typed object."""
    captured: Dict[str, Any] = {}
    client = _client_with_stream(
        _stream_chunks(usage_details={"cached_tokens": 4}), captured
    )
    events = await _drain(client)
    resp = [e for e in events if e["type"] == "message_complete"][0]["response"]
    assert resp.usage.cache_read_input_tokens == 4


def test_non_streaming_parse_usage_extracts_cached_tokens() -> None:
    client = OpenAIClient(api_key="sk-mock")
    usage = client._parse_usage(
        SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            prompt_tokens_details=SimpleNamespace(cached_tokens=60),
        )
    )
    assert usage.input_tokens == 100
    assert usage.output_tokens == 20
    assert usage.cache_read_input_tokens == 60


def test_parse_usage_tolerates_missing_details() -> None:
    client = OpenAIClient(api_key="sk-mock")
    usage = client._parse_usage(
        SimpleNamespace(prompt_tokens=10, completion_tokens=2)
    )
    assert usage.cache_read_input_tokens == 0
    assert client._parse_usage(None).total_tokens == 0


# ---------------------------------------------------------------------------
# max_tokens → max_completion_tokens: proactive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    ["o1", "o1-preview", "o3", "o3-mini", "o4-mini", "gpt-5", "gpt-5.2-codex"],
)
def test_reasoning_families_require_max_completion_tokens(model: str) -> None:
    assert _model_requires_max_completion_tokens(model) is True


@pytest.mark.parametrize(
    "model",
    ["gpt-4o", "gpt-4.1-mini", "gpt-3.5-turbo", "open-mistral", "llama-3"],
)
def test_classic_families_keep_max_tokens(model: str) -> None:
    assert _model_requires_max_completion_tokens(model) is False


def test_build_kwargs_sends_max_completion_tokens_for_o_series() -> None:
    client = OpenAIClient(api_key="sk-mock")
    kwargs = client._build_kwargs(_req(model="o3-mini", max_tokens=2048))
    assert kwargs["max_completion_tokens"] == 2048
    assert "max_tokens" not in kwargs


def test_build_kwargs_keeps_max_tokens_for_classic_models() -> None:
    client = OpenAIClient(api_key="sk-mock")
    kwargs = client._build_kwargs(_req(model="gpt-4o", max_tokens=2048))
    assert kwargs["max_tokens"] == 2048
    assert "max_completion_tokens" not in kwargs


# ---------------------------------------------------------------------------
# max_tokens → max_completion_tokens: reactive heal
# ---------------------------------------------------------------------------


_RENAME_400 = (
    "Unsupported parameter: 'max_tokens' is not supported with this model. "
    "Use 'max_completion_tokens' instead."
)


def test_heal_hook_renames_on_the_live_400_message() -> None:
    client = OpenAIClient(api_key="sk-mock")
    kwargs = {"model": "o9-future", "max_tokens": 512, "messages": []}
    healed = client._heal_request_kwargs(kwargs, RuntimeError(_RENAME_400))
    assert healed is not None
    assert healed["max_completion_tokens"] == 512
    assert "max_tokens" not in healed
    # Original kwargs untouched (hook contract: pure).
    assert kwargs["max_tokens"] == 512


def test_heal_hook_accepts_message_naming_only_the_replacement() -> None:
    msg = "use 'max_completion_tokens' instead of 'max_tokens' for this model"
    client = OpenAIClient(api_key="sk-mock")
    healed = client._heal_request_kwargs(
        {"model": "m", "max_tokens": 64}, RuntimeError(msg)
    )
    assert healed is not None and healed["max_completion_tokens"] == 64


@pytest.mark.parametrize(
    "msg,kwargs",
    [
        # Unrelated 400 — never heal blindly.
        ("rate limit exceeded", {"model": "m", "max_tokens": 64}),
        # Mentions max_tokens but is a range complaint, not the rename.
        ("max_tokens must be at least 1", {"model": "m", "max_tokens": 0}),
        # The rename message but nothing to rename.
        (_RENAME_400, {"model": "m"}),
    ],
)
def test_heal_hook_returns_none_when_not_applicable(
    msg: str, kwargs: Dict[str, Any]
) -> None:
    client = OpenAIClient(api_key="sk-mock")
    assert client._heal_request_kwargs(kwargs, RuntimeError(msg)) is None


async def test_send_heals_rename_and_emits_drift_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    events: List[Dict[str, Any]] = []
    client = OpenAIClient(api_key="sk-mock", event_sink=events.append)
    calls: List[Dict[str, Any]] = []

    fake_raw = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
        model="o9-future",
        id="cmpl_mock",
    )

    async def create(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        if "max_tokens" in kwargs:
            raise RuntimeError(_RENAME_400)
        return fake_raw

    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    with caplog.at_level(logging.WARNING, logger="geny_executor.llm_client.base"):
        resp = await client.create_message(
            model_config=ModelConfig(model="o9-future", max_tokens=512),
            messages=[{"role": "user", "content": "x"}],
        )

    assert resp.text == "ok"
    assert len(calls) == 2
    assert calls[1]["max_completion_tokens"] == 512
    assert "max_tokens" not in calls[1]

    drift = [e for e in events if e["type"] == "llm_client.drift_healed"]
    assert len(drift) == 1
    assert drift[0] == {
        "type": "llm_client.drift_healed",
        "provider": "openai",
        "model": "o9-future",
        "field": "max_tokens",
        "message": _RENAME_400,
    }
    assert any(
        r.levelno == logging.WARNING and "self-healed" in r.getMessage()
        for r in caplog.records
    )


async def test_streaming_heals_rename_at_stream_open() -> None:
    """The 400 surfaces at ``create()`` even with ``stream=True`` — the
    heal retries stream setup and the consumer sees a normal stream."""
    events: List[Dict[str, Any]] = []
    client = OpenAIClient(api_key="sk-mock", event_sink=events.append)
    calls: List[Dict[str, Any]] = []
    chunks = _stream_chunks(usage_details=None)

    async def create(**kwargs: Any):
        calls.append(kwargs)
        if "max_tokens" in kwargs:
            raise RuntimeError(_RENAME_400)

        async def _gen():
            for chunk in chunks:
                yield chunk

        return _gen()

    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    collected = []
    async for evt in client.create_message_stream(
        model_config=ModelConfig(model="o9-future", max_tokens=64),
        messages=[{"role": "user", "content": "x"}],
    ):
        collected.append(evt)

    assert len(calls) == 2
    assert calls[1]["max_completion_tokens"] == 64
    resp = [e for e in collected if e["type"] == "message_complete"][0]["response"]
    assert resp.text == "hello"
    assert resp.usage.input_tokens == 11  # usage survives the heal+retry
    drift = [e for e in events if e["type"] == "llm_client.drift_healed"]
    assert len(drift) == 1 and drift[0]["field"] == "max_tokens"


async def test_unhealable_stream_error_classified() -> None:
    client = OpenAIClient(api_key="sk-mock")

    async def create(**kwargs: Any):
        raise RuntimeError("connection reset by peer")

    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    from geny_executor.core.errors import APIError

    with pytest.raises(APIError) as ei:
        async for _ in client.create_message_stream(
            model_config=ModelConfig(model="gpt-4o", max_tokens=64),
            messages=[{"role": "user", "content": "x"}],
        ):
            pass
    assert ei.value.category is ErrorCategory.UNKNOWN


# ---------------------------------------------------------------------------
# Provenance (APIResponse.raw)
# ---------------------------------------------------------------------------


def test_parse_response_raw_carries_provenance() -> None:
    client = OpenAIClient(api_key="sk-mock")
    fake_raw = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        model="gpt-mock",
        id="cmpl_mock",
    )
    resp = client._parse_response(fake_raw)
    assert resp.raw["provider"] == "openai"
    assert resp.raw["sdk_version"] == openai_sdk.__version__
    assert resp.raw["response"] is fake_raw


async def test_streaming_response_raw_carries_provenance() -> None:
    captured: Dict[str, Any] = {}
    client = _client_with_stream(_stream_chunks(usage_details=None), captured)
    events = await _drain(client)
    resp = [e for e in events if e["type"] == "message_complete"][0]["response"]
    assert resp.raw["provider"] == "openai"
    assert resp.raw["sdk_version"] == openai_sdk.__version__


def test_vllm_provenance_reports_vllm_provider_with_openai_sdk() -> None:
    from geny_executor.llm_client.vllm import VLLMClient

    client = VLLMClient(base_url="http://localhost:8000/v1")
    prov = client._provenance()
    assert prov["provider"] == "vllm"
    assert prov["sdk_version"] == openai_sdk.__version__
