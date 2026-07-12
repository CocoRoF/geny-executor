"""Anthropic 400 classification + heal-hook routing (2.2.0, audit §3.4).

The 2.1.x TOKEN_LIMIT heuristic was ``'token' in msg or 'context' in
msg`` over the whole BadRequestError — which routed *param-shape* 400s
into TOKEN_LIMIT. The canary case is the drift message this client's own
module documents: ``thinking.adaptive.budget_tokens: Extra inputs are
not permitted`` contains "token", so the next thinking-shape drift would
have been diagnosed as "compact the conversation" instead of "the
request shape is stale". The categories drive opposite recovery paths
(TOKEN_LIMIT → recoverable/compaction, BAD_REQUEST → fatal) so the
misroute compounds: retries that can never succeed.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Dict, List

import httpx
import pytest

import anthropic as anthropic_sdk

from geny_executor.core.config import ModelConfig
from geny_executor.core.errors import ErrorCategory
from geny_executor.llm_client.anthropic import (
    AnthropicClient,
    _classify_bad_request_message,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _bad_request(message: str) -> anthropic_sdk.BadRequestError:
    """A real SDK BadRequestError carrying ``message`` — classification
    must work on what the SDK actually raises, not on synthetic shims."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(400, request=request, text="{}")
    return anthropic_sdk.BadRequestError(
        message, response=response, body={"error": {"message": message}}
    )


def _fake_message(text: str = "ok") -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=3,
            output_tokens=2,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        model="claude-mock",
        id="msg_mock",
    )


# ---------------------------------------------------------------------------
# Pure classifier
# ---------------------------------------------------------------------------


def test_budget_tokens_drift_message_is_bad_request() -> None:
    """The audit's exact misclassification case."""
    msg = "thinking.adaptive.budget_tokens: extra inputs are not permitted"
    assert _classify_bad_request_message(msg) is ErrorCategory.BAD_REQUEST


@pytest.mark.parametrize(
    "msg",
    [
        # Real overflow phrasings observed on the live Messages API.
        "prompt is too long: 213116 tokens > 200000 maximum",
        "input length and `max_tokens` exceed context limit: 215748 + 8192 > 204798",
        # Anchors the spec mandates even if Anthropic's wording shifts.
        "this model's maximum context length is 200000 tokens",
        "input length exceeds the limit for this model",
        "request contains too many tokens",
    ],
)
def test_real_overflow_messages_are_token_limit(msg: str) -> None:
    assert _classify_bad_request_message(msg.lower()) is ErrorCategory.TOKEN_LIMIT


@pytest.mark.parametrize(
    "msg",
    [
        # Deprecation notices (the 2.1.2 incident wording).
        "temperature is deprecated for this model.",
        "`top_p` is deprecated for this model.",
        # Dotted param paths — pydantic-style validation rejections.
        "messages.0.content.0.image._meta: extra inputs are not permitted",
        "thinking.type.enabled is not supported for this model. tokens",
        # Generic param-shape with a 'token'-ish word inside (the trap).
        "thinking.adaptive.budget_tokens: extra inputs are not permitted",
    ],
)
def test_param_shape_messages_are_bad_request(msg: str) -> None:
    assert _classify_bad_request_message(msg.lower()) is ErrorCategory.BAD_REQUEST


def test_unrecognized_400_defaults_to_bad_request() -> None:
    """No anchor, no param path → BAD_REQUEST, never TOKEN_LIMIT. The
    old heuristic's 'token'/'context' substrings are not enough alone."""
    assert (
        _classify_bad_request_message("there was a problem with tokens")
        is ErrorCategory.BAD_REQUEST
    )
    assert (
        _classify_bad_request_message("invalid context setting")
        is ErrorCategory.BAD_REQUEST
    )


# ---------------------------------------------------------------------------
# Through _classify_error with real SDK exceptions
# ---------------------------------------------------------------------------


def test_classify_error_budget_tokens_400_is_bad_request() -> None:
    client = AnthropicClient(api_key="sk-mock")
    err = client._classify_error(
        _bad_request(
            "thinking.adaptive.budget_tokens: Extra inputs are not permitted"
        )
    )
    assert err.category is ErrorCategory.BAD_REQUEST
    assert err.status_code == 400


def test_classify_error_context_overflow_400_is_token_limit() -> None:
    client = AnthropicClient(api_key="sk-mock")
    err = client._classify_error(
        _bad_request("prompt is too long: 213116 tokens > 200000 maximum")
    )
    assert err.category is ErrorCategory.TOKEN_LIMIT
    assert err.status_code == 400


def test_classify_error_other_categories_unchanged() -> None:
    """The non-400 chain is untouched by the tightening."""
    client = AnthropicClient(api_key="sk-mock")
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    rl = anthropic_sdk.RateLimitError(
        "rate limited",
        response=httpx.Response(429, request=request, text="{}"),
        body=None,
    )
    assert client._classify_error(rl).category is ErrorCategory.RATE_LIMITED


# ---------------------------------------------------------------------------
# Heal-hook routing (the 2.1.2 retry net, now through BaseClient)
# ---------------------------------------------------------------------------


class _Dep400(Exception):
    message = "temperature is deprecated for this model."


async def test_send_heals_deprecation_and_emits_drift_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """End-to-end through ``create_message``: first call 400s with the
    deprecation message, the heal strips temperature and retries, and
    the success is reported via ``llm_client.drift_healed`` + WARNING."""
    events: List[Dict[str, Any]] = []
    client = AnthropicClient(api_key="sk-mock", event_sink=events.append)
    calls: List[Dict[str, Any]] = []

    async def create(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        if len(calls) == 1:
            raise _Dep400()
        return _fake_message()

    client._client = SimpleNamespace(
        messages=SimpleNamespace(create=create)
    )

    with caplog.at_level(logging.WARNING, logger="geny_executor.llm_client.base"):
        resp = await client.create_message(
            model_config=ModelConfig(model="claude-sonnet-4-6", temperature=0.7),
            messages=[{"role": "user", "content": "x"}],
        )

    assert resp.text == "ok"
    assert len(calls) == 2
    assert "temperature" in calls[0]
    assert "temperature" not in calls[1]

    drift = [e for e in events if e["type"] == "llm_client.drift_healed"]
    assert len(drift) == 1
    assert drift[0]["provider"] == "anthropic"
    assert drift[0]["field"] == "temperature"
    assert drift[0]["model"] == "claude-sonnet-4-6"
    assert "deprecated" in drift[0]["message"]
    assert any(
        r.levelno == logging.WARNING and "self-healed" in r.getMessage()
        for r in caplog.records
    )


async def test_stream_heals_deprecation_and_emits_drift_event() -> None:
    """Streaming path: the SDK validates kwargs at stream open, the heal
    retries the whole stream, and the drift event fires before
    message_complete reaches the consumer."""

    class _FakeStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

        def __aiter__(self):
            # 2.50.0 (TTFT D1): full-event iteration contract.
            async def _gen():
                yield SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(type="text_delta", text="healed"),
                )

            return _gen()

        async def get_final_message(self):
            return _fake_message("healed")

    events: List[Dict[str, Any]] = []
    client = AnthropicClient(api_key="sk-mock", event_sink=events.append)
    calls: List[Dict[str, Any]] = []

    def stream(**kwargs: Any) -> _FakeStream:
        calls.append(kwargs)
        if len(calls) == 1:
            raise _Dep400()
        return _FakeStream()

    client._client = SimpleNamespace(messages=SimpleNamespace(stream=stream))

    collected = []
    async for evt in client.create_message_stream(
        model_config=ModelConfig(model="claude-sonnet-4-6", temperature=0.7),
        messages=[{"role": "user", "content": "x"}],
    ):
        collected.append(evt)

    assert len(calls) == 2
    assert "temperature" not in calls[1]
    completes = [e for e in collected if e["type"] == "message_complete"]
    assert completes and completes[0]["response"].text == "healed"
    drift = [e for e in events if e["type"] == "llm_client.drift_healed"]
    assert len(drift) == 1
    assert drift[0]["field"] == "temperature"


def test_heal_hook_routes_to_module_table() -> None:
    """``_heal_request_kwargs`` delegates to the 2.1.x module function —
    identical behaviour, new seam."""
    client = AnthropicClient(api_key="sk-mock")
    kwargs = {"model": "m", "temperature": 0.5, "max_tokens": 64}
    healed = client._heal_request_kwargs(kwargs, _Dep400())
    assert healed is not None
    assert "temperature" not in healed
    assert client._heal_request_kwargs(
        {"model": "m", "max_tokens": 64}, RuntimeError("rate limit")
    ) is None


# ---------------------------------------------------------------------------
# Provenance (APIResponse.raw)
# ---------------------------------------------------------------------------


def test_parse_response_raw_carries_provenance() -> None:
    client = AnthropicClient(api_key="sk-mock")
    fake = _fake_message()
    resp = client._parse_response(fake)
    assert resp.raw["provider"] == "anthropic"
    assert resp.raw["sdk_version"] == anthropic_sdk.__version__
    assert resp.raw["response"] is fake
