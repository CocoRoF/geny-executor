"""Anthropic provider conformance."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from types import SimpleNamespace


from geny_executor.llm_client import AnthropicClient
from geny_executor.llm_client.base import BaseClient

from tests.llm_client.conformance.harness import ConformanceTestSuite


def _fake_final_message():
    """Shape-compatible stand-in for the SDK's final Message object."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="streamed text")],
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=9,
            output_tokens=4,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        model="claude-mock",
        id="msg_mock",
    )


class _FakeMessageStream:
    """Async context manager mimicking ``client.messages.stream(...)``."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    def __aiter__(self):
        # 2.50.0 (TTFT D1): the client iterates the FULL event stream
        # (raw content_block_delta events), not ``text_stream``.
        async def _gen():
            for text in ("streamed ", "text"):
                yield SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(type="text_delta", text=text),
                )

        return _gen()

    async def get_final_message(self):
        return _fake_final_message()


class TestAnthropicConformance(ConformanceTestSuite):
    provider_name = "anthropic"

    def make_client(self, *, mode="mocked") -> BaseClient:
        # Mocked mode uses a dummy api_key; no actual HTTP is performed by
        # the harness tests that only inspect capabilities.
        return AnthropicClient(api_key="sk-mock")

    def make_usage_stream_client(self) -> BaseClient:
        client = AnthropicClient(api_key="sk-mock")
        client._client = SimpleNamespace(
            messages=SimpleNamespace(stream=lambda **kwargs: _FakeMessageStream())
        )
        return client

    def test_anthropic_capabilities_thinking_and_top_k(self) -> None:
        client = self.make_client()
        assert client.supports("thinking") is True
        assert client.supports("top_k") is True
        assert client.supports("tools") is True
        assert client.supports("tool_choice") is True

    def test_anthropic_not_subprocess(self) -> None:
        client = self.make_client()
        assert client.capabilities.is_subprocess is False
        assert client.capabilities.requires_workspace is False
