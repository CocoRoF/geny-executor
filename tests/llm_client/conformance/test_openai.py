"""OpenAI provider conformance."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from types import SimpleNamespace

import pytest

pytest.importorskip("openai")

from geny_executor.llm_client.openai import OpenAIClient  # noqa: E402
from geny_executor.llm_client.base import BaseClient  # noqa: E402

from tests.llm_client.conformance.harness import ConformanceTestSuite  # noqa: E402


def make_fake_openai_sdk(*, usage_details_as_dict: bool = False):
    """A stub standing in for ``AsyncOpenAI`` — Chat Completions stream
    that ends with the usage-only chunk ``stream_options.include_usage``
    unlocks. ``usage_details_as_dict=True`` mimics vLLM-style servers
    that ship ``prompt_tokens_details`` as a plain dict instead of a
    typed object. Shared with the vLLM conformance suite."""
    details = {"cached_tokens": 3} if usage_details_as_dict else SimpleNamespace(
        cached_tokens=3
    )
    chunks = [
        SimpleNamespace(
            model="mock-model",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="streamed text", tool_calls=None),
                    finish_reason=None,
                )
            ],
        ),
        SimpleNamespace(
            model="mock-model",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, tool_calls=None),
                    finish_reason="stop",
                )
            ],
        ),
        # The usage chunk: empty choices, usage attached. This only
        # arrives when stream_options={"include_usage": True} was sent.
        SimpleNamespace(
            model="mock-model",
            choices=[],
            usage=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=5,
                prompt_tokens_details=details,
            ),
        ),
    ]

    captured_kwargs: dict = {}

    async def _aiter():
        for chunk in chunks:
            yield chunk

    async def create(**kwargs):
        captured_kwargs.clear()
        captured_kwargs.update(kwargs)
        return _aiter()

    sdk = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return sdk, captured_kwargs


class TestOpenAIConformance(ConformanceTestSuite):
    provider_name = "openai"

    def make_client(self, *, mode="mocked") -> BaseClient:
        return OpenAIClient(api_key="sk-mock")

    def make_usage_stream_client(self) -> BaseClient:
        client = OpenAIClient(api_key="sk-mock")
        client._client, _ = make_fake_openai_sdk()
        return client

    def test_openai_capabilities(self) -> None:
        client = self.make_client()
        assert client.supports("thinking") is False
        assert client.supports("top_k") is False
        assert client.supports("tools") is True
        assert client.supports("tool_choice") is True
        # OpenAI does support JSON schema response_format.
        assert client.supports("structured_output") is True

    def test_openai_not_subprocess(self) -> None:
        client = self.make_client()
        assert client.capabilities.is_subprocess is False
