"""Conformance for the branded local (OpenAI-compatible) providers.

ollama / lmstudio / custom run the shared :class:`ConformanceTestSuite`
contract — same canonical request/response + streaming-usage promise the
``openai`` and ``vllm`` suites enforce.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import pytest

pytest.importorskip("openai")

from geny_executor.llm_client.base import BaseClient  # noqa: E402
from geny_executor.llm_client.openai_compatible import (  # noqa: E402
    CustomOpenAIClient,
    LMStudioClient,
    OllamaClient,
)

from tests.llm_client.conformance.harness import ConformanceTestSuite  # noqa: E402
from tests.llm_client.conformance.test_openai import make_fake_openai_sdk  # noqa: E402


class TestOllamaConformance(ConformanceTestSuite):
    provider_name = "ollama"

    def make_client(self, *, mode="mocked") -> BaseClient:
        return OllamaClient()  # default base_url

    def make_usage_stream_client(self) -> BaseClient:
        client = OllamaClient()
        # Ollama ships prompt_tokens_details as a plain dict.
        client._client, _ = make_fake_openai_sdk(usage_details_as_dict=True)
        return client

    def test_default_base_url(self) -> None:
        assert self.make_client()._base_url == "http://localhost:11434/v1"

    def test_tool_capable_by_default(self) -> None:
        client = self.make_client()
        assert client.supports("tools") is True
        assert client.supports("tool_choice") is True


class TestLMStudioConformance(ConformanceTestSuite):
    provider_name = "lmstudio"

    def make_client(self, *, mode="mocked") -> BaseClient:
        return LMStudioClient()

    def make_usage_stream_client(self) -> BaseClient:
        client = LMStudioClient()
        client._client, _ = make_fake_openai_sdk()
        return client

    def test_default_base_url(self) -> None:
        assert self.make_client()._base_url == "http://127.0.0.1:1234/v1"


class TestCustomConformance(ConformanceTestSuite):
    provider_name = "custom"

    def make_client(self, *, mode="mocked") -> BaseClient:
        # ``custom`` has no default endpoint — a base_url is mandatory.
        return CustomOpenAIClient(base_url="http://localhost:8080/v1")

    def make_usage_stream_client(self) -> BaseClient:
        client = CustomOpenAIClient(base_url="http://localhost:8080/v1")
        client._client, _ = make_fake_openai_sdk()
        return client

    def test_requires_base_url(self) -> None:
        with pytest.raises(ValueError):
            CustomOpenAIClient()

    def test_not_subprocess(self) -> None:
        assert self.make_client().capabilities.is_subprocess is False
