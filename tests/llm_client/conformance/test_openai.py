"""OpenAI provider conformance."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import pytest

pytest.importorskip("openai")

from geny_executor.llm_client.openai import OpenAIClient  # noqa: E402
from geny_executor.llm_client.base import BaseClient  # noqa: E402

from tests.llm_client.conformance.harness import ConformanceTestSuite  # noqa: E402


class TestOpenAIConformance(ConformanceTestSuite):
    provider_name = "openai"

    def make_client(self, *, mode="mocked") -> BaseClient:
        return OpenAIClient(api_key="sk-mock")

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
