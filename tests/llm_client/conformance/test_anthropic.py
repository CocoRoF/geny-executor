"""Anthropic provider conformance."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import pytest

from geny_executor.llm_client import AnthropicClient
from geny_executor.llm_client.base import BaseClient

from tests.llm_client.conformance.harness import ConformanceTestSuite


class TestAnthropicConformance(ConformanceTestSuite):
    provider_name = "anthropic"

    def make_client(self, *, mode="mocked") -> BaseClient:
        # Mocked mode uses a dummy api_key; no actual HTTP is performed by
        # the harness tests that only inspect capabilities.
        return AnthropicClient(api_key="sk-mock")

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
