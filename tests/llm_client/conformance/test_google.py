"""Google provider conformance."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import pytest

pytest.importorskip("google.genai")

from geny_executor.llm_client.google import GoogleClient  # noqa: E402
from geny_executor.llm_client.base import BaseClient  # noqa: E402

from tests.llm_client.conformance.harness import ConformanceTestSuite  # noqa: E402


class TestGoogleConformance(ConformanceTestSuite):
    provider_name = "google"

    def make_client(self, *, mode="mocked") -> BaseClient:
        return GoogleClient(api_key="sk-mock")

    def test_google_capabilities(self) -> None:
        client = self.make_client()
        assert client.supports("thinking") is False  # mapped via thinking_config
        assert client.supports("top_k") is True
        assert client.supports("tools") is True
        assert client.supports("tool_choice") is True
        assert client.supports("structured_output") is True

    def test_google_not_subprocess(self) -> None:
        client = self.make_client()
        assert client.capabilities.is_subprocess is False
