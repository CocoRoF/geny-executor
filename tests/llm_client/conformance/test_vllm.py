"""vLLM provider conformance."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import pytest

pytest.importorskip("openai")

from geny_executor.llm_client.vllm import VLLMClient  # noqa: E402
from geny_executor.llm_client.base import BaseClient  # noqa: E402

from tests.llm_client.conformance.harness import ConformanceTestSuite  # noqa: E402


class TestVLLMConformance(ConformanceTestSuite):
    provider_name = "vllm"

    def make_client(self, *, mode="mocked") -> BaseClient:
        # vLLM mandates a base_url at construction time.
        return VLLMClient(base_url="http://localhost:8000/v1")

    def test_vllm_capabilities(self) -> None:
        client = self.make_client()
        assert client.supports("thinking") is False
        assert client.supports("top_k") is False
        # vLLM defaults tools=False (conservative) — overridable via
        # configure_capabilities().
        assert client.supports("tools") is False
        assert client.supports("tool_choice") is False

    def test_vllm_requires_base_url(self) -> None:
        from geny_executor.llm_client.vllm import VLLMClient as V
        with pytest.raises(ValueError):
            V(base_url=None)

    def test_vllm_not_subprocess(self) -> None:
        client = self.make_client()
        assert client.capabilities.is_subprocess is False
