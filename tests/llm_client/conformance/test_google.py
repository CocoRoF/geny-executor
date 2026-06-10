"""Google provider conformance."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import pytest

pytest.importorskip("google.genai")

from types import SimpleNamespace  # noqa: E402

from geny_executor.llm_client.google import GoogleClient  # noqa: E402
from geny_executor.llm_client.base import BaseClient  # noqa: E402

from tests.llm_client.conformance.harness import ConformanceTestSuite  # noqa: E402


def _fake_gemini_stream_chunks():
    """generateContent stream chunks: text part, then a final chunk
    carrying ``usage_metadata`` (the wire ships usage on the last chunk)."""
    part = SimpleNamespace(text="streamed text", thought=False)
    candidate = SimpleNamespace(
        finish_reason=None, content=SimpleNamespace(parts=[part])
    )
    final_candidate = SimpleNamespace(
        finish_reason="STOP", content=SimpleNamespace(parts=[])
    )
    return [
        SimpleNamespace(candidates=[candidate], usage_metadata=None),
        SimpleNamespace(
            candidates=[final_candidate],
            usage_metadata=SimpleNamespace(
                prompt_token_count=8, candidates_token_count=3
            ),
        ),
    ]


class TestGoogleConformance(ConformanceTestSuite):
    provider_name = "google"

    def make_client(self, *, mode="mocked") -> BaseClient:
        return GoogleClient(api_key="sk-mock")

    def make_usage_stream_client(self) -> BaseClient:
        async def generate_content_stream(**kwargs):
            async def _gen():
                for chunk in _fake_gemini_stream_chunks():
                    yield chunk

            return _gen()

        client = GoogleClient(api_key="sk-mock")
        client._client = SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(
                    generate_content_stream=generate_content_stream
                )
            )
        )
        return client

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
