"""Copilot CLI provider conformance (Phase C2)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import pytest

from geny_executor.core.config import ModelConfig
from geny_executor.core.errors import APIError, ErrorCategory
from geny_executor.llm_client.copilot import CopilotCLIClient
from geny_executor.llm_client.base import BaseClient

from tests.llm_client.conformance.harness import ConformanceTestSuite


FAKE_GH = str(
    (Path(__file__).resolve().parents[2] / "_fixtures" / "fake_gh.py")
)


class TestCopilotCLIConformance(ConformanceTestSuite):
    provider_name = "copilot_cli"

    def make_client(
        self,
        *,
        mode="mocked",
        scenario: str = "ok",
        text: str | None = None,
    ) -> BaseClient:
        env_extras = {"FAKE_GH_SCENARIO": scenario}
        if text is not None:
            env_extras["FAKE_GH_TEXT"] = text
        return CopilotCLIClient(
            gh_binary_path=FAKE_GH,
            timeout_s=5.0,
            env_extras=env_extras,
        )

    # ---------------------------------------------------------------- shape
    def test_is_subprocess(self) -> None:
        c = self.make_client()
        assert c.capabilities.is_subprocess is True
        assert c.capabilities.requires_workspace is False
        assert c.capabilities.streaming_granularity == "none"

    def test_lacks_streaming_and_tools(self) -> None:
        c = self.make_client()
        assert c.supports("streaming") is False
        assert c.supports("tools") is False
        assert c.supports("structured_output") is False
        assert c.supports("token_usage") is False

    # ---------------------------------------------------------------- e2e
    @pytest.mark.asyncio
    async def test_basic_text_completion(self) -> None:
        c = self.make_client(text="Hello!")
        resp = await c.create_message(
            model_config=ModelConfig(model="default"),
            messages=[{"role": "user", "content": "say hi"}],
        )
        assert resp.text == "Hello!"
        assert resp.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_translates_auth_error(self) -> None:
        c = self.make_client(scenario="auth_fail")
        with pytest.raises(APIError) as ei:
            await c.create_message(
                model_config=ModelConfig(model="default"),
                messages=[{"role": "user", "content": "x"}],
            )
        assert ei.value.category is ErrorCategory.CLI_AUTH_FAILED

    @pytest.mark.asyncio
    async def test_translates_not_installed(self) -> None:
        c = self.make_client(scenario="not_installed")
        with pytest.raises(APIError) as ei:
            await c.create_message(
                model_config=ModelConfig(model="default"),
                messages=[{"role": "user", "content": "x"}],
            )
        assert ei.value.category is ErrorCategory.CLI_NOT_FOUND

    @pytest.mark.asyncio
    async def test_streaming_falls_back(self) -> None:
        """copilot_cli's supports_streaming=False → BaseClient default emits
        one message_complete event."""
        c = self.make_client(text="streamed")
        events = []
        async for evt in c.create_message_stream(
            model_config=ModelConfig(model="default"),
            messages=[{"role": "user", "content": "go"}],
        ):
            events.append(evt)
        assert any(e.get("type") == "message_complete" for e in events)

    @pytest.mark.asyncio
    async def test_binary_not_found_raises_cli_not_found(self) -> None:
        c = CopilotCLIClient(gh_binary_path="/totally/missing/gh", timeout_s=2.0)
        with pytest.raises(APIError) as ei:
            await c.create_message(
                model_config=ModelConfig(model="default"),
                messages=[{"role": "user", "content": "x"}],
            )
        assert ei.value.category is ErrorCategory.CLI_NOT_FOUND
