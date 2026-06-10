"""Claude Code CLI provider conformance.

Phase B3 plugs ClaudeCodeCLIClient into the conformance harness. Mocked
mode is the default — every test drives the fake CLI under
``tests/_fixtures/fake_claude.py``. Capability-gated tests inherit the
broader suite from :mod:`ConformanceTestSuite`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import pytest

from geny_executor.core.config import ModelConfig
from geny_executor.core.errors import ErrorCategory, APIError
from geny_executor.llm_client.claude_code import ClaudeCodeCLIClient
from geny_executor.llm_client.base import BaseClient

from tests.llm_client.conformance.harness import ConformanceTestSuite


FAKE_CLAUDE = str(
    (Path(__file__).resolve().parents[2] / "_fixtures" / "fake_claude.py")
)


class TestClaudeCodeCLIConformance(ConformanceTestSuite):
    provider_name = "claude_code_cli"

    # ----------------------------------------------------------- fixture
    def make_client(
        self,
        *,
        mode="mocked",
        scenario: str = "ok_text",
        text: str | None = None,
    ) -> BaseClient:
        env_extras = {"FAKE_CLAUDE_SCENARIO": scenario}
        if text is not None:
            env_extras["FAKE_CLAUDE_TEXT"] = text
        return ClaudeCodeCLIClient(
            binary_path=FAKE_CLAUDE,
            workspace_dir=os.getcwd(),
            api_key="sk-mock",
            bare_mode=True,
            timeout_s=5.0,
            env_extras=env_extras,
        )

    def make_usage_stream_client(self) -> BaseClient:
        # The fake CLI's stream-json output carries usage on the result
        # line — the real wire shape, no extra stubbing required.
        return self.make_client(text="usage probe")

    # ----------------------------------------------------------- shape
    def test_is_subprocess(self) -> None:
        client = self.make_client()
        assert client.capabilities.is_subprocess is True
        assert client.capabilities.requires_workspace is True

    def test_supports_session_continuity_and_mcp(self) -> None:
        client = self.make_client()
        assert client.supports("session_continuity") is True
        assert client.supports("mcp_passthrough") is True

    def test_supports_thinking_and_budget_limit(self) -> None:
        client = self.make_client()
        assert client.supports("thinking") is True
        assert client.supports("budget_limit") is True

    def test_drops_unsupported_fields(self) -> None:
        client = self.make_client()
        for f in ("tool_choice", "stop_sequences", "top_k", "temperature", "top_p", "max_tokens"):
            assert f in client.capabilities.drops, f

    # ----------------------------------------------------------- end-to-end
    @pytest.mark.asyncio
    async def test_basic_text_completion(self) -> None:
        client = self.make_client(text="hi")
        resp = await client.create_message(
            model_config=ModelConfig(model="sonnet"),
            messages=[{"role": "user", "content": "say hi"}],
        )
        assert resp.text == "hi"
        assert resp.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_streaming_yields_text_deltas(self) -> None:
        client = self.make_client(text="abc")
        events = []
        async for evt in client.create_message_stream(
            model_config=ModelConfig(model="sonnet"),
            messages=[{"role": "user", "content": "say abc"}],
        ):
            events.append(evt)
        text_deltas = [e["text"] for e in events if e.get("type") == "text_delta"]
        assert "".join(text_deltas) == "abc"

    @pytest.mark.asyncio
    async def test_token_usage_and_cost_populated(self) -> None:
        client = self.make_client(text="hello")
        resp = await client.create_message(
            model_config=ModelConfig(model="sonnet"),
            messages=[{"role": "user", "content": "x"}],
        )
        assert resp.usage.input_tokens > 0
        assert resp.usage.output_tokens > 0
        assert resp.usage.cost_usd is not None
        assert resp.usage.duration_ms is not None

    @pytest.mark.asyncio
    async def test_tool_use_blocks_dropped(self) -> None:
        """The CLI handles tool dispatch internally — ``tool_use``
        blocks observed in its output are intentionally dropped from
        the assembled response so host pipelines (Geny's Stage 10,
        the canonical reference consumer) don't try to re-dispatch
        them and ghost-error. ``stop_reason`` is preserved so callers
        can still see the CLI ended in a tool turn. See
        ``StreamJsonAccumulator.finalize`` for the full rationale."""
        client = self.make_client(scenario="ok_tool_use")
        resp = await client.create_message(
            model_config=ModelConfig(model="sonnet"),
            messages=[{"role": "user", "content": "read /tmp/x"}],
        )
        assert resp.tool_calls == []
        assert resp.stop_reason == "tool_use"

    @pytest.mark.asyncio
    async def test_thinking_blocks_returned(self) -> None:
        client = self.make_client(scenario="ok_thinking")
        # create_message_stream is an async generator — iterate directly.
        events = []
        async for evt in client.create_message_stream(
            model_config=ModelConfig(
                model="opus",
                thinking_enabled=True,
                thinking_budget_tokens=10_000,
            ),
            messages=[{"role": "user", "content": "think"}],
        ):
            events.append(evt)
        kinds = {e.get("type") for e in events}
        assert "thinking_delta" in kinds

    @pytest.mark.asyncio
    async def test_translates_auth_error(self) -> None:
        client = self.make_client(scenario="auth_fail")
        with pytest.raises(APIError) as ei:
            await client.create_message(
                model_config=ModelConfig(model="sonnet"),
                messages=[{"role": "user", "content": "x"}],
            )
        assert ei.value.category is ErrorCategory.CLI_AUTH_FAILED

    @pytest.mark.asyncio
    async def test_translates_permission_error(self) -> None:
        client = self.make_client(scenario="permission_fail")
        with pytest.raises(APIError) as ei:
            await client.create_message(
                model_config=ModelConfig(model="sonnet"),
                messages=[{"role": "user", "content": "x"}],
            )
        assert ei.value.category is ErrorCategory.CLI_PERMISSION_DENIED

    @pytest.mark.asyncio
    async def test_binary_not_found_raises_cli_not_found(self) -> None:
        client = ClaudeCodeCLIClient(
            binary_path="/totally/missing/claude",
            workspace_dir=os.getcwd(),
            api_key="sk-mock",
        )
        with pytest.raises(APIError) as ei:
            await client.create_message(
                model_config=ModelConfig(model="sonnet"),
                messages=[{"role": "user", "content": "x"}],
            )
        assert ei.value.category is ErrorCategory.CLI_NOT_FOUND

    @pytest.mark.asyncio
    async def test_timeout_emits_cli_timeout(self) -> None:
        client = self.make_client(scenario="hang")
        client._timeout_s = 0.4
        with pytest.raises(APIError) as ei:
            await client.create_message(
                model_config=ModelConfig(model="sonnet"),
                messages=[{"role": "user", "content": "x"}],
            )
        assert ei.value.category is ErrorCategory.CLI_TIMEOUT
