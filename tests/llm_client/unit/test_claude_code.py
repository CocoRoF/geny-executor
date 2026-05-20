"""Tests for :class:`ClaudeCodeCLIClient` (Phase B2).

End-to-end coverage uses the fake ``claude`` binary under
``tests/_fixtures/fake_claude.py`` driven by the ``FAKE_CLAUDE_SCENARIO``
env var.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import pytest

from geny_executor.core.config import ModelConfig
from geny_executor.core.errors import APIError, ErrorCategory
from geny_executor.llm_client.claude_code import ClaudeCodeCLIClient
from geny_executor.llm_client.registry import ClientRegistry


FAKE_CLAUDE = str(
    (Path(__file__).resolve().parents[2] / "_fixtures" / "fake_claude.py")
)


def _client(scenario: str = "ok_text", text: str | None = None, **kwargs) -> ClaudeCodeCLIClient:
    """Build a client wired to the fake binary with a scenario env extra.

    Scenario / FAKE_CLAUDE_TEXT are forwarded via ``env_extras`` so they
    survive the runner's env scrub (which only whitelists HOME/PATH/etc).
    """
    env_extras = kwargs.pop("env_extras", None) or {}
    env_extras = dict(env_extras)
    env_extras.setdefault("FAKE_CLAUDE_SCENARIO", scenario)
    if text is not None:
        env_extras["FAKE_CLAUDE_TEXT"] = text
    defaults = dict(
        binary_path=FAKE_CLAUDE,
        workspace_dir=os.getcwd(),
        api_key="sk-fake",
        bare_mode=True,
        timeout_s=10.0,
        env_extras=env_extras,
    )
    defaults.update(kwargs)
    return ClaudeCodeCLIClient(**defaults)


def _setenv(monkeypatch: pytest.MonkeyPatch, scenario: str) -> None:
    """Legacy helper retained for tests that need the host env to track
    the scenario too — most tests now pass it via ``_client(scenario=...)``."""
    monkeypatch.setenv("FAKE_CLAUDE_SCENARIO", scenario)


# ---------------------------------------------------------------------------
# Static surface
# ---------------------------------------------------------------------------


def test_registry_has_claude_code_cli() -> None:
    assert "claude_code_cli" in ClientRegistry.available()
    cls = ClientRegistry.get("claude_code_cli")
    assert cls is ClaudeCodeCLIClient


def test_capabilities_shape() -> None:
    caps = ClaudeCodeCLIClient.capabilities
    assert caps.supports_thinking is True
    assert caps.supports_tools is True
    assert caps.supports_streaming is True
    assert caps.supports_tool_choice is False
    assert caps.supports_stop_sequences is False
    assert caps.supports_top_k is False
    assert caps.supports_structured_output is True
    assert caps.supports_session_continuity is True
    assert caps.supports_mcp_passthrough is True
    assert caps.supports_budget_limit is True
    assert caps.supports_token_usage is True
    assert caps.supports_cost_usage is True
    assert caps.is_subprocess is True
    assert caps.requires_workspace is True
    assert caps.streaming_granularity == "token"


def test_provider_attr() -> None:
    c = _client()
    assert c.provider == "claude_code_cli"


# ---------------------------------------------------------------------------
# Binary resolution
# ---------------------------------------------------------------------------


def test_init_with_explicit_binary() -> None:
    c = _client()
    assert c._binary == FAKE_CLAUDE


def test_send_with_missing_binary_raises_cli_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    monkeypatch.setenv("PATH", "/nowhere")
    c = ClaudeCodeCLIClient(
        binary_path="/totally/missing/claude",
        workspace_dir=os.getcwd(),
        api_key="sk-fake",
    )
    assert c._binary == ""

    async def run():
        req = _make_request()
        with pytest.raises(APIError) as ei:
            await c._send(req)
        assert ei.value.category is ErrorCategory.CLI_NOT_FOUND

    asyncio.run(run())


# ---------------------------------------------------------------------------
# One-shot (json output)
# ---------------------------------------------------------------------------


def _make_request(stream: bool = False, **kwargs):
    from geny_executor.llm_client.types import APIRequest

    base = dict(
        model="sonnet",
        messages=[{"role": "user", "content": "hi"}],
        system="be brief.",
        stream=stream,
    )
    base.update(kwargs)
    return APIRequest(**base)


@pytest.mark.asyncio
async def test_send_oneshot_ok_text() -> None:
    c = _client(text="Hello!")
    resp = await c._send(_make_request())
    assert resp.text == "Hello!"
    assert resp.stop_reason == "end_turn"
    assert resp.usage.input_tokens == 5
    assert resp.usage.cost_usd == pytest.approx(0.0002)
    assert resp.usage.duration_ms == 50
    assert resp.model == "claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_send_oneshot_tool_use_blocks_dropped() -> None:
    """``tool_use`` blocks are dropped from the response — the CLI
    dispatched them internally. ``stop_reason`` is preserved
    verbatim so callers can still tell the CLI ended in a tool turn
    (e.g. CLI hit max-iter mid-loop with pending tool calls). See
    ``StreamJsonAccumulator.finalize`` for the full rationale."""
    c = _client(scenario="ok_tool_use")
    resp = await c._send(_make_request())
    assert resp.tool_calls == []
    assert resp.stop_reason == "tool_use"


@pytest.mark.asyncio
async def test_send_oneshot_auth_failure_maps_category() -> None:
    c = _client(scenario="auth_fail")
    with pytest.raises(APIError) as ei:
        await c._send(_make_request())
    assert ei.value.category is ErrorCategory.CLI_AUTH_FAILED


@pytest.mark.asyncio
async def test_send_oneshot_permission_failure_maps_category() -> None:
    c = _client(scenario="permission_fail")
    with pytest.raises(APIError) as ei:
        await c._send(_make_request())
    assert ei.value.category is ErrorCategory.CLI_PERMISSION_DENIED


@pytest.mark.asyncio
async def test_send_oneshot_crash_maps_protocol_error() -> None:
    c = _client(scenario="crash")
    with pytest.raises(APIError) as ei:
        await c._send(_make_request())
    assert ei.value.category is ErrorCategory.CLI_PROTOCOL_ERROR


@pytest.mark.asyncio
async def test_send_oneshot_timeout() -> None:
    c = _client(scenario="hang")
    c._timeout_s = 0.4
    with pytest.raises(APIError) as ei:
        await c._send(_make_request())
    assert ei.value.category is ErrorCategory.CLI_TIMEOUT


# ---------------------------------------------------------------------------
# Streaming (stream-json output)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_streaming_text() -> None:
    c = _client(text="Hi")
    resp = await c._send(_make_request(stream=True))
    assert resp.text == "Hi"
    assert resp.stop_reason == "end_turn"
    assert resp.usage.cost_usd == pytest.approx(0.0002)
    assert resp.model == "claude-sonnet-4-6"
    assert resp.message_id == "fake-session-1"


@pytest.mark.asyncio
async def test_send_streaming_thinking() -> None:
    c = _client(scenario="ok_thinking")
    resp = await c._send(_make_request(stream=True))
    assert resp.thinking_blocks
    assert resp.thinking_blocks[0].thinking_text.startswith("Thinking step 1")
    assert resp.text == "Answer."


@pytest.mark.asyncio
async def test_create_message_stream_yields_text_deltas() -> None:
    c = _client(text="abc")
    events = []
    async for evt in c.create_message_stream(
        model_config=ModelConfig(model="sonnet"),
        messages=[{"role": "user", "content": "go"}],
    ):
        events.append(evt)
    text_deltas = [e for e in events if e.get("type") == "text_delta"]
    assert "".join(d["text"] for d in text_deltas) == "abc"
    assert any(e.get("type") == "message_complete" for e in events)
    assert any(e.get("type") == "result" for e in events)


@pytest.mark.asyncio
async def test_create_message_stream_message_complete_carries_response() -> None:
    """Regression: the terminal ``message_complete`` event must carry an
    assembled ``APIResponse`` in ``chunk["response"]``. The s06_api
    stage's streaming consumer raises ``Stream ended without
    message_complete`` when this field is missing — that was the
    Claude-Code-as-Stage-6 outage symptom.
    """
    c = _client(text="hello world")
    completes = []
    async for evt in c.create_message_stream(
        model_config=ModelConfig(model="sonnet"),
        messages=[{"role": "user", "content": "go"}],
    ):
        if evt.get("type") == "message_complete":
            completes.append(evt)

    # Exactly one terminal envelope, populated.
    assert len(completes) == 1
    final = completes[0]
    assert "response" in final, "message_complete must include the response"
    resp = final["response"]
    assert resp.text == "hello world"
    assert resp.stop_reason == "end_turn"
    assert resp.usage.cost_usd is not None
    assert resp.model  # resolved from the system envelope or model_config


@pytest.mark.asyncio
async def test_create_message_stream_message_form_collects_text() -> None:
    """Regression: when Claude Code emits the ``assistant.message.content[]``
    shape (the 2.x default, no ``--include-partial-messages``), text
    blocks must be accumulated into the terminal APIResponse. The
    earlier accumulator only handled the delta shape so every session
    came back with ``output_len=0`` even though the CLI did real work.
    """
    c = _client(scenario="ok_message_form", text="안녕하세요")
    events = []
    async for evt in c.create_message_stream(
        model_config=ModelConfig(model="sonnet"),
        messages=[{"role": "user", "content": "ㅎㅇ"}],
    ):
        events.append(evt)

    text_deltas = [e for e in events if e.get("type") == "text_delta"]
    assert text_deltas, "message form must produce at least one text_delta"
    assert "".join(d["text"] for d in text_deltas) == "안녕하세요"

    completes = [e for e in events if e.get("type") == "message_complete"]
    assert len(completes) == 1
    resp = completes[0]["response"]
    assert resp.text == "안녕하세요"
    assert resp.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_create_message_stream_authentication_failed_raises() -> None:
    """Regression: the CLI emits an ``assistant`` envelope with
    ``error=authentication_failed`` + placeholder text "Not logged
    in" when no credential is available. The placeholder must not
    be returned as the assistant's reply — raise APIError so the
    pipeline surfaces the auth problem to the user."""
    c = _client(scenario="message_form_auth_failed")
    with pytest.raises(APIError) as exc_info:
        async for _ in c.create_message_stream(
            model_config=ModelConfig(model="sonnet"),
            messages=[{"role": "user", "content": "hi"}],
        ):
            pass
    assert exc_info.value.category == ErrorCategory.CLI_AUTH_FAILED


@pytest.mark.asyncio
async def test_send_streaming_message_form_text() -> None:
    """Non-streaming caller via ``_send(stream=True)`` must also
    collect text from the message form. Mirrors the streaming-from-
    consumer-POV test above for the assembler path."""
    c = _client(scenario="ok_message_form", text="배포 완료")
    resp = await c._send(_make_request(stream=True))
    assert resp.text == "배포 완료"
    assert resp.stop_reason == "end_turn"


# ---------------------------------------------------------------------------
# Argv shape verification via the echo scenario
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_argv_carries_bare_and_workspace(monkeypatch) -> None:
    # ``--bare`` is auto-stripped on the OAuth path (no
    # ANTHROPIC_API_KEY in env). Pin the API-key env so this argv
    # surface test exercises the API-key path.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    c = _client(scenario="echo_argv")
    resp = await c._send(_make_request(model="opus", system="rule X"))
    import json

    argv = json.loads(resp.text)
    assert "--print" in argv
    assert "--bare" in argv
    assert "--model" in argv and "opus" in argv
    assert "--system-prompt" in argv and "rule X" in argv


def test_send_oneshot_propagates_api_key_via_env() -> None:
    """The fake CLI doesn't inspect env, but ``_env_extras`` should still
    expose ANTHROPIC_API_KEY to the child."""
    c = _client(api_key="sk-special")
    extras = c._env_extras()
    assert extras["ANTHROPIC_API_KEY"] == "sk-special"


# ---------------------------------------------------------------------------
# Pipeline integration: _build_client_for via Pipeline + CredentialBundle
# ---------------------------------------------------------------------------


def test_pipeline_credentials_kwargs_mapping() -> None:
    """``_creds_to_client_kwargs`` knows how to build a ClaudeCodeCLIClient
    from a CredentialBundle entry shaped by Geny's CredentialBundleBuilder."""
    from geny_executor.core.pipeline import _creds_to_client_kwargs
    from geny_executor.llm_client.credentials import ProviderCredentials

    creds = ProviderCredentials(
        api_key="sk-x",
        binary_path=FAKE_CLAUDE,
        extras={
            "workspace_root": "/tmp/sess",
            "bare_mode": True,
            "default_permission_mode": "acceptEdits",
            "max_budget_usd": 1.0,
            "settings_path": "/etc/settings.json",
            "mcp_config": "/etc/mcp.json",
            "allow_tools": ("Read",),
            "disallow_tools": (),
            "extra_args": (),
            "timeout_s": 90.0,
        },
    )
    kwargs = _creds_to_client_kwargs("claude_code_cli", creds)
    assert kwargs["api_key"] == "sk-x"
    assert kwargs["binary_path"] == FAKE_CLAUDE
    assert kwargs["workspace_dir"] == "/tmp/sess"  # remapped from workspace_root
    assert kwargs["default_permission_mode"] == "acceptEdits"
    assert kwargs["max_budget_usd"] == 1.0
    assert kwargs["allow_tools"] == ("Read",)
    assert kwargs["timeout_s"] == 90.0
