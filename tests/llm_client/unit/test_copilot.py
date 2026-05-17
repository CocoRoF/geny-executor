"""Tests for :class:`CopilotCLIClient` (Phase C1)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import pytest

from geny_executor.core.config import ModelConfig
from geny_executor.core.errors import APIError, ErrorCategory
from geny_executor.llm_client.copilot import CopilotCLIClient
from geny_executor.llm_client.registry import ClientRegistry
from geny_executor.llm_client.translators._cli import (
    compose_copilot_prompt,
    copilot_argv,
    parse_plain_text_to_response,
)


FAKE_GH = str(
    (Path(__file__).resolve().parents[2] / "_fixtures" / "fake_gh.py")
)


def _client(scenario: str = "ok", text: str | None = None, **kwargs) -> CopilotCLIClient:
    env_extras = kwargs.pop("env_extras", None) or {}
    env_extras = dict(env_extras)
    env_extras.setdefault("FAKE_GH_SCENARIO", scenario)
    if text is not None:
        env_extras["FAKE_GH_TEXT"] = text
    defaults = dict(
        gh_binary_path=FAKE_GH,
        timeout_s=5.0,
        env_extras=env_extras,
    )
    defaults.update(kwargs)
    return CopilotCLIClient(**defaults)


def _make_request(**kwargs):
    from geny_executor.llm_client.types import APIRequest
    base = dict(
        model="default",
        messages=[{"role": "user", "content": "hi"}],
        system="be brief.",
        stream=False,
    )
    base.update(kwargs)
    return APIRequest(**base)


# ---------------------------------------------------------------------------
# Registry + capability shape
# ---------------------------------------------------------------------------


def test_registry_has_copilot_cli() -> None:
    assert "copilot_cli" in ClientRegistry.available()
    cls = ClientRegistry.get("copilot_cli")
    assert cls is CopilotCLIClient


def test_capabilities_shape() -> None:
    caps = CopilotCLIClient.capabilities
    assert caps.is_subprocess is True
    assert caps.supports_streaming is False
    assert caps.streaming_granularity == "none"
    assert caps.supports_tools is False
    assert caps.supports_thinking is False
    assert caps.supports_structured_output is False
    assert caps.supports_token_usage is False
    assert caps.supports_cost_usage is False
    assert caps.requires_workspace is False
    for f in ("tools", "tool_choice", "thinking_enabled", "stop_sequences", "top_k", "temperature", "top_p", "max_tokens", "response_format", "session_hint"):
        assert f in caps.drops, f


def test_provider_attr() -> None:
    c = _client()
    assert c.provider == "copilot_cli"


# ---------------------------------------------------------------------------
# Prompt composition
# ---------------------------------------------------------------------------


def test_compose_copilot_prompt_single_user_message() -> None:
    out = compose_copilot_prompt("", [{"role": "user", "content": "hi"}])
    assert out == "## User\nhi"


def test_compose_copilot_prompt_with_system() -> None:
    out = compose_copilot_prompt("be terse.", [{"role": "user", "content": "yo"}])
    assert out.startswith("## System\nbe terse.")
    assert "## User\nyo" in out


def test_compose_copilot_prompt_multi_turn() -> None:
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "second"},
    ]
    out = compose_copilot_prompt("", msgs)
    assert "## User\nfirst" in out
    assert "## Assistant\nack" in out
    assert out.endswith("## User\nsecond")


def test_compose_copilot_prompt_tool_result_in_content() -> None:
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": "look at this"},
            {"type": "tool_result", "content": "OK\n"},
        ]},
    ]
    out = compose_copilot_prompt("", msgs)
    assert "look at this" in out
    assert "[tool_result]\nOK" in out


def test_compose_copilot_prompt_system_block_list() -> None:
    sys_blocks = [
        {"type": "text", "text": "rule 1"},
        {"type": "text", "text": "rule 2"},
        {"type": "image"},  # ignored
    ]
    out = compose_copilot_prompt(sys_blocks, [{"role": "user", "content": "ok"}])
    assert "rule 1\nrule 2" in out


# ---------------------------------------------------------------------------
# argv builder
# ---------------------------------------------------------------------------


def test_copilot_argv_minimal() -> None:
    assert copilot_argv() == ["copilot"]


def test_copilot_argv_allow_tools() -> None:
    argv = copilot_argv(allow_tools=["shell(git)", "fs(read)"])
    assert argv == ["copilot", "--allow-tool", "shell(git)", "--allow-tool", "fs(read)"]


def test_copilot_argv_extra_args() -> None:
    argv = copilot_argv(extra_args=["--verbose"])
    assert argv[-1] == "--verbose"


# ---------------------------------------------------------------------------
# parse_plain_text_to_response
# ---------------------------------------------------------------------------


def test_parse_plain_text_to_response_simple() -> None:
    resp = parse_plain_text_to_response("hello world\n", model="default")
    assert resp.text == "hello world"
    assert resp.stop_reason == "end_turn"
    assert resp.usage.input_tokens == 0
    assert resp.usage.output_tokens == 0


# ---------------------------------------------------------------------------
# End-to-end via fake gh
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_oneshot_ok() -> None:
    c = _client(text="Greetings!")
    resp = await c._send(_make_request())
    assert resp.text == "Greetings!"
    assert resp.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_send_oneshot_auth_failure() -> None:
    c = _client(scenario="auth_fail")
    with pytest.raises(APIError) as ei:
        await c._send(_make_request())
    assert ei.value.category is ErrorCategory.CLI_AUTH_FAILED


@pytest.mark.asyncio
async def test_send_oneshot_not_installed() -> None:
    c = _client(scenario="not_installed")
    with pytest.raises(APIError) as ei:
        await c._send(_make_request())
    assert ei.value.category is ErrorCategory.CLI_NOT_FOUND


@pytest.mark.asyncio
async def test_send_oneshot_permission_failure() -> None:
    c = _client(scenario="permission_fail")
    with pytest.raises(APIError) as ei:
        await c._send(_make_request())
    assert ei.value.category is ErrorCategory.CLI_PERMISSION_DENIED


@pytest.mark.asyncio
async def test_send_oneshot_crash() -> None:
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


@pytest.mark.asyncio
async def test_argv_carries_allow_tools_and_prompt() -> None:
    """fake_gh.echo_argv strips the leading 'copilot' subcommand before
    echoing; we only assert on the trailing flags."""
    c = _client(scenario="echo_argv", allow_tools=["shell(git)"])
    resp = await c._send(_make_request(system="rules", messages=[{"role": "user", "content": "task"}]))
    argv = json.loads(resp.text)
    assert "--allow-tool" in argv and "shell(git)" in argv
    assert "-p" in argv
    prompt = argv[argv.index("-p") + 1]
    assert "## System\nrules" in prompt
    assert "## User\ntask" in prompt


@pytest.mark.asyncio
async def test_missing_binary_raises_cli_not_found() -> None:
    c = CopilotCLIClient(gh_binary_path="/totally/missing/gh", timeout_s=2.0)
    with pytest.raises(APIError) as ei:
        await c._send(_make_request())
    assert ei.value.category is ErrorCategory.CLI_NOT_FOUND


# ---------------------------------------------------------------------------
# Streaming fallback (BaseClient default → one message_complete event)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_message_stream_falls_back_to_message_complete() -> None:
    c = _client(text="streamed")
    events = []
    async for evt in c.create_message_stream(
        model_config=ModelConfig(model="default"),
        messages=[{"role": "user", "content": "go"}],
    ):
        events.append(evt)
    # Default fallback emits a single message_complete with the response
    assert any(e.get("type") == "message_complete" for e in events)


# ---------------------------------------------------------------------------
# CredentialBundle mapping (already added in Phase B2 for the copilot branch)
# ---------------------------------------------------------------------------


def test_pipeline_credentials_kwargs_mapping() -> None:
    from geny_executor.core.pipeline import _creds_to_client_kwargs
    from geny_executor.llm_client.credentials import ProviderCredentials

    creds = ProviderCredentials(
        binary_path=FAKE_GH,
        extras={
            "allow_tools": ("shell(git)",),
            "cwd": "/tmp/wd",
            "extra_args": ("--verbose",),
            "timeout_s": 30.0,
        },
    )
    kwargs = _creds_to_client_kwargs("copilot_cli", creds)
    assert kwargs["gh_binary_path"] == FAKE_GH
    assert kwargs["allow_tools"] == ("shell(git)",)
    assert kwargs["cwd"] == "/tmp/wd"
    assert kwargs["timeout_s"] == 30.0
