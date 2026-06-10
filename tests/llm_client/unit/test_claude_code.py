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
    # The client holds its own api_key ("sk-fake" via _client) so
    # auth_mode='auto' resolves to the API-key path → ``--bare``. The
    # parent-process env must be irrelevant (the env sniff was deleted in
    # 2.2.0 — PR #868 history), so explicitly clear it here.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = _client(scenario="echo_argv")
    resp = await c._send(_make_request(model="opus", system="rule X"))
    import json

    argv = json.loads(resp.text)
    assert "--print" in argv
    assert "--bare" in argv
    assert "--model" in argv and "opus" in argv
    assert "--system-prompt" in argv and "rule X" in argv


@pytest.mark.asyncio
async def test_argv_oauth_auth_mode_strips_bare_despite_api_key(monkeypatch) -> None:
    """Explicit ``auth_mode='oauth'`` wins over a configured api_key —
    the host is declaring the subscription path, and ``--bare`` would
    crash it with 'Not logged in · Please run /login'."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-stray")  # must be ignored too
    c = _client(scenario="echo_argv", auth_mode="oauth")
    resp = await c._send(_make_request())
    import json

    argv = json.loads(resp.text)
    assert "--bare" not in argv


def test_send_oneshot_propagates_api_key_via_env() -> None:
    """The fake CLI doesn't inspect env, but ``_env_extras`` should still
    expose ANTHROPIC_API_KEY to the child."""
    c = _client(api_key="sk-special")
    extras = c._env_extras()
    assert extras["ANTHROPIC_API_KEY"] == "sk-special"


# ---------------------------------------------------------------------------
# stream_event wire form — end-to-end via the fake CLI (audit §2.3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_message_stream_stream_event_form() -> None:
    """End-to-end over the REAL 2.1.x wire shape (``stream_event`` lines
    + duplicate-bearing terminal envelope + result with top-level
    ``result`` text). This is the scenario whose absence let the v2.1.4
    incident ship through a fully green suite — the fake CLI only spoke
    the pre-2.1.x delta dialect (audit §2.3)."""
    c = _client(scenario="ok_stream_event", text="streaming is real")
    events = []
    async for evt in c.create_message_stream(
        model_config=ModelConfig(model="sonnet"),
        messages=[{"role": "user", "content": "go"}],
    ):
        events.append(evt)

    text_deltas = [e for e in events if e.get("type") == "text_delta"]
    # True per-token streaming: multiple chunks, reassembling exactly.
    assert len(text_deltas) >= 2
    assert "".join(d["text"] for d in text_deltas) == "streaming is real"

    completes = [e for e in events if e.get("type") == "message_complete"]
    assert len(completes) == 1
    resp = completes[0]["response"]
    # No duplication from the terminal assistant envelope.
    assert resp.text == "streaming is real"
    assert resp.stop_reason == "end_turn"
    assert resp.usage.cost_usd == pytest.approx(0.0123)
    # Clean wire: no telemetry keys; version handshake recorded.
    assert "unknown_line_count" not in resp.raw
    assert resp.raw["cli_version"].startswith("2.1.149-fake")


@pytest.mark.asyncio
async def test_send_oneshot_real_result_envelope() -> None:
    """Non-streaming against the REAL ``--output-format json`` envelope
    (top-level ``result`` string + ``total_cost_usd``; audit §3.4)."""
    c = _client(scenario="ok_result_envelope", text="안녕, 친구!")
    resp = await c._send(_make_request())
    assert resp.text == "안녕, 친구!"
    assert resp.stop_reason == "end_turn"
    assert resp.usage.cost_usd == pytest.approx(0.15079925)
    assert resp.usage.input_tokens == 2
    assert resp.message_id  # session id fallback


# ---------------------------------------------------------------------------
# Unknown-wire-shape telemetry + strict_wire (audit §2.2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_unknown_lines_tolerated_and_reported() -> None:
    """Default posture: tolerate the drifted lines, keep the answer, and
    TELL SOMEONE — counts in raw, one ``llm_client.unknown_wire_shape``
    event per call on the sink."""
    sink_events: list[dict] = []
    c = _client(
        scenario="stream_unknown_lines", text="drift",
        event_sink=sink_events.append,
    )
    events = []
    async for evt in c.create_message_stream(
        model_config=ModelConfig(model="sonnet"),
        messages=[{"role": "user", "content": "go"}],
    ):
        events.append(evt)

    resp = [e for e in events if e.get("type") == "message_complete"][0]["response"]
    assert resp.text == "drift"  # the call still succeeded
    assert resp.raw["unknown_line_count"] == 1
    assert resp.raw["malformed_line_count"] == 1
    assert resp.raw["first_unknown_type"] == "telepathy_event"

    wire_events = [
        e for e in sink_events if e["type"] == "llm_client.unknown_wire_shape"
    ]
    assert len(wire_events) == 1, "exactly one signal per call, not per line"
    assert wire_events[0]["provider"] == "claude_code_cli"
    assert wire_events[0]["unknown_type"] == "telepathy_event"
    assert wire_events[0]["count"] == 2  # 1 unknown + 1 malformed


@pytest.mark.asyncio
async def test_strict_wire_raises_on_unknown_lines() -> None:
    """CI-canary posture: the next wire drift becomes a failing call."""
    c = _client(scenario="stream_unknown_lines", strict_wire=True)
    with pytest.raises(APIError) as ei:
        async for _ in c.create_message_stream(
            model_config=ModelConfig(model="sonnet"),
            messages=[{"role": "user", "content": "go"}],
        ):
            pass
    assert ei.value.category is ErrorCategory.CLI_PROTOCOL_ERROR
    assert "telepathy_event" in str(ei.value)


@pytest.mark.asyncio
async def test_strict_wire_raises_on_send_streaming_path() -> None:
    """The assembler path (``_send(stream=True)``) enforces strict_wire
    too — both consumers of the stream share the telemetry contract."""
    c = _client(scenario="stream_unknown_lines", strict_wire=True)
    with pytest.raises(APIError) as ei:
        await c._send(_make_request(stream=True))
    assert ei.value.category is ErrorCategory.CLI_PROTOCOL_ERROR


@pytest.mark.asyncio
async def test_send_streaming_unknown_lines_tolerated_by_default() -> None:
    c = _client(scenario="stream_unknown_lines", text="drift")
    resp = await c._send(_make_request(stream=True))
    assert resp.text == "drift"
    assert resp.raw["unknown_line_count"] == 1


# ---------------------------------------------------------------------------
# CLI version handshake (audit Tier 2 — all four 2.1.x incidents were skew)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cli_version_attached_to_response_raw() -> None:
    c = _client(text="v")
    resp = await c._send(_make_request())
    assert resp.raw["cli_version"] == "2.1.149-fake (Claude Code)"
    assert c._cli_version_value == "2.1.149-fake (Claude Code)"


@pytest.mark.asyncio
async def test_cli_version_probed_once_per_instance() -> None:
    """The handshake is lazy and cached — one ``--version`` spawn per
    client instance, not per call."""
    from geny_executor.llm_client._cli_runtime import CLIProcessRunner

    made: list[dict] = []

    def factory(**kwargs):
        made.append(kwargs)
        return CLIProcessRunner(**kwargs)

    c = _client(text="x", runner_factory=factory)
    await c._send(_make_request())
    await c._send(_make_request())
    # First call: probe runner + call runner. Second call: call runner only.
    assert len(made) == 3


@pytest.mark.asyncio
async def test_cli_version_in_error_message_on_cli_failure() -> None:
    c = _client(scenario="crash")
    with pytest.raises(APIError) as ei:
        await c._send(_make_request())
    assert ei.value.category is ErrorCategory.CLI_PROTOCOL_ERROR
    assert "cli_version=2.1.149-fake" in str(ei.value)


@pytest.mark.asyncio
async def test_cli_version_probe_failure_never_fails_call() -> None:
    """A binary that can't answer ``--version`` caches 'unknown' and the
    real call proceeds (and fails on its own merits, version-stamped)."""
    c = ClaudeCodeCLIClient(
        binary_path="/bin/false",
        workspace_dir=os.getcwd(),
        api_key="sk-fake",
        timeout_s=10.0,
    )
    with pytest.raises(APIError) as ei:
        await c._send(_make_request())
    # The failure is the CLI's own non-zero exit — not the probe.
    assert ei.value.category is ErrorCategory.CLI_PROTOCOL_ERROR
    assert c._cli_version_value == "unknown"
    assert "cli_version=unknown" in str(ei.value)


# ---------------------------------------------------------------------------
# runner_factory seam (GAPT sandbox patch absorber, audit Tier 1-1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_factory_receives_spawn_parameters() -> None:
    """Hosts wrap process spawning (docker sandbox) via the factory —
    the supported replacement for monkey-patching
    ``CLIProcessRunner._spawn``, which pinned GAPT to 2.1.0."""
    from geny_executor.llm_client._cli_runtime import CLIProcessRunner

    made: list[dict] = []

    def factory(**kwargs):
        made.append(dict(kwargs))
        return CLIProcessRunner(**kwargs)

    c = _client(text="seam ok", runner_factory=factory, timeout_s=33.0)
    resp = await c._send(_make_request())
    assert resp.text == "seam ok"

    assert made, "factory was never consulted"
    for call in made:
        assert call["binary"] == FAKE_CLAUDE
        assert call["cwd"] == os.getcwd()
        assert call["env_extras"]["ANTHROPIC_API_KEY"] == "sk-fake"
        assert "timeout_s" in call
    # The version probe routes through the same factory (so the recorded
    # version matches the binary that actually runs) with a short cap,
    # while the real call keeps the client timeout.
    timeouts = sorted(c["timeout_s"] for c in made)
    assert timeouts[0] <= 10.0  # probe runner
    assert timeouts[-1] == 33.0  # call runner


# ---------------------------------------------------------------------------
# Session continuity — session_hint wiring (audit §3.5 decoy list)
# ---------------------------------------------------------------------------


def _hl_request(c: ClaudeCodeCLIClient, **request_kwargs):
    """Build a request through the same high-level path stages use."""
    base = dict(
        model_config=ModelConfig(model="sonnet"),
        messages=[{"role": "user", "content": "hi"}],
        system="",
        tools=None,
        tool_choice=None,
        stream=True,
    )
    base.update(request_kwargs)
    return c._build_request(**base)


def test_session_hint_constructor_threads_to_resume_argv() -> None:
    """``supports_session_continuity=True`` was a producer-less decoy:
    the argv builder knew ``--resume`` but the high-level surface could
    never carry a hint. The client-level default closes the loop."""
    c = _client(session_hint={"session_id": "sess-42", "resume": True})
    req = _hl_request(c)
    assert req.session_hint == {"session_id": "sess-42", "resume": True}
    argv = c._build_argv(req)
    assert "--resume" in argv and "sess-42" in argv


def test_session_hint_updatable_between_turns_via_configure() -> None:
    c = _client()
    assert _hl_request(c).session_hint is None  # no hint by default
    c.configure(session_hint={"session_id": "turn-2", "resume": True})
    argv = c._build_argv(_hl_request(c))
    assert "--resume" in argv and "turn-2" in argv


def test_session_hint_per_request_wins_over_client_default() -> None:
    from geny_executor.llm_client.translators._cli import claude_code_argv

    c = _client(session_hint={"session_id": "client-default", "resume": True})
    req = _hl_request(c)
    req.session_hint = {"session_id": "explicit", "resume": True}
    argv = c._build_argv(req)
    assert "explicit" in argv and "client-default" not in argv
    # Sanity: the builder itself honours the request hint.
    assert "--resume" in claude_code_argv(req)


# ---------------------------------------------------------------------------
# _classify_cli_result anchoring (MCP/tool noise must not read as auth)
# ---------------------------------------------------------------------------


def _cli_result(stderr: bytes, returncode: int = 1):
    from geny_executor.llm_client._cli_runtime import CLIResult

    return CLIResult(returncode=returncode, stdout=b"", stderr=stderr, duration_ms=5)


def test_classify_incidental_auth_noise_is_not_auth_failed() -> None:
    """Regression: the old heuristic matched bare 'auth'+'fail'
    substrings anywhere, so an MCP server named 'oauth-helper' failing
    to start was classified CLI_AUTH_FAILED (fatal, non-retryable)."""
    from geny_executor.llm_client.claude_code import _classify_cli_result

    err = _classify_cli_result(
        _cli_result(b"MCP server 'oauth-helper' failed to start: connection refused")
    )
    assert err.category is ErrorCategory.CLI_PROTOCOL_ERROR


@pytest.mark.parametrize(
    "stderr",
    [
        b"Error: not authenticated. Run `claude auth login`.",
        b"HTTP 401 Unauthorized",
        b"error=authentication_failed",
        b"Invalid API key provided",
    ],
)
def test_classify_real_auth_phrases_still_map(stderr: bytes) -> None:
    from geny_executor.llm_client.claude_code import _classify_cli_result

    err = _classify_cli_result(_cli_result(stderr))
    assert err.category is ErrorCategory.CLI_AUTH_FAILED


def test_classify_appends_cli_version_when_known() -> None:
    from geny_executor.llm_client.claude_code import _classify_cli_result

    err = _classify_cli_result(_cli_result(b"boom"), cli_version="2.1.149")
    assert "cli_version=2.1.149" in str(err)


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
