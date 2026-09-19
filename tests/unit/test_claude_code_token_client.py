"""Claude Code driven as a token generator, not as a second agent.

The retired ``claude_code_cli`` client handed the whole loop to the CLI: it ran
its own Read/Write/Bash under its own permission model, and the pipeline saw
an announcement of what already happened. That makes Claude Code a different
agent rather than a model behind this harness, and it cannot share a
conversation with any other provider.

This client inverts it — tools off, one generation, text out — so the same
conversation can move between a Claude login, a second Claude login and a
ChatGPT plan without losing its tools, memory or permission policy. Since
2.69.0 it does that through the official Claude Agent SDK rather than
hand-built CLI flags.

The invariants that keep it honest:

 · The model is given NO built-in tools and no MCP. If a native tool ever
   reaches it anyway, the call aborts rather than letting the CLI act on the
   user's machine unpoliced.
 · Auth comes from the account, never from the ambient environment: an
   exported ANTHROPIC_API_KEY must not be able to choose which account pays.
 · Host settings are not read. A developer's own ~/.claude/settings.json is
   not part of an agent's turn.

These tests script the SDK's ``Transport`` rather than a fake `claude`
binary. That is deliberate: what this module owns is the mapping from SDK
messages to canonical chunks, and a fake binary would test the CLI's wire
protocol instead — the very thing we adopted the SDK to stop owning.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List

import pytest

from geny_executor.core.config import ModelConfig
from geny_executor.core.errors import APIError, ErrorCategory
from geny_executor.llm_client.claude_code_tokens import ClaudeCodeTokenClient, child_env


class ScriptedTransport:
    """An SDK transport that replays a fixed list of CLI messages.

    Implements just enough of the control protocol for ``Query.initialize``
    to complete, then yields the scripted messages.
    """

    def __init__(self, messages: List[Dict[str, Any]]) -> None:
        self._scripted = messages
        self._pending: List[Dict[str, Any]] = []
        self.written: List[Dict[str, Any]] = []
        self._ready = False

    async def connect(self) -> None:
        self._ready = True

    async def write(self, data: str) -> None:
        for line in data.splitlines():
            if not line.strip():
                continue
            msg = json.loads(line)
            self.written.append(msg)
            if msg.get("type") == "control_request":
                self._pending.append(
                    {
                        "type": "control_response",
                        "response": {
                            "subtype": "success",
                            "request_id": msg.get("request_id"),
                            "response": {"commands": [], "output_style": "default"},
                        },
                    }
                )

    async def read_messages(self) -> AsyncIterator[Dict[str, Any]]:
        # answer the handshake first, then play the script
        while self._pending:
            yield self._pending.pop(0)
        for msg in self._scripted:
            yield msg

    def is_ready(self) -> bool:
        return self._ready

    async def end_input(self) -> None:
        return None

    async def close(self) -> None:
        self._ready = False


def _init(tools: List[str] | None = None) -> Dict[str, Any]:
    return {
        "type": "system",
        "subtype": "init",
        "model": "claude-sonnet-4-5",
        "tools": tools or [],
        "apiKeySource": "none",
        "session_id": "s1",
    }


def _delta(text: str) -> Dict[str, Any]:
    return {
        "type": "stream_event",
        "uuid": "u1",
        "session_id": "s1",
        "event": {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": text},
        },
    }


def _result(**over: Any) -> Dict[str, Any]:
    base = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "duration_ms": 5,
        "duration_api_ms": 4,
        "num_turns": 1,
        "session_id": "s1",
        "total_cost_usd": 0.001,
        "usage": {"input_tokens": 10, "output_tokens": 2},
        "result": "done",
    }
    base.update(over)
    return base


_SAYS_HELLO = [_init(), _delta("Hello."), _result()]

_CALLS_A_TOOL = [
    _init(),
    _delta("Reading it. "),
    _delta('<tool_call>\n{"name": "Read", "arguments": {"file_path": "/a.txt"}}\n</tool_call>'),
    _result(),
]

_USES_ITS_OWN_TOOL = [
    _init(),
    {
        "type": "stream_event",
        "uuid": "u1",
        "session_id": "s1",
        "event": {
            "type": "content_block_start",
            "content_block": {"type": "tool_use", "name": "Bash"},
        },
    },
    _result(),
]

_READ_TOOL = {"name": "Read", "description": "read a file", "input_schema": {"type": "object"}}


def _client(messages: List[Dict[str, Any]], **kwargs: Any) -> ClaudeCodeTokenClient:
    holder: Dict[str, Any] = {}

    def factory() -> ScriptedTransport:
        holder["transport"] = ScriptedTransport(messages)
        return holder["transport"]

    client = ClaudeCodeTokenClient(transport_factory=factory, **kwargs)
    client._probe = holder  # type: ignore[attr-defined]
    return client


async def _run(client: ClaudeCodeTokenClient, **kwargs: Any) -> List[Dict[str, Any]]:
    chunks = []
    async for chunk in client.create_message_stream(
        model_config=ModelConfig(model="claude-sonnet-4-5"),
        messages=kwargs.pop("messages", [{"role": "user", "content": "hi"}]),
        **kwargs,
    ):
        chunks.append(chunk)
    return chunks


class TestTokenOnly:
    @pytest.mark.asyncio
    async def test_the_model_is_given_no_tools_of_its_own(self) -> None:
        """The reason this client exists. ``tools=[]`` is the supported form
        of the old ``--tools ""`` flag; ``strict_mcp_config`` keeps project
        and user MCP servers out; ``setting_sources=[]`` keeps the host's
        settings.json out."""
        from claude_agent_sdk import ClaudeAgentOptions

        seen: Dict[str, Any] = {}
        real = ClaudeAgentOptions.__init__

        def spy(self: Any, *a: Any, **kw: Any) -> None:
            seen.update(kw)
            real(self, *a, **kw)

        ClaudeAgentOptions.__init__ = spy  # type: ignore[method-assign]
        try:
            await _run(_client(_SAYS_HELLO))
        finally:
            ClaudeAgentOptions.__init__ = real  # type: ignore[method-assign]

        assert seen["tools"] == [], "the model was offered built-in tools"
        assert seen["max_turns"] == 1, "more than one turn is an agent loop"
        assert seen["strict_mcp_config"] is True
        assert seen["mcp_servers"] == {}
        assert seen["setting_sources"] == [], "host settings.json would leak in"

    @pytest.mark.asyncio
    async def test_text_streams_through(self) -> None:
        chunks = await _run(_client(_SAYS_HELLO))
        assert {"type": "text_delta", "text": "Hello."} in chunks
        final = chunks[-1]["response"]
        assert final.stop_reason == "end_turn"
        assert final.usage.input_tokens == 10
        assert final.usage.cost_usd == 0.001

    @pytest.mark.asyncio
    async def test_a_written_tool_call_becomes_a_canonical_tool_use(self) -> None:
        """The whole architecture in one assertion: the model WRITES a call,
        and this harness turns it into the same tool_use block every other
        provider produces — so Stage 10 dispatches it identically."""
        chunks = await _run(_client(_CALLS_A_TOOL), tools=[_READ_TOOL])
        final = chunks[-1]["response"]
        assert final.stop_reason == "tool_use"
        calls = [b for b in final.content if b.type == "tool_use"]
        assert [(c.tool_name, c.tool_input) for c in calls] == [
            ("Read", {"file_path": "/a.txt"})
        ]
        assert "Reading it." in "".join(
            c.get("text", "") for c in chunks if c.get("type") == "text_delta"
        )

    @pytest.mark.asyncio
    async def test_the_tool_catalogue_goes_in_the_system_prompt(self) -> None:
        from claude_agent_sdk import ClaudeAgentOptions

        seen: Dict[str, Any] = {}
        real = ClaudeAgentOptions.__init__

        def spy(self: Any, *a: Any, **kw: Any) -> None:
            seen.update(kw)
            real(self, *a, **kw)

        ClaudeAgentOptions.__init__ = spy  # type: ignore[method-assign]
        try:
            await _run(_client(_SAYS_HELLO), tools=[_READ_TOOL])
        finally:
            ClaudeAgentOptions.__init__ = real  # type: ignore[method-assign]
        assert "Read" in seen["system_prompt"]
        assert "tool_call" in seen["system_prompt"]

    @pytest.mark.asyncio
    async def test_a_native_tool_reaching_the_model_aborts_the_call(self) -> None:
        with pytest.raises(APIError) as caught:
            await _run(_client(_USES_ITS_OWN_TOOL))
        assert "its own tool" in str(caught.value)
        assert caught.value.category == ErrorCategory.BAD_REQUEST

    @pytest.mark.asyncio
    async def test_tools_advertised_at_init_abort_the_call(self) -> None:
        """If ``tools=[]`` ever stops holding, say so — do not let the model
        act through a tool this pipeline never granted."""
        with pytest.raises(APIError) as caught:
            await _run(_client([_init(tools=["Bash"]), _delta("hi"), _result()]))
        assert "its own tool" in str(caught.value)


class TestAccountIsolation:
    def test_a_login_account_gets_its_own_config_dir(self) -> None:
        env = child_env(
            auth_method="login",
            config_dir="/accounts/a/config",
            oauth_token=None,
            api_key=None,
            base={"PATH": "/usr/bin"},
        )
        assert env["CLAUDE_CONFIG_DIR"] == "/accounts/a/config"

    def test_an_exported_key_cannot_choose_the_account(self) -> None:
        env = child_env(
            auth_method="token",
            config_dir=None,
            oauth_token="oauth-token",
            api_key=None,
            base={"ANTHROPIC_API_KEY": "sk-someone-elses", "PATH": "/usr/bin"},
        )
        assert "ANTHROPIC_API_KEY" not in env
        assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-token"

    def test_system_auth_keeps_the_user_s_own_environment(self) -> None:
        env = child_env(
            auth_method="system",
            config_dir=None,
            oauth_token=None,
            api_key=None,
            base={"ANTHROPIC_API_KEY": "sk-mine", "PATH": "/usr/bin"},
        )
        assert env["ANTHROPIC_API_KEY"] == "sk-mine"

    def test_the_autoupdater_is_off(self) -> None:
        env = child_env(
            auth_method="system", config_dir=None, oauth_token=None, api_key=None, base={}
        )
        assert env["DISABLE_AUTOUPDATER"] == "1"

    @pytest.mark.asyncio
    async def test_the_call_runs_in_the_account_scratch_not_the_host_cwd(self, tmp_path) -> None:
        """An empty directory: no project CLAUDE.md, no .claude/ settings and
        no .mcp.json get discovered from wherever the sidecar happens to run."""
        from claude_agent_sdk import ClaudeAgentOptions

        scratch = tmp_path / "scratch"
        seen: Dict[str, Any] = {}
        real = ClaudeAgentOptions.__init__

        def spy(self: Any, *a: Any, **kw: Any) -> None:
            seen.update(kw)
            real(self, *a, **kw)

        ClaudeAgentOptions.__init__ = spy  # type: ignore[method-assign]
        try:
            await _run(_client(_SAYS_HELLO, scratch_dir=str(scratch)))
        finally:
            ClaudeAgentOptions.__init__ = real  # type: ignore[method-assign]
        assert seen["cwd"] == str(scratch)
        assert list(scratch.iterdir()) == []


class TestFailures:
    @pytest.mark.asyncio
    async def test_a_usage_limit_is_rate_limited_so_the_router_moves_on(self) -> None:
        """The category is load-bearing: the router fails over from a rate
        limit and not from a bad request."""
        script = [_init(), _result(is_error=True, subtype="error",
                                   result="Claude usage limit reached.")]
        with pytest.raises(APIError) as caught:
            await _run(_client(script))
        assert caught.value.category == ErrorCategory.RATE_LIMITED

    @pytest.mark.asyncio
    async def test_a_logged_out_account_is_auth(self) -> None:
        script = [_init(), _result(is_error=True, subtype="error",
                                   result="Not logged in. Please run /login.")]
        with pytest.raises(APIError) as caught:
            await _run(_client(script))
        assert caught.value.category in (
            ErrorCategory.AUTH,
            ErrorCategory.CLI_AUTH_FAILED,
        )

    @pytest.mark.asyncio
    async def test_a_missing_binary_is_cli_not_found(self) -> None:
        """No transport injected → the SDK really looks for the binary."""
        client = ClaudeCodeTokenClient(binary_path="/nonexistent/claude")
        with pytest.raises(APIError) as caught:
            await _run(client)
        assert caught.value.category == ErrorCategory.CLI_NOT_FOUND
