"""Claude Code driven as a token generator, not as a second agent.

The retired ``claude_code_cli`` client handed the whole loop to the CLI: it ran
its own Read/Write/Bash under its own permission model, and the pipeline sees
an announcement of what already happened. That makes Claude Code a different
agent rather than a model behind this harness, and it cannot share a
conversation with any other provider.

This client inverts it — tools off, one generation, text out — so the same
conversation can move between a Claude login, a second Claude login and a
ChatGPT plan without losing its tools, memory or permission policy.

The invariants that keep it honest:

 · The CLI is invoked with its tool palette empty and MCP locked out. If a
   native tool ever reaches the model anyway, the call aborts rather than
   letting the CLI act on the user's machine unpoliced.
 · Auth comes from the account, never from the ambient environment: an
   exported ANTHROPIC_API_KEY must not be able to choose which account pays.
 · A CLI too old for an optional flag is retried without it, not failed.

The fake `claude` here is a Python script that speaks the same stream-json
the real binary does.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any, Dict, List

import pytest

from geny_executor.core.config import ModelConfig
from geny_executor.core.errors import APIError, ErrorCategory
from geny_executor.llm_client.claude_code_tokens import ClaudeCodeTokenClient, child_env


def _fake_claude(tmp_path: Path, body: str, *, name: str = "claude") -> str:
    """A scripted `claude`: records argv + env, then emits stream-json."""
    script = tmp_path / name
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        f"RECORD = {str(tmp_path / 'invocation.json')!r}\n"
        "sys.stdin.read()\n"
        "with open(RECORD, 'w') as fh:\n"
        "    json.dump({'argv': sys.argv[1:], 'env': dict(os.environ), 'cwd': os.getcwd()}, fh)\n"
        "def emit(obj):\n"
        "    print(json.dumps(obj), flush=True)\n"
        f"{body}\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


_SAYS_HELLO = """
emit({'type': 'system', 'subtype': 'init', 'model': 'claude-sonnet-4-5', 'tools': []})
emit({'type': 'stream_event', 'event': {'type': 'content_block_delta',
      'delta': {'type': 'text_delta', 'text': 'Hello.'}}})
emit({'type': 'result', 'subtype': 'success', 'is_error': False, 'result': 'Hello.',
      'usage': {'input_tokens': 10, 'output_tokens': 2}, 'total_cost_usd': 0.001})
"""

_CALLS_A_TOOL = """
call = '<tool_call>\\n{"name": "Read", "arguments": {"file_path": "/a.txt"}}\\n</tool_call>'
emit({'type': 'stream_event', 'event': {'type': 'content_block_delta',
      'delta': {'type': 'text_delta', 'text': 'Reading it. '}}})
emit({'type': 'stream_event', 'event': {'type': 'content_block_delta',
      'delta': {'type': 'text_delta', 'text': call}}})
emit({'type': 'result', 'subtype': 'success', 'is_error': False, 'result': 'x', 'usage': {}})
"""

_USES_ITS_OWN_TOOL = """
emit({'type': 'stream_event', 'event': {'type': 'content_block_start',
      'content_block': {'type': 'tool_use', 'name': 'Bash'}}})
emit({'type': 'result', 'subtype': 'success', 'is_error': False, 'result': '', 'usage': {}})
"""

_READ_TOOL = {"name": "Read", "description": "read a file", "input_schema": {"type": "object"}}


async def _run(client: ClaudeCodeTokenClient, **kwargs: Any) -> List[Dict[str, Any]]:
    chunks = []
    async for chunk in client.create_message_stream(
        model_config=ModelConfig(model="claude-sonnet-4-5"),
        messages=kwargs.pop("messages", [{"role": "user", "content": "hi"}]),
        **kwargs,
    ):
        chunks.append(chunk)
    return chunks


def _invocation(tmp_path: Path) -> Dict[str, Any]:
    return json.loads((tmp_path / "invocation.json").read_text(encoding="utf-8"))


class TestTokenOnly:
    @pytest.mark.asyncio
    async def test_the_cli_is_invoked_with_no_tools_of_its_own(self, tmp_path: Path) -> None:
        binary = _fake_claude(tmp_path, _SAYS_HELLO)
        client = ClaudeCodeTokenClient(binary_path=binary, scratch_dir=str(tmp_path / "scratch"))
        await _run(client, tools=[_READ_TOOL])
        argv = _invocation(tmp_path)["argv"]
        assert argv[argv.index("--tools") + 1] == ""
        assert argv[argv.index("--max-turns") + 1] == "1"
        assert "--strict-mcp-config" in argv
        assert argv[argv.index("--mcp-config") + 1] == '{"mcpServers":{}}'

    @pytest.mark.asyncio
    async def test_text_streams_through(self, tmp_path: Path) -> None:
        client = ClaudeCodeTokenClient(binary_path=_fake_claude(tmp_path, _SAYS_HELLO),
                                       scratch_dir=str(tmp_path / "scratch"))
        chunks = await _run(client)
        assert "".join(c["text"] for c in chunks if c["type"] == "text_delta") == "Hello."
        usage = chunks[-1]["response"].usage
        assert (usage.input_tokens, usage.output_tokens) == (10, 2)

    @pytest.mark.asyncio
    async def test_a_written_tool_call_becomes_a_canonical_tool_use(self, tmp_path: Path) -> None:
        client = ClaudeCodeTokenClient(binary_path=_fake_claude(tmp_path, _CALLS_A_TOOL),
                                       scratch_dir=str(tmp_path / "scratch"))
        chunks = await _run(client, tools=[_READ_TOOL])
        response = chunks[-1]["response"]
        assert response.stop_reason == "tool_use"
        use = [b for b in response.content if b.type == "tool_use"][0]
        assert (use.tool_name, use.tool_input) == ("Read", {"file_path": "/a.txt"})
        # the prose before the call is the user's, the call itself is not
        visible = "".join(c["text"] for c in chunks if c["type"] == "text_delta")
        assert visible.strip() == "Reading it."

    @pytest.mark.asyncio
    async def test_the_tool_catalogue_goes_in_the_system_prompt_file(self, tmp_path: Path) -> None:
        binary = _fake_claude(tmp_path, _SAYS_HELLO)
        client = ClaudeCodeTokenClient(binary_path=binary, scratch_dir=str(tmp_path / "scratch"))
        await _run(client, system="You are Geny.", tools=[_READ_TOOL])
        argv = _invocation(tmp_path)["argv"]
        assert "--system-prompt-file" in argv
        # the file is deleted after the call; the CLI read it while it existed
        assert argv[argv.index("--system-prompt-file") + 1].endswith(".md")

    @pytest.mark.asyncio
    async def test_a_native_tool_reaching_the_model_aborts_the_call(self, tmp_path: Path) -> None:
        client = ClaudeCodeTokenClient(binary_path=_fake_claude(tmp_path, _USES_ITS_OWN_TOOL),
                                       scratch_dir=str(tmp_path / "scratch"))
        with pytest.raises(APIError) as caught:
            await _run(client, tools=[_READ_TOOL])
        assert "Bash" in str(caught.value)


class TestAccountIsolation:
    def test_a_login_account_gets_its_own_config_dir(self) -> None:
        env = child_env(auth_method="login", config_dir="/accounts/a/config",
                        oauth_token=None, api_key=None,
                        base={"ANTHROPIC_API_KEY": "leaked", "PATH": "/usr/bin"})
        assert env["CLAUDE_CONFIG_DIR"] == "/accounts/a/config"
        assert env["CLAUDE_SECURESTORAGE_CONFIG_DIR"] == "/accounts/a/config"
        assert "ANTHROPIC_API_KEY" not in env

    def test_an_exported_key_cannot_choose_the_account(self) -> None:
        env = child_env(auth_method="token", config_dir=None, oauth_token="oauth-token",
                        api_key=None, base={"ANTHROPIC_API_KEY": "someone elses key"})
        assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-token"
        assert "ANTHROPIC_API_KEY" not in env

    def test_system_auth_keeps_the_user_s_own_environment(self) -> None:
        """'system' means exactly what the user's own `claude` does — only
        the stamps a parent Claude Code session leaves behind are dropped."""
        env = child_env(auth_method="system", config_dir=None, oauth_token=None, api_key=None,
                        base={"ANTHROPIC_API_KEY": "mine", "CLAUDECODE": "1",
                              "CLAUDE_CODE_ENTRYPOINT": "cli"})
        assert env["ANTHROPIC_API_KEY"] == "mine"
        assert "CLAUDECODE" not in env and "CLAUDE_CODE_ENTRYPOINT" not in env

    def test_the_autoupdater_is_off(self) -> None:
        env = child_env(auth_method="system", config_dir=None, oauth_token=None, api_key=None, base={})
        assert env["DISABLE_AUTOUPDATER"] == "1"

    @pytest.mark.asyncio
    async def test_the_spawn_runs_in_the_account_scratch_not_the_host_cwd(self, tmp_path: Path) -> None:
        """Otherwise the CLI discovers whatever CLAUDE.md / .claude settings
        happen to sit next to the server process."""
        scratch = tmp_path / "scratch"
        client = ClaudeCodeTokenClient(binary_path=_fake_claude(tmp_path, _SAYS_HELLO),
                                       scratch_dir=str(scratch))
        await _run(client)
        assert Path(_invocation(tmp_path)["cwd"]).resolve() == scratch.resolve()


class TestOlderBinaries:
    @pytest.mark.asyncio
    async def test_an_unknown_flag_is_dropped_and_the_call_retried(self, tmp_path: Path) -> None:
        body = """
import os
STATE = os.path.join(os.path.dirname(RECORD), 'tries')
tries = 0
if os.path.exists(STATE):
    tries = int(open(STATE).read())
open(STATE, 'w').write(str(tries + 1))
if '--no-chrome' in sys.argv and tries == 0:
    sys.stderr.write("error: unknown option '--no-chrome'\\n")
    sys.exit(1)
emit({'type': 'stream_event', 'event': {'type': 'content_block_delta',
      'delta': {'type': 'text_delta', 'text': 'ok'}}})
emit({'type': 'result', 'subtype': 'success', 'is_error': False, 'result': 'ok', 'usage': {}})
"""
        client = ClaudeCodeTokenClient(binary_path=_fake_claude(tmp_path, body),
                                       scratch_dir=str(tmp_path / "scratch"))
        chunks = await _run(client)
        assert "".join(c["text"] for c in chunks if c["type"] == "text_delta") == "ok"
        assert "--no-chrome" not in _invocation(tmp_path)["argv"]


class TestFailures:
    @pytest.mark.asyncio
    async def test_a_missing_binary_is_cli_not_found(self, tmp_path: Path) -> None:
        client = ClaudeCodeTokenClient(binary_path=str(tmp_path / "nope"),
                                       scratch_dir=str(tmp_path / "scratch"))
        with pytest.raises(APIError) as caught:
            await _run(client)
        assert caught.value.category == ErrorCategory.CLI_NOT_FOUND

    @pytest.mark.asyncio
    async def test_a_usage_limit_is_rate_limited_so_the_router_moves_on(self, tmp_path: Path) -> None:
        body = """
emit({'type': 'result', 'subtype': 'error', 'is_error': True,
      'result': 'Claude usage limit reached. Your limit will reset at 5pm.', 'usage': {}})
"""
        client = ClaudeCodeTokenClient(binary_path=_fake_claude(tmp_path, body),
                                       scratch_dir=str(tmp_path / "scratch"),
                                       account_label="personal")
        with pytest.raises(APIError) as caught:
            await _run(client)
        assert caught.value.category == ErrorCategory.RATE_LIMITED
        assert "personal" in str(caught.value)

    @pytest.mark.asyncio
    async def test_a_logged_out_account_is_auth(self, tmp_path: Path) -> None:
        body = """
emit({'type': 'result', 'subtype': 'error', 'is_error': True,
      'result': 'Not logged in. Please run /login.', 'usage': {}})
"""
        client = ClaudeCodeTokenClient(binary_path=_fake_claude(tmp_path, body),
                                       scratch_dir=str(tmp_path / "scratch"))
        with pytest.raises(APIError) as caught:
            await _run(client)
        assert caught.value.category == ErrorCategory.AUTH
