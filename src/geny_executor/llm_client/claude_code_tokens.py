"""Claude Code as a *token generator* — the harness owns the loop.

Claude Code ships its own agentic loop: run it the usual way and the
CLI runs its own Read/Write/Bash, keeps its own permission model, and
these 21 stages see only an announcement of what already happened. That
makes Claude Code a different agent rather than a model behind this
pipeline, and it cannot share a conversation with any other provider.
Until 2.68.0 a ``claude_code_cli`` provider did exactly that; it is gone.

This client inverts it:

    claude -p --tools ""            no built-in tools at all
           --strict-mcp-config --mcp-config {"mcpServers":{}}
           --max-turns 1            one generation, never an agent loop
           --system-prompt-file     Geny's system prompt + tool protocol
           --output-format stream-json --include-partial-messages

The CLI authenticates exactly as it does for the user (their login, a
setup-token, or a Console key), streams tokens, and exits. Tool calls come
back as `<tool_call>` text (see tool_protocol.py), become canonical
`tool_use` blocks, and Stage 10 executes them inside the workspace jail with
the pipeline's permission policy and hooks — the same path as every API provider.

Multiple accounts are separate `CLAUDE_CONFIG_DIR`s (the CLI keys both its
credential file and, on macOS, its Keychain item on that directory), so any
number of Claude logins coexist without touching the user's own ~/.claude.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from geny_executor.core.errors import APIError, ErrorCategory
from geny_executor.core.state import TokenUsage
from geny_executor.llm_client.base import BaseClient, ClientCapabilities
from geny_executor.llm_client.types import APIRequest, APIResponse, ContentBlock

from geny_executor.llm_client import text_tool_protocol as tp
from geny_executor.llm_client._failover import Notify, classify_text

logger = logging.getLogger(__name__)

#: auth channels the CLI honours from the environment — stripped from the
#: inherited env so an unrelated exported key can never pick the account
AUTH_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS",
    "AWS_BEARER_TOKEN_BEDROCK",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CONFIG_DIR",
    "CLAUDE_SECURESTORAGE_CONFIG_DIR",
    # session stamps a parent Claude Code session leaves behind
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_CHILD_SESSION",
    "CLAUDE_CODE_BRIDGE_SESSION_ID",
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
)
SESSION_ENV = AUTH_ENV[-5:]


#: flags a given binary rejected with "unknown option" — learned at runtime,
#: because `--help` hides some accepted flags (`--system-prompt-file`,
#: `--max-turns`) and so cannot be the source of truth
# ── stdout stream limit ────────────────────────────────────────────────
# The CLI emits one stream-json event per line, and the model's own text
# rides INSIDE those lines — a long answer, a base64 image, or a big
# forged-tool payload easily exceeds asyncio's default StreamReader limit
# (64 KiB), and readline() then kills the whole turn with "Separator is
# found, but chunk is longer than limit" (the buffer is discarded, so the
# line is unrecoverable — the 2026-07-14 delegated-PPTX failure). The
# default is deliberately generous: 32 MiB, a cap rather than an
# allocation (memory is used only per actual line).
def _stream_limit() -> int:
    raw = os.environ.get("GENY_CLI_STREAM_LIMIT", "").strip()
    try:
        v = int(raw) if raw else 0
    except ValueError:
        v = 0
    return v if v >= 2**16 else 32 * 1024 * 1024


_UNSUPPORTED: dict[str, set[str]] = {}
_UNKNOWN_OPTION = re.compile(r"unknown option '?(--[a-zA-Z-]+)'?", re.I)


def child_env(
    *,
    auth_method: str,
    config_dir: Optional[str],
    oauth_token: Optional[str],
    api_key: Optional[str],
    base: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    parent = dict(base if base is not None else os.environ)
    # "system" means "exactly what the user's own claude does" — keep their
    # auth env (a gateway base URL, a key they exported) and drop only the
    # stamps a parent Claude Code session leaves behind
    strip = SESSION_ENV if auth_method == "system" else AUTH_ENV
    upper = {k.upper() for k in strip}
    env = {k: v for k, v in parent.items() if k.upper() not in upper}
    if auth_method == "login" and config_dir:
        env["CLAUDE_CONFIG_DIR"] = config_dir
        # Claude Code 2.1.220+ derives its Keychain service name from this
        env["CLAUDE_SECURESTORAGE_CONFIG_DIR"] = config_dir
    elif auth_method == "token" and oauth_token:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token
        if config_dir:
            env["CLAUDE_CONFIG_DIR"] = config_dir
    elif auth_method == "api_key" and api_key:
        env["ANTHROPIC_API_KEY"] = api_key
        if config_dir:
            env["CLAUDE_CONFIG_DIR"] = config_dir
    # 'system' = whatever the user's own `claude` is logged into
    env["DISABLE_AUTOUPDATER"] = "1"
    env["CLAUDE_CODE_SKIP_PROMPT_HISTORY"] = "1"
    env.setdefault("CI", "1")
    return env


def _windows_shim_target(binary: str) -> Optional[List[str]]:
    """npm's `claude.cmd` → `node <…>/@anthropic-ai/claude-code/cli.js`.

    Going through cmd.exe corrupts arguments (it does not honour the `\\"`
    escapes subprocess produces, and treats `<`, `>` and newlines as syntax)
    and `terminate()` would only kill the shell, leaving the CLI generating.
    Running the script directly avoids both."""
    base = Path(binary).parent
    for rel in (
        Path("node_modules") / "@anthropic-ai" / "claude-code" / "cli.js",
        Path("..") / "node_modules" / "@anthropic-ai" / "claude-code" / "cli.js",
    ):
        script = (base / rel).resolve()
        if script.exists():
            local_node = base / "node.exe"
            node = str(local_node) if local_node.exists() else (shutil.which("node") or "node")
            return [node, str(script)]
    return None


def command_for(binary: str, args: List[str]) -> List[str]:
    if sys.platform == "win32" and binary.lower().endswith((".cmd", ".bat")):
        target = _windows_shim_target(binary)
        if target:
            return [*target, *args]
        return [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/s", "/c", binary, *args]
    return [binary, *args]


async def kill_tree(proc: Any) -> None:
    """Stop the CLI and anything it started (on Windows terminate() does not
    reach grandchildren)."""
    if proc.returncode is not None:
        return
    if sys.platform == "win32":
        with contextlib.suppress(Exception):
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/PID",
                str(proc.pid),
                "/T",
                "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(killer.wait(), timeout=5)
    else:
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(proc.wait(), timeout=5)
    if proc.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()


_EFFORTS = ("low", "medium", "high", "xhigh", "max")


def effort_for(request: APIRequest, fixed: Optional[str]) -> Optional[str]:
    if fixed in _EFFORTS:
        return fixed
    thinking = request.thinking or {}
    if not thinking or str(thinking.get("type", "")).lower() in ("", "disabled", "off"):
        return None
    budget = int(thinking.get("budget_tokens", 0) or 0)
    if budget and budget <= 5_000:
        return "low"
    if budget and budget <= 15_000:
        return "medium"
    return "high"


class _FlagRejected(Exception):
    def __init__(self, flag: str) -> None:
        super().__init__(flag)
        self.flag = flag


class ClaudeCodeTokenClient(BaseClient):
    """`claude -p` with tools off: text in, text (+ parsed tool calls) out."""

    provider = "geny_claude_code"
    capabilities = ClientCapabilities(
        supports_thinking=True,
        supports_tools=True,
        supports_streaming=True,
        supports_tool_choice=False,
        supports_stop_sequences=False,
        supports_top_k=False,
        supports_system_prompt=True,
        supports_structured_output=False,
        supports_token_usage=True,
        supports_cost_usage=True,
        requires_workspace=False,
        streaming_granularity="token",
        # ``claude -p`` takes none of these, and an undeclared drop is a
        # silent one — the whole point of the declaration is that the
        # host sees an ``llm_client.field_dropped`` event instead of a
        # setting that quietly does nothing. Parity with geny_codex,
        # which drops the same list.
        drops=(
            "temperature",
            "top_p",
            "top_k",
            "stop_sequences",
            "max_tokens",
            "tool_choice",
        ),
    )

    def __init__(
        self,
        api_key: str = "",
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        *,
        binary_path: Optional[str] = None,
        account_id: str = "",
        account_label: str = "",
        auth_method: str = "system",
        config_dir: Optional[str] = None,
        oauth_token: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        scratch_dir: Optional[str] = None,
        effort: Optional[str] = None,
        timeout_s: float = 600.0,
        notify: Optional[Notify] = None,
        event_sink: Any = None,
        **_ignored: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
            event_sink=event_sink,
        )
        self._binary = binary_path or shutil.which("claude") or "claude"
        self._account_id = account_id
        self._account_label = account_label
        self._auth_method = auth_method
        self._config_dir = config_dir
        self._oauth_token = oauth_token or (api_key if auth_method == "token" else None)
        self._anthropic_key = anthropic_api_key or (api_key if auth_method == "api_key" else None)
        self._scratch = scratch_dir
        self._effort = effort
        self._timeout_s = float(timeout_s or 600.0)
        self._notify = notify

    # ── plumbing ─────────────────────────────────────────────────────
    def _env(self) -> dict[str, str]:
        if self._auth_method == "login" and self._config_dir:
            Path(self._config_dir).mkdir(parents=True, exist_ok=True)
        return child_env(
            auth_method=self._auth_method,
            config_dir=self._config_dir,
            oauth_token=self._oauth_token,
            api_key=self._anthropic_key,
        )

    def _cwd(self) -> str:
        # an empty directory: no project CLAUDE.md, no .claude/ settings or
        # MCP config gets discovered from wherever the sidecar happens to run
        root = (
            Path(self._scratch)
            if self._scratch
            else Path(tempfile.gettempdir()) / "geny-claude-scratch"
        )
        root.mkdir(parents=True, exist_ok=True)
        return str(root)

    def _argv(self, request: APIRequest, prompt_file: str) -> List[str]:
        """Everything that makes the call token-only. Optional flags a given
        CLI turned out not to know are dropped (see `_UNSUPPORTED`)."""
        skip = _UNSUPPORTED.get(self._binary, set())
        argv = [
            "-p",
            "--verbose",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
        ]

        def opt(flag: str, *values: str) -> None:
            if flag not in skip:
                argv.extend([flag, *values])

        opt("--include-partial-messages")
        if request.model:
            argv += ["--model", str(request.model)]
        argv += ["--system-prompt-file", prompt_file]
        # no built-in tools — the reason this client exists. Every variadic
        # flag is followed by another option, never by a bare value.
        if "--tools" not in skip:
            argv += ["--tools", ""]
        else:
            argv += ["--disallowedTools", "*"]
        opt("--strict-mcp-config")
        opt("--mcp-config", '{"mcpServers":{}}')
        # the browser integration injects its MCP tools even with an empty
        # --mcp-config (observed on 2.1.274)
        opt("--no-chrome")
        argv += ["--max-turns", "1"]
        opt("--no-session-persistence")
        opt("--disable-slash-commands")
        opt("--safe-mode")
        effort = effort_for(request, self._effort)
        if effort:
            opt("--effort", effort)
        return argv

    def _notify_host(self, payload: dict[str, Any]) -> None:
        if self._notify is None:
            return
        try:
            self._notify({"accountId": self._account_id, **payload})
        except Exception:
            pass

    # ── BaseClient ───────────────────────────────────────────────────
    async def _send(self, request: APIRequest, *, purpose: str = "") -> APIResponse:
        response: Optional[APIResponse] = None
        async for chunk in self._stream(request):
            if chunk.get("type") == "message_complete":
                response = chunk["response"]
        if response is None:
            raise APIError("claude produced no result", category=ErrorCategory.NETWORK)
        return response

    async def create_message_stream(
        self,
        *,
        model_config: Any,
        messages: List[Dict[str, Any]],
        system: Any = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        purpose: str = "",
    ) -> AsyncIterator[Dict[str, Any]]:
        request = self._build_request(
            model_config=model_config,
            messages=messages,
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            stream=True,
        )
        async for chunk in self._stream(request):
            yield chunk

    async def _stream(self, request: APIRequest) -> AsyncIterator[Dict[str, Any]]:
        env = self._env()
        cwd = self._cwd()
        system_prompt = tp.render_system_prompt(request.system, request.tools)
        fd, prompt_file = tempfile.mkstemp(prefix="geny-sys-", suffix=".md", dir=cwd)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(system_prompt or "You are a helpful assistant.")
        try:
            envelope = {
                "type": "user",
                "message": {"role": "user", "content": tp.render_transcript(request.messages)},
            }
            for _attempt in range(8):
                argv = self._argv(request, prompt_file)
                try:
                    async for chunk in self._run(
                        argv, env, cwd, envelope, tp.tool_names(request.tools), bool(request.tools)
                    ):
                        yield chunk
                    return
                except _FlagRejected as rejected:
                    # an older CLI: forget that flag for this binary, retry
                    _UNSUPPORTED.setdefault(self._binary, set()).add(rejected.flag)
            raise APIError(
                "claude rejected too many flags — update Claude Code",
                category=ErrorCategory.BAD_REQUEST,
            )
        finally:
            with contextlib.suppress(OSError):
                os.unlink(prompt_file)

    async def _run(
        self,
        argv: List[str],
        env: dict[str, str],
        cwd: str,
        envelope: dict[str, Any],
        known: set[str],
        tools_enabled: bool,
    ) -> AsyncIterator[Dict[str, Any]]:
        started = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *command_for(self._binary, argv),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
                limit=_stream_limit(),
            )
        except FileNotFoundError as exc:
            raise APIError(
                f"claude CLI not found ({self._binary}) — install Claude Code and retry.",
                category=ErrorCategory.CLI_NOT_FOUND,
                cause=exc,
            ) from exc
        except OSError as exc:  # E2BIG, EACCES, …
            raise APIError(
                f"claude failed to start: {exc}", category=ErrorCategory.CLI_NOT_FOUND, cause=exc
            ) from exc

        assert proc.stdin and proc.stdout and proc.stderr
        stderr_chunks: list[bytes] = []

        async def drain_stderr() -> None:
            assert proc.stderr
            while True:
                data = await proc.stderr.read(4096)
                if not data:
                    return
                if sum(len(c) for c in stderr_chunks) < 64_000:
                    stderr_chunks.append(data)

        stderr_task = asyncio.create_task(drain_stderr())
        try:
            proc.stdin.write((json.dumps(envelope, ensure_ascii=False) + "\n").encode("utf-8"))
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        with contextlib.suppress(Exception):
            proc.stdin.close()

        splitter = tp.StreamSplitter(known=known or None)
        streamed_text = False
        thinking_parts: list[str] = []
        final_text: Optional[str] = None
        result: Optional[dict[str, Any]] = None
        error_text: Optional[str] = None
        stopped_early = False
        native_tool: Optional[str] = None
        model_used = ""

        try:
            while True:
                remaining = self._timeout_s - (time.monotonic() - started)
                if remaining <= 0:
                    raise APIError(
                        f"claude timed out after {self._timeout_s:.0f}s",
                        category=ErrorCategory.CLI_TIMEOUT,
                    )
                try:
                    line = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
                except asyncio.TimeoutError as exc:
                    raise APIError(
                        f"claude timed out after {self._timeout_s:.0f}s",
                        category=ErrorCategory.CLI_TIMEOUT,
                    ) from exc
                except ValueError as exc:
                    # asyncio raises ValueError("Separator is found, but chunk
                    # is longer than limit") when ONE line exceeds the reader
                    # limit — and discards the buffered bytes, so that line is
                    # unrecoverable. With the 32 MiB default this is near
                    # impossible; if it happens anyway, losing ONE event beats
                    # killing the turn. Log loudly and keep reading.
                    logger.warning(
                        "claude stdout line exceeded the %d-byte limit — "
                        "skipping one event and continuing (%s)",
                        _stream_limit(),
                        exc,
                    )
                    continue
                if not line:
                    break
                try:
                    msg = json.loads(line.decode("utf-8", "replace"))
                except json.JSONDecodeError:
                    continue
                if not isinstance(msg, dict):
                    continue
                kind = msg.get("type")

                if kind == "stream_event":
                    event = msg.get("event") or {}
                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta") or {}
                        if delta.get("type") == "text_delta" and delta.get("text"):
                            streamed_text = True
                            visible = (
                                splitter.feed(delta["text"]) if tools_enabled else delta["text"]
                            )
                            if not tools_enabled:
                                splitter.visible.append(visible)
                            if visible:
                                yield {"type": "text_delta", "text": visible}
                            if splitter.finished:
                                stopped_early = True
                                break
                        elif delta.get("type") == "thinking_delta" and delta.get("thinking"):
                            thinking_parts.append(delta["thinking"])
                            yield {"type": "thinking_delta", "text": delta["thinking"]}
                    elif event.get("type") == "content_block_start" and (
                        (event.get("content_block") or {}).get("type")
                        in ("tool_use", "server_tool_use")
                    ):
                        # a native tool reached the model despite --tools "":
                        # stop before the CLI can execute anything itself
                        native_tool = str((event.get("content_block") or {}).get("name") or "?")
                        stopped_early = True
                        break
                    elif event.get("type") == "message_start":
                        model_used = str(((event.get("message") or {}).get("model")) or model_used)
                elif kind == "assistant":
                    content = ((msg.get("message") or {}).get("content")) or []
                    text = "".join(
                        str(b.get("text") or "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                    if text:
                        final_text = (final_text or "") + text
                    model_used = str((msg.get("message") or {}).get("model") or model_used)
                elif kind == "result":
                    result = msg
                    if msg.get("is_error") or (
                        msg.get("subtype") not in (None, "success")
                        and msg.get("subtype") != "error_max_turns"
                    ):
                        error_text = str(
                            msg.get("result") or msg.get("error") or msg.get("subtype")
                        )
                    break
                elif kind == "rate_limit_event":
                    self._notify_host(
                        {"kind": "rate_limit", "info": msg.get("rate_limit_info") or msg}
                    )
                elif kind == "system" and msg.get("subtype") == "init":
                    model_used = str(msg.get("model") or model_used)
                    exposed = [t for t in (msg.get("tools") or []) if isinstance(t, str)]
                    if exposed:
                        self._notify_host(
                            {
                                "kind": "notice",
                                "level": "warn",
                                "message": f"Claude Code exposed {len(exposed)} built-in tool(s) "
                                f"({', '.join(exposed[:4])}…) — the call aborts if one is used.",
                            }
                        )
                    self._notify_host(
                        {
                            "kind": "init",
                            "model": msg.get("model"),
                            "apiKeySource": msg.get("apiKeySource"),
                        }
                    )
                elif kind == "auth_status" and msg.get("error"):
                    error_text = str(msg.get("error"))
        finally:
            if proc.returncode is None and not (stopped_early or result is not None):
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(proc.wait(), timeout=5)
            if proc.returncode is None:
                await kill_tree(proc)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(stderr_task, timeout=2)
            if not stderr_task.done():
                stderr_task.cancel()

        stderr = b"".join(stderr_chunks).decode("utf-8", "replace").strip()

        if native_tool is not None:
            raise APIError(
                f"Claude Code tried to use its own tool ({native_tool}) — token-only isolation "
                "is broken. Update Claude Code, or run this account in CLI-agent mode.",
                category=ErrorCategory.BAD_REQUEST,
            )
        rejected = _UNKNOWN_OPTION.search(stderr) or _UNKNOWN_OPTION.search(error_text or "")
        if rejected and result is None and not streamed_text:
            raise _FlagRejected(rejected.group(1))

        if error_text is None and result is None and not stopped_early:
            error_text = stderr or f"claude exited with code {proc.returncode} without a result"

        if error_text is not None:
            category = classify_text(error_text)
            if category == ErrorCategory.UNKNOWN:
                # a result the CLI reported (prompt too long, bad model…) is
                # not fixed by retrying; only a missing result is transient
                category = (
                    ErrorCategory.BAD_REQUEST
                    if result is not None
                    else ErrorCategory.CLI_PROTOCOL_ERROR
                )
            label = f" [{self._account_label}]" if self._account_label else ""
            raise APIError(f"Claude Code{label}: {error_text[:600]}", category=category)

        # the partial stream is the source of truth when it existed; the
        # assistant envelope covers CLIs without --include-partial-messages
        if not streamed_text and final_text:
            if tools_enabled:
                visible = splitter.feed(final_text)
            else:
                visible = final_text
                splitter.visible.append(final_text)
            if visible:
                yield {"type": "text_delta", "text": visible}
        tail = splitter.close() if tools_enabled else ""
        if tail:
            yield {"type": "text_delta", "text": tail}

        blocks: list[ContentBlock] = []
        if thinking_parts:
            # no signature exists for CLI thinking, so it is surfaced to the
            # UI above but never replayed into history
            pass
        text = splitter.text
        if text:
            blocks.append(ContentBlock(type="text", text=text, raw={"type": "text", "text": text}))
        for call in splitter.calls:
            blocks.append(
                ContentBlock(
                    type="tool_use",
                    tool_use_id=call.id,
                    tool_name=call.name,
                    tool_input=call.arguments,
                    raw={
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    },
                )
            )
        if not blocks:
            blocks.append(ContentBlock(type="text", text="", raw={"type": "text", "text": ""}))

        usage_raw = (result or {}).get("usage") or {}
        usage = TokenUsage(
            input_tokens=int(usage_raw.get("input_tokens") or 0),
            output_tokens=int(usage_raw.get("output_tokens") or 0),
            cache_creation_input_tokens=int(usage_raw.get("cache_creation_input_tokens") or 0),
            cache_read_input_tokens=int(usage_raw.get("cache_read_input_tokens") or 0),
            cost_usd=(result or {}).get("total_cost_usd"),
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        if splitter.malformed and not splitter.calls:
            self._notify_host(
                {
                    "kind": "notice",
                    "level": "warn",
                    "message": "The model emitted a malformed tool call — it was ignored.",
                }
            )
        yield {
            "type": "message_complete",
            "response": APIResponse(
                content=blocks,
                stop_reason="tool_use" if splitter.calls else "end_turn",
                usage=usage,
                model=model_used,
                message_id=str((result or {}).get("session_id") or ""),
                raw={"provider": self.provider, "account": self._account_id},
            ),
        }
