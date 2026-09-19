"""Claude Code as a *token generator* — the harness owns the loop.

Claude Code ships its own agentic loop: run it the usual way and the
CLI runs its own Read/Write/Bash, keeps its own permission model, and
these 21 stages see only an announcement of what already happened. That
makes Claude Code a different agent rather than a model behind this
pipeline, and it cannot share a conversation with any other provider.
Until 2.68.0 a ``claude_code_cli`` provider did exactly that; it is gone.

This client inverts it, through the official **Claude Agent SDK**
(``claude-agent-sdk``)::

    ClaudeAgentOptions(
        tools=[],              # no built-in tools at all
        max_turns=1,           # one generation, never an agent loop
        strict_mcp_config=True, mcp_servers={},
        setting_sources=[],    # and no host settings.json either
        system_prompt=...,     # Geny's prompt + the tool protocol
        include_partial_messages=True,
    )

It used to hand-build that as fourteen ``claude -p`` flags and parse the
stream-json by hand, which meant owning an interface nobody promised us:
the module carried a table of flags it had *learned at runtime* a given
binary rejected, by regexing "unknown option" out of stderr and retrying.
Every field above is the supported spelling of one of those flags, so
that table is gone and so is the argv builder.

The CLI still authenticates exactly as it does for the user — their
login, a setup-token, or a Console key — because the SDK drives that same
CLI. Tool calls come back as `<tool_call>` text (see text_tool_protocol),
become canonical `tool_use` blocks, and Stage 10 executes them inside the
workspace jail with the pipeline's permission policy and hooks — the same
path as every API provider.

Multiple accounts are separate `CLAUDE_CONFIG_DIR`s (the CLI keys both its
credential file and, on macOS, its Keychain item on that directory), so any
number of Claude logins coexist without touching the user's own ~/.claude.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional

from geny_executor.core.errors import APIError, ErrorCategory
from geny_executor.core.state import TokenUsage
from geny_executor.llm_client.base import BaseClient, ClientCapabilities
from geny_executor.llm_client.types import APIRequest, APIResponse, ContentBlock

from geny_executor.llm_client import text_tool_protocol as tp
from geny_executor.llm_client._failover import Notify, classify_text

if TYPE_CHECKING:  # the SDK is a runtime dep; this keeps import cost off the hot path
    from claude_agent_sdk.types import EffortLevel

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


_EFFORTS = ("low", "medium", "high", "xhigh", "max")


def _effort_literal(effort: Optional[str]) -> Optional[EffortLevel]:
    """``effort_for`` answers a plain string; the SDK's field is a Literal.

    Narrow rather than cast: an effort the SDK does not know is dropped,
    not smuggled past the type checker into a rejected request.
    """
    return effort if effort in _EFFORTS else None  # type: ignore[return-value]


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


class ClaudeCodeTokenClient(BaseClient):
    """`claude -p` with tools off: text in, text (+ parsed tool calls) out."""

    provider = "geny_claude_code"
    capabilities = ClientCapabilities(
        supports_thinking=True,
        supports_tools=True,
        supports_vision=True,
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
        transport_factory: Any = None,
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
        # Tests inject an SDK ``Transport`` here instead of a fake binary, so
        # they exercise THIS module's message→chunk mapping rather than the
        # CLI's wire protocol. ``None`` (production) lets the SDK spawn the
        # real ``claude``.
        self._transport_factory = transport_factory

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

    # ── the one call ─────────────────────────────────────────────────
    async def _stream(self, request: APIRequest) -> AsyncIterator[Dict[str, Any]]:
        """One generation through the Agent SDK, yielding canonical chunks.

        ``tools=[]`` is the whole point: the model gets NONE of Claude
        Code's built-ins, so it cannot act — it can only write. Our tool
        catalogue rides in the system prompt as ``<tool_call>`` text,
        comes back as text, and Stage 10 executes it under this
        pipeline's jail, permissions and hooks.
        """
        from claude_agent_sdk import ClaudeAgentOptions, query
        from claude_agent_sdk.types import (
            AssistantMessage,
            ResultMessage,
            StreamEvent,
            SystemMessage,
            TextBlock,
            ThinkingBlock,
            ToolUseBlock,
        )

        started = time.monotonic()
        tools_enabled = bool(request.tools)
        splitter = tp.StreamSplitter(known=tp.tool_names(request.tools) or None)

        envelope = {
            "type": "user",
            "message": {"role": "user", "content": tp.render_transcript(request.messages)},
            "parent_tool_use_id": None,
            "session_id": "default",
        }

        async def _prompt() -> AsyncIterator[dict[str, Any]]:
            yield envelope

        env = self._env()
        # The SDK's own request timeout, so a stalled vendor call ends
        # inside the CLI rather than leaving us to kill a process group.
        env.setdefault("API_TIMEOUT_MS", str(int(self._timeout_s * 1000)))

        stderr_lines: list[str] = []
        options = ClaudeAgentOptions(
            # No built-in tools at all — the supported form of the
            # `--tools ""` flag this client used to hand-build.
            tools=[],
            max_turns=1,
            model=str(request.model) if request.model else None,
            system_prompt=tp.render_system_prompt(request.system, request.tools)
            or "You are a helpful assistant.",
            # Nothing but what we pass: no project .mcp.json, no user or
            # plugin servers.
            strict_mcp_config=True,
            mcp_servers={},
            # …and no settings.json either. The hand-spawned CLI had no
            # equivalent, so the HOST's ~/.claude/settings.json was being
            # read into every agent turn.
            setting_sources=[],
            permission_mode="dontAsk",
            include_partial_messages=True,
            cwd=self._cwd(),
            env=env,
            cli_path=self._binary,
            max_buffer_size=_stream_limit(),
            effort=_effort_literal(effort_for(request, self._effort)),
            stderr=stderr_lines.append,
        )

        streamed_text = False
        thinking_parts: list[str] = []
        final_text: Optional[str] = None
        native_tool: Optional[str] = None
        model_used = str(request.model or "")
        result: Optional[ResultMessage] = None
        error_text: Optional[str] = None

        try:
            transport = self._transport_factory() if self._transport_factory else None
            async for message in query(prompt=_prompt(), options=options, transport=transport):
                if isinstance(message, StreamEvent):
                    event = message.event or {}
                    etype = event.get("type")
                    if etype == "content_block_delta":
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
                                break
                        elif delta.get("type") == "thinking_delta" and delta.get("thinking"):
                            thinking_parts.append(delta["thinking"])
                            yield {"type": "thinking_delta", "text": delta["thinking"]}
                    elif etype == "content_block_start" and (
                        (event.get("content_block") or {}).get("type")
                        in ("tool_use", "server_tool_use")
                    ):
                        # A native tool reached the model despite tools=[]:
                        # stop before the CLI can execute anything itself.
                        native_tool = str((event.get("content_block") or {}).get("name") or "?")
                        break
                    elif etype == "message_start":
                        model_used = str(((event.get("message") or {}).get("model")) or model_used)

                elif isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock) and block.text:
                            final_text = (final_text or "") + block.text
                        elif isinstance(block, ThinkingBlock) and block.thinking:
                            if not thinking_parts:
                                thinking_parts.append(block.thinking)
                        elif isinstance(block, ToolUseBlock):
                            native_tool = block.name
                    model_used = str(message.model or model_used)

                elif isinstance(message, SystemMessage):
                    if message.subtype == "init":
                        data = message.data or {}
                        model_used = str(data.get("model") or model_used)
                        exposed = [t for t in (data.get("tools") or []) if isinstance(t, str)]
                        if exposed:
                            # tools=[] did not hold — say so rather than let
                            # the model act through a tool we never granted.
                            native_tool = exposed[0]
                            break

                elif isinstance(message, ResultMessage):
                    result = message
                    if message.is_error or (
                        message.subtype not in (None, "success")
                        and message.subtype != "error_max_turns"
                    ):
                        error_text = str(
                            message.result or message.api_error_status or message.subtype
                        )
                    if message.subtype == "error_max_turns":
                        self._notify_host({"kind": "rate_limit", "info": {"subtype": "max_turns"}})
        except Exception as exc:  # noqa: BLE001 — classified below
            raise self._as_api_error(exc, stderr_lines) from exc

        if native_tool is not None:
            raise APIError(
                f"Claude Code tried to use its own tool ({native_tool}) — token-only isolation "
                "is broken. Update Claude Code, or run this account in CLI-agent mode.",
                category=ErrorCategory.BAD_REQUEST,
            )

        if error_text is None and result is None and not streamed_text:
            error_text = "\n".join(stderr_lines).strip() or "claude returned no result"

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
        # assistant envelope covers a CLI without partial messages
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

        usage_raw = (result.usage if result is not None else None) or {}
        usage = TokenUsage(
            input_tokens=int(usage_raw.get("input_tokens") or 0),
            output_tokens=int(usage_raw.get("output_tokens") or 0),
            cache_creation_input_tokens=int(usage_raw.get("cache_creation_input_tokens") or 0),
            cache_read_input_tokens=int(usage_raw.get("cache_read_input_tokens") or 0),
            cost_usd=(result.total_cost_usd if result is not None else None),
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
                message_id=str((result.session_id if result is not None else "") or ""),
                raw={"provider": self.provider, "account": self._account_id},
            ),
        }

    def _as_api_error(self, exc: Exception, stderr_lines: list[str]) -> APIError:
        """Map an SDK failure onto this library's error vocabulary.

        The SDK raises typed errors, which is the point of using it: a
        missing binary is a different problem from a dead login, and the
        router fails over from one of those and not the other.
        """
        from claude_agent_sdk import (
            CLIConnectionError,
            CLIJSONDecodeError,
            CLINotFoundError,
            ProcessError,
        )

        if isinstance(exc, APIError):
            return exc
        detail = str(exc) or exc.__class__.__name__
        stderr = "\n".join(stderr_lines).strip()
        label = f" [{self._account_label}]" if self._account_label else ""
        if isinstance(exc, CLINotFoundError):
            return APIError(
                f"Claude Code{label}: the `claude` binary was not found ({self._binary}). "
                "Install Claude Code or set the binary path.",
                category=ErrorCategory.CLI_NOT_FOUND,
                cause=exc,
            )
        if isinstance(exc, (ProcessError, CLIConnectionError, CLIJSONDecodeError)):
            text = f"{detail}\n{stderr}".strip() if stderr else detail
            category = classify_text(text)
            if category == ErrorCategory.UNKNOWN:
                category = ErrorCategory.CLI_PROTOCOL_ERROR
            return APIError(f"Claude Code{label}: {text[:600]}", category=category, cause=exc)
        category = classify_text(detail)
        if category == ErrorCategory.UNKNOWN:
            category = ErrorCategory.NETWORK
        return APIError(f"Claude Code{label}: {detail[:600]}", category=category, cause=exc)
