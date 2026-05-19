"""Claude Code CLI backend.

Wraps Anthropic's ``claude`` command-line agent as a :class:`BaseClient`.
Production prod-grade backend — same canonical APIRequest/APIResponse
contract as every vendor SDK, just routing through a subprocess.

Authentication
--------------
``claude`` reads credentials from one of:
  - ``ANTHROPIC_API_KEY`` env var (passed by this client when ``api_key=`` is set)
  - Subscription auth saved by ``claude auth`` / ``claude setup-token``
  - ``apiKeyHelper`` declared in a ``--settings`` file

This client never forwards the host's full env — only an explicit whitelist
plus the credentials it was told to expose.

Tool execution
--------------
When ``state.llm_client`` is a Claude Code client, the CLI executes its
own built-in tools (Read/Write/Bash/MCP) inside the spawned subprocess.
Geny's tool stage detects this via capabilities (``is_subprocess=True &&
supports_tools=True && requires_workspace=True``) and skips host-side
tool dispatch — see ``stages/s10_tool``.
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Sequence

from geny_executor.core.config import ModelConfig
from geny_executor.core.errors import APIError, ErrorCategory
from geny_executor.llm_client._cli_runtime import (
    CLIAuthFailed,
    CLIBinaryNotFound,
    CLIProcessRunner,
    CLIProtocolError,
    CLIResult,
    CLITimeout,
    aiter_bytes,
    detect_binary,
)
from geny_executor.llm_client.base import BaseClient, ClientCapabilities
from geny_executor.llm_client.translators._cli import (
    assemble_response_from_stream_json,
    build_stream_json_stdin,
    claude_code_argv,
    parse_json_output_to_response,
)
from geny_executor.llm_client.types import (
    APIRequest,
    APIResponse,
    ContentBlock,
    TokenUsage,
)


__all__ = ["ClaudeCodeCLIClient"]


def _classify_cli_result(result: CLIResult) -> APIError:
    """Heuristic mapping of CLI exit codes / stderr → APIError category."""
    stderr = result.stderr.decode("utf-8", errors="replace").lower()
    if "not authenticated" in stderr or "unauthorized" in stderr or "auth" in stderr and "fail" in stderr:
        return APIError(
            f"Claude Code CLI auth failed (exit {result.returncode}): {stderr[:300]}",
            category=ErrorCategory.CLI_AUTH_FAILED,
        )
    if "permission" in stderr and ("denied" in stderr or "deny" in stderr or "blocked" in stderr):
        return APIError(
            f"Claude Code CLI permission denied: {stderr[:300]}",
            category=ErrorCategory.CLI_PERMISSION_DENIED,
        )
    return APIError(
        f"Claude Code CLI exited with code {result.returncode}: {stderr[:300]}",
        category=ErrorCategory.CLI_PROTOCOL_ERROR,
    )


class ClaudeCodeCLIClient(BaseClient):
    """Subprocess-backed Claude Code client."""

    provider = "claude_code_cli"
    capabilities = ClientCapabilities(
        supports_thinking=True,
        supports_tools=True,
        supports_streaming=True,
        supports_tool_choice=False,
        supports_stop_sequences=False,
        supports_top_k=False,
        supports_system_prompt=True,
        supports_structured_output=True,
        supports_session_continuity=True,
        supports_mcp_passthrough=True,
        supports_budget_limit=True,
        supports_token_usage=True,
        supports_cost_usage=True,
        is_subprocess=True,
        requires_workspace=True,
        streaming_granularity="token",
        drops=(
            "tool_choice",
            "stop_sequences",
            "top_k",
            "temperature",
            "top_p",
            "max_tokens",
        ),
    )

    def __init__(
        self,
        *,
        binary_path: Optional[str] = None,
        workspace_dir: Optional[str] = None,
        api_key: str = "",
        settings_path: Optional[str] = None,
        bare_mode: bool = True,
        max_budget_usd: Optional[float] = None,
        default_permission_mode: str = "default",
        mcp_config: Any = None,
        allow_tools: Sequence[str] = (),
        disallow_tools: Sequence[str] = (),
        extra_args: Sequence[str] = (),
        timeout_s: float = 300.0,
        env_extras: Optional[Dict[str, str]] = None,
        event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=None,
            default_headers=None,
            event_sink=event_sink,
        )
        # Binary resolution.
        # - When the caller passes an explicit ``binary_path`` we respect
        #   their choice: if it points to a missing file we surface the
        #   error at send time (CLI_NOT_FOUND) rather than silently using
        #   a different ``claude`` on PATH.
        # - When no override is given we try CLAUDE_CODE_BINARY then
        #   shutil.which("claude").
        if binary_path:
            self._binary = detect_binary("claude", binary_path) or ""
        else:
            env_override = os.environ.get("CLAUDE_CODE_BINARY", "")
            self._binary = (
                detect_binary("claude", env_override) if env_override else None
            ) or detect_binary("claude", None) or ""
        self._workspace_dir = workspace_dir
        self._settings_path = settings_path
        self._bare_mode = bare_mode
        self._max_budget_usd = max_budget_usd
        self._default_permission_mode = default_permission_mode
        self._mcp_config = mcp_config
        self._allow_tools = tuple(allow_tools)
        self._disallow_tools = tuple(disallow_tools)
        self._extra_args = tuple(extra_args)
        self._timeout_s = timeout_s
        self._extra_env: Dict[str, str] = dict(env_extras) if env_extras else {}

    # ─────────────────────────────────────────────────────── helpers ─

    def _env_extras(self) -> Dict[str, str]:
        extras: Dict[str, str] = dict(self._extra_env)
        if self._api_key:
            extras["ANTHROPIC_API_KEY"] = self._api_key
        return extras

    def _make_runner(self) -> CLIProcessRunner:
        if not self._binary:
            raise CLIBinaryNotFound(
                "claude binary not found. Set binary_path=, CLAUDE_CODE_BINARY env var, "
                "or ensure 'claude' is on PATH."
            )
        return CLIProcessRunner(
            binary=self._binary,
            cwd=self._workspace_dir,
            env_extras=self._env_extras(),
            timeout_s=self._timeout_s,
        )

    def _build_argv(self, request: APIRequest) -> List[str]:
        return claude_code_argv(
            request,
            bare_mode=self._bare_mode,
            permission_mode=self._default_permission_mode,
            max_budget_usd=self._max_budget_usd,
            settings_path=self._settings_path,
            mcp_config=self._mcp_config,
            allow_tools=self._allow_tools,
            disallow_tools=self._disallow_tools,
            extra_args=self._extra_args,
        )

    # ─────────────────────────────────────────────────────── _send ─

    async def _send(self, request: APIRequest, *, purpose: str = "") -> APIResponse:
        try:
            runner = self._make_runner()
        except CLIBinaryNotFound as e:
            raise APIError(str(e), category=ErrorCategory.CLI_NOT_FOUND) from e

        argv = self._build_argv(request)
        stdin = build_stream_json_stdin(request.messages) if request.stream else None

        try:
            if request.stream:
                return await assemble_response_from_stream_json(
                    runner.stream(argv, stdin_iter=aiter_bytes(stdin)),
                    model=request.model,
                )
            result = await runner.run_oneshot(argv, stdin=stdin)
            if result.returncode != 0:
                raise _classify_cli_result(result)
            return parse_json_output_to_response(result.stdout, model=request.model)
        except CLIBinaryNotFound as e:
            raise APIError(str(e), category=ErrorCategory.CLI_NOT_FOUND) from e
        except CLITimeout as e:
            raise APIError(str(e), category=ErrorCategory.CLI_TIMEOUT) from e
        except CLIAuthFailed as e:
            raise APIError(str(e), category=ErrorCategory.CLI_AUTH_FAILED) from e
        except CLIProtocolError as e:
            raise APIError(str(e), category=ErrorCategory.CLI_PROTOCOL_ERROR) from e
        except RuntimeError as e:
            # stream-json error envelope was raised by the assembler.
            raise APIError(str(e), category=ErrorCategory.CLI_PROTOCOL_ERROR) from e

    # ───────────────────────────────────────────────── streaming API ─

    async def create_message_stream(
        self,
        *,
        model_config: ModelConfig,
        messages: List[Dict[str, Any]],
        system: Any = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        purpose: str = "",
    ) -> AsyncIterator[Dict[str, Any]]:
        """Yield per-token canonical events as the CLI streams output.

        Events match the format documented in
        ``translators._cli.stream_json_line_to_canonical_event``:
        ``text_delta``, ``thinking_delta``, ``input_json_delta``,
        ``tool_use``, ``content_block_stop``, ``result``, ``error``.

        After the CLI exits we emit one final
        ``{"type": "message_complete", "response": APIResponse}``
        event with the fully assembled response (text + thinking +
        tool_use blocks, stop_reason, usage). Without this terminal
        envelope the s06_api stage's streaming consumer raises
        ``Stream ended without message_complete`` — it builds the
        assistant message from ``chunk["response"]`` and the previous
        implementation never populated that field. (Mirrors the
        ``anthropic`` / ``openai`` / ``google`` SDK clients' contract.)
        """
        request = self._build_request(
            model_config=model_config,
            messages=messages,
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            stream=True,
        )

        try:
            runner = self._make_runner()
        except CLIBinaryNotFound as e:
            raise APIError(str(e), category=ErrorCategory.CLI_NOT_FOUND) from e

        argv = self._build_argv(request)
        stdin = build_stream_json_stdin(messages)

        from geny_executor.llm_client._cli_runtime import parse_stream_json_line
        from geny_executor.llm_client.translators._cli import (
            stream_json_line_to_canonical_event,
        )

        # Accumulator state — mirrors ``assemble_response_from_stream_json``
        # so the final message_complete envelope carries the same
        # APIResponse the non-streaming path produces.
        import json as _json

        text_buf: List[str] = []
        thinking_buf: List[str] = []
        tool_uses: List[Dict[str, Any]] = []
        current_tool: Optional[Dict[str, Any]] = None
        final_obj: Optional[Dict[str, Any]] = None
        message_id = ""
        stop_reason = "end_turn"
        resolved_model = model_config.model

        try:
            async for raw in runner.stream(argv, stdin_iter=aiter_bytes(stdin)):
                line_obj = parse_stream_json_line(raw)
                if line_obj is None:
                    continue

                # ── Accumulate for the terminal APIResponse ──
                ltype = str(line_obj.get("type", ""))
                if ltype == "system":
                    message_id = str(
                        line_obj.get("session_id")
                        or line_obj.get("message_id")
                        or message_id
                    )
                    resolved_model = str(line_obj.get("model") or resolved_model)
                elif ltype == "assistant":
                    delta = line_obj.get("delta") or {}
                    dtype = str(delta.get("type", ""))
                    if dtype == "text_delta":
                        text_buf.append(str(delta.get("text", "")))
                    elif dtype == "thinking_delta":
                        thinking_buf.append(str(delta.get("text", "")))
                    elif dtype == "input_json_delta":
                        if current_tool is not None:
                            current_tool.setdefault("_partial_json", "")
                            current_tool["_partial_json"] += str(
                                delta.get("partial_json", "")
                            )
                    else:
                        cb = line_obj.get("content_block")
                        if isinstance(cb, dict) and cb.get("type") == "tool_use":
                            current_tool = {
                                "id": cb.get("id"),
                                "name": cb.get("name"),
                                "input": cb.get("input") or {},
                            }
                elif ltype == "content_block_stop":
                    if current_tool is not None:
                        partial = current_tool.pop("_partial_json", "")
                        if partial and not current_tool.get("input"):
                            try:
                                current_tool["input"] = _json.loads(partial)
                            except _json.JSONDecodeError:
                                current_tool["input"] = {"_raw": partial}
                        tool_uses.append(current_tool)
                        current_tool = None
                elif ltype == "result":
                    final_obj = line_obj
                    stop_reason = str(line_obj.get("stop_reason", stop_reason))

                # ── Yield the per-line canonical event ──
                # Suppress the translator's bare ``message_complete``
                # (it carries no response field) — we emit the
                # populated version after the loop. Everything else
                # passes through unchanged.
                event = stream_json_line_to_canonical_event(line_obj)
                if event is None:
                    continue
                if event.get("type") == "message_complete":
                    continue
                yield event

            # ── Assemble + emit the terminal message_complete ──
            blocks: List[ContentBlock] = []
            if thinking_buf:
                blocks.append(
                    ContentBlock(type="thinking", thinking_text="".join(thinking_buf))
                )
            if text_buf:
                blocks.append(ContentBlock(type="text", text="".join(text_buf)))
            for tu in tool_uses:
                blocks.append(
                    ContentBlock(
                        type="tool_use",
                        tool_use_id=tu.get("id"),
                        tool_name=tu.get("name"),
                        tool_input=tu.get("input") or {},
                    )
                )

            usage_in: Dict[str, Any] = (final_obj or {}).get("usage", {}) or {}
            usage = TokenUsage(
                input_tokens=int(usage_in.get("input_tokens", 0) or 0),
                output_tokens=int(usage_in.get("output_tokens", 0) or 0),
                cache_creation_input_tokens=int(
                    usage_in.get("cache_creation_input_tokens", 0) or 0
                ),
                cache_read_input_tokens=int(
                    usage_in.get("cache_read_input_tokens", 0) or 0
                ),
                cost_usd=usage_in.get("cost_usd"),
                duration_ms=(final_obj or {}).get("duration_ms"),
            )

            response = APIResponse(
                content=blocks,
                stop_reason=stop_reason,
                usage=usage,
                model=resolved_model,
                message_id=message_id,
                raw=final_obj or {},
            )
            yield {"type": "message_complete", "response": response}
        except CLIBinaryNotFound as e:
            raise APIError(str(e), category=ErrorCategory.CLI_NOT_FOUND) from e
        except CLITimeout as e:
            raise APIError(str(e), category=ErrorCategory.CLI_TIMEOUT) from e
        except CLIProtocolError as e:
            raise APIError(str(e), category=ErrorCategory.CLI_PROTOCOL_ERROR) from e
