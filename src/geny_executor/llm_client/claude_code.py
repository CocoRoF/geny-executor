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
    StreamJsonAccumulator,
    assemble_response_from_stream_json,
    build_stream_json_stdin,
    claude_code_argv,
    parse_json_output_to_response,
)
from geny_executor.llm_client.types import APIRequest, APIResponse


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

        # Shared accumulator handles both stream-json shapes:
        #   - delta form (``--include-partial-messages`` on, true streaming)
        #   - full-message form (Claude Code 2.x default — content[]
        #     arrives in one ``assistant`` envelope).
        # Without the message-form branch, every assistant frame yielded
        # zero text and the terminal APIResponse came back empty —
        # exactly the symptom the user reported (``output_len=0``).
        accum = StreamJsonAccumulator(model=model_config.model)

        try:
            async for raw in runner.stream(argv, stdin_iter=aiter_bytes(stdin)):
                line_obj = parse_stream_json_line(raw)
                if line_obj is None:
                    continue
                if "__malformed__" in line_obj:
                    continue
                # Surface CLI-side errors as APIError so the stage's
                # retry/escalate path runs instead of silently producing
                # an empty response.
                if str(line_obj.get("type", "")) == "error":
                    raise APIError(
                        f"Claude Code CLI reported error: "
                        f"{line_obj.get('message') or line_obj!r}",
                        category=ErrorCategory.CLI_PROTOCOL_ERROR,
                    )
                # Surface the authentication_failed annotation that the
                # CLI emits on the assistant frame when no credential
                # is available — without this we'd swallow the
                # "Not logged in" placeholder text as the assistant's
                # answer and call the session "successful".
                if str(line_obj.get("error", "")) == "authentication_failed":
                    raise APIError(
                        "Claude Code CLI is not authenticated (claude --print "
                        "returned error=authentication_failed). Sign in via "
                        "Settings → LLM Backends → Claude Code (CLI).",
                        category=ErrorCategory.CLI_AUTH_FAILED,
                    )

                # Feed accumulator + stream canonical events to consumer.
                for event in accum.feed(line_obj):
                    yield event

            yield {"type": "message_complete", "response": accum.finalize()}
        except CLIBinaryNotFound as e:
            raise APIError(str(e), category=ErrorCategory.CLI_NOT_FOUND) from e
        except CLITimeout as e:
            raise APIError(str(e), category=ErrorCategory.CLI_TIMEOUT) from e
        except CLIProtocolError as e:
            raise APIError(str(e), category=ErrorCategory.CLI_PROTOCOL_ERROR) from e
