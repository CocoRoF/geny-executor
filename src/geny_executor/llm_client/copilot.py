"""GitHub Copilot CLI backend.

Wraps ``gh copilot`` (the Copilot CLI extension to the ``gh`` binary) as a
:class:`BaseClient`. Plain-text stdout only — no streaming, no structured
output, no tool round-trip (only ``--allow-tool`` allowlist gating). The
client advertises this honestly via ``ClientCapabilities``.

Authentication
--------------
``gh copilot`` reads its credentials from the host's ``gh auth`` state
(typically ``~/.config/gh/hosts.yml``). This client makes no attempt to
manage that — operators run ``gh auth login`` and ``gh extension install
github/gh-copilot`` themselves. The client surfaces auth failures via
``ErrorCategory.CLI_AUTH_FAILED`` based on stderr heuristics.

Streaming
---------
``supports_streaming=False``. ``create_message_stream`` falls back to
``BaseClient``'s default (one ``message_complete`` event after the full
response).
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Sequence

from geny_executor.core.errors import APIError, ErrorCategory
from geny_executor.llm_client._cli_runtime import (
    CLIAuthFailed,
    CLIBinaryNotFound,
    CLIProcessRunner,
    CLIProtocolError,
    CLIResult,
    CLITimeout,
    detect_binary,
)
from geny_executor.llm_client.base import BaseClient, ClientCapabilities
from geny_executor.llm_client.translators._cli import (
    compose_copilot_prompt,
    copilot_argv,
    parse_plain_text_to_response,
)
from geny_executor.llm_client.types import APIRequest, APIResponse


__all__ = ["CopilotCLIClient"]


def _classify_cli_result(result: CLIResult) -> APIError:
    """Map gh copilot exit-codes / stderr → APIError category."""
    stderr = result.stderr.decode("utf-8", errors="replace").lower()
    if "not logged in" in stderr or "authentication" in stderr or "unauthorized" in stderr or "auth required" in stderr:
        return APIError(
            f"Copilot CLI auth failed (exit {result.returncode}): {stderr[:300]}",
            category=ErrorCategory.CLI_AUTH_FAILED,
        )
    if "not installed" in stderr or "extension not found" in stderr:
        return APIError(
            f"gh copilot extension not installed: {stderr[:300]}",
            category=ErrorCategory.CLI_NOT_FOUND,
        )
    if "permission" in stderr and ("denied" in stderr or "deny" in stderr or "blocked" in stderr):
        return APIError(
            f"Copilot CLI permission denied: {stderr[:300]}",
            category=ErrorCategory.CLI_PERMISSION_DENIED,
        )
    return APIError(
        f"Copilot CLI exited with code {result.returncode}: {stderr[:300]}",
        category=ErrorCategory.CLI_PROTOCOL_ERROR,
    )


class CopilotCLIClient(BaseClient):
    """Subprocess-backed GitHub Copilot CLI client."""

    provider = "copilot_cli"
    capabilities = ClientCapabilities(
        supports_thinking=False,
        supports_tools=False,
        supports_streaming=False,
        supports_tool_choice=False,
        supports_stop_sequences=False,
        supports_top_k=False,
        supports_system_prompt=True,  # via prompt prepend
        supports_structured_output=False,
        supports_session_continuity=False,
        supports_mcp_passthrough=False,
        supports_budget_limit=False,
        supports_token_usage=False,
        supports_cost_usage=False,
        is_subprocess=True,
        requires_workspace=False,
        streaming_granularity="none",
        drops=(
            "tools",
            "tool_choice",
            "thinking_enabled",
            "stop_sequences",
            "top_k",
            "temperature",
            "top_p",
            "max_tokens",
            "response_format",
            "session_hint",
        ),
    )

    def __init__(
        self,
        *,
        gh_binary_path: Optional[str] = None,
        allow_tools: Sequence[str] = (),
        cwd: Optional[str] = None,
        extra_args: Sequence[str] = (),
        timeout_s: float = 180.0,
        env_extras: Optional[Dict[str, str]] = None,
        event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        super().__init__(
            api_key="",
            base_url=None,
            default_headers=None,
            event_sink=event_sink,
        )
        # Binary resolution: explicit > GH_BINARY env > which("gh").
        if gh_binary_path:
            self._gh = detect_binary("gh", gh_binary_path) or ""
        else:
            env_override = os.environ.get("GH_BINARY", "")
            self._gh = (
                detect_binary("gh", env_override) if env_override else None
            ) or detect_binary("gh", None) or ""
        self._allow_tools = tuple(allow_tools)
        self._cwd = cwd
        self._extra_args = tuple(extra_args)
        self._timeout_s = timeout_s
        self._extra_env: Dict[str, str] = dict(env_extras) if env_extras else {}

    def _env_extras(self) -> Dict[str, str]:
        return dict(self._extra_env)

    def _make_runner(self) -> CLIProcessRunner:
        if not self._gh:
            raise CLIBinaryNotFound(
                "gh binary not found. Install GitHub CLI and set gh_binary_path= "
                "or ensure 'gh' is on PATH (then install the copilot extension "
                "via `gh extension install github/gh-copilot`)."
            )
        return CLIProcessRunner(
            binary=self._gh,
            cwd=self._cwd,
            env_extras=self._env_extras(),
            timeout_s=self._timeout_s,
        )

    # ─────────────────────────────────────────────────────── _send ─

    async def _send(self, request: APIRequest, *, purpose: str = "") -> APIResponse:
        try:
            runner = self._make_runner()
        except CLIBinaryNotFound as e:
            raise APIError(str(e), category=ErrorCategory.CLI_NOT_FOUND) from e

        prompt = compose_copilot_prompt(request.system, request.messages)
        argv = copilot_argv(
            allow_tools=self._allow_tools,
            extra_args=self._extra_args,
        )
        argv += ["-p", prompt]

        try:
            result = await runner.run_oneshot(argv)
            if result.returncode != 0:
                raise _classify_cli_result(result)
            text = result.stdout.decode("utf-8", errors="replace")
            return parse_plain_text_to_response(text, model=request.model)
        except CLIBinaryNotFound as e:
            raise APIError(str(e), category=ErrorCategory.CLI_NOT_FOUND) from e
        except CLITimeout as e:
            raise APIError(str(e), category=ErrorCategory.CLI_TIMEOUT) from e
        except CLIAuthFailed as e:
            raise APIError(str(e), category=ErrorCategory.CLI_AUTH_FAILED) from e
        except CLIProtocolError as e:
            raise APIError(str(e), category=ErrorCategory.CLI_PROTOCOL_ERROR) from e
