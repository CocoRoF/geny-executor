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

Which channel drives a given client is declared via ``auth_mode=``
(``'api_key' | 'oauth' | 'setup_token' | 'auto'``); it decides whether
``--bare`` is emitted. ``'auto'`` resolves from the client's own
``api_key`` — never from the host process env, which is scrubbed before
spawn and historically lied about the child's credential reality
(PR #868).

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

from dataclasses import replace
import logging
import os
from contextlib import aclosing
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
    ContainerCLIRunner,
    SandboxHandle,
    aiter_bytes,
    detect_binary,
)
from geny_executor.llm_client.base import BaseClient, ClientCapabilities
from geny_executor.llm_client.translators._cli import (
    StreamJsonAccumulator,
    assemble_response_from_stream_json,
    build_stream_json_stdin,
    messages_have_images,
    claude_code_argv,
    parse_json_output_to_response,
)
from geny_executor.llm_client.types import APIRequest, APIResponse


logger = logging.getLogger(__name__)

__all__ = ["ClaudeCodeCLIClient"]


#: Anchored stderr phrases that indicate an authentication failure.
#:
#: Deliberately *specific*: the pre-2.2.0 heuristic matched bare
#: ``'auth' and 'fail'`` substrings anywhere in stderr, so any MCP/tool
#: noise mentioning e.g. an "oauth-helper failed to start" was
#: misclassified as CLI_AUTH_FAILED — a fatal, non-retryable category —
#: when the actual failure was a transient protocol error. Only phrases
#: the CLI itself emits on credential problems belong here.
_AUTH_FAILURE_PHRASES = (
    "not authenticated",
    "unauthorized",
    "authentication_failed",
    "invalid api key",
)


def _classify_cli_result(result: CLIResult, *, cli_version: str = "") -> APIError:
    """Heuristic mapping of CLI exit codes / stderr → APIError category.

    ``cli_version`` (when the caller has completed the version handshake)
    is appended to the message — all four 2.1.x incidents were version
    skew, and post-hoc diagnosis needs that one fact recorded at the
    moment of failure, not reconstructed from deploy logs.
    """
    stderr = result.stderr.decode("utf-8", errors="replace").lower()
    suffix = f" [cli_version={cli_version}]" if cli_version else ""
    if any(phrase in stderr for phrase in _AUTH_FAILURE_PHRASES):
        return APIError(
            f"Claude Code CLI auth failed (exit {result.returncode}): {stderr[:300]}{suffix}",
            category=ErrorCategory.CLI_AUTH_FAILED,
        )
    if "permission" in stderr and ("denied" in stderr or "deny" in stderr or "blocked" in stderr):
        return APIError(
            f"Claude Code CLI permission denied: {stderr[:300]}{suffix}",
            category=ErrorCategory.CLI_PERMISSION_DENIED,
        )
    return APIError(
        f"Claude Code CLI exited with code {result.returncode}: {stderr[:300]}{suffix}",
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

    #: Wall-clock cap for the one-time ``--version`` handshake. Short on
    #: purpose: the probe must never meaningfully delay the first real
    #: call, and a hung probe degrades to ``cli_version="unknown"``.
    _VERSION_PROBE_TIMEOUT_S = 10.0

    def __init__(
        self,
        *,
        binary_path: Optional[str] = None,
        workspace_dir: Optional[str] = None,
        api_key: str = "",
        auth_mode: str = "auto",
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
        strict_wire: bool = False,
        runner_factory: Optional[Callable[..., CLIProcessRunner]] = None,
        session_hint: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Construct a Claude Code CLI client.

        2.2.0 boundary-hardening kwargs (audit §2.2/§2.3, Tier 1-1/2):

        ``auth_mode``
            ``'api_key' | 'oauth' | 'setup_token' | 'auto'``. Declares
            which credential channel the CLI should be driven through;
            replaces the deleted process-env sniff in the argv builder
            (which read the *parent* env — a variable the scrubbed child
            never necessarily sees; PR #868 history). ``'auto'`` resolves
            to ``api_key`` iff ``api_key=`` is non-empty, else the
            subscription (OAuth) path.
        ``strict_wire``
            When True, any unknown / malformed stream-json line fails the
            call with ``CLI_PROTOCOL_ERROR`` instead of being tolerated.
            Meant for CI canaries that should turn the next wire drift
            into a failing test *before* release — never for prod, where
            tolerate-and-report is the right posture.
        ``runner_factory``
            Optional ``Callable[..., CLIProcessRunner]`` receiving
            ``binary=``, ``cwd=``, ``env_extras=``, ``timeout_s=``.
            The supported seam for hosts that wrap process spawning
            (GAPT's docker sandbox) — absorbs the
            ``CLIProcessRunner._spawn`` monkey-patch that pinned GAPT to
            2.1.0. The version-handshake probe routes through the same
            factory so the recorded version matches the binary that
            actually runs.
        ``session_hint``
            Default ``{"session_id": ..., "resume": bool}`` applied to
            requests built through the high-level
            ``create_message`` / ``create_message_stream`` surface
            (which had no way to carry one — making
            ``supports_session_continuity=True`` an empty promise).
            Hosts update it between turns via
            ``client.configure(session_hint=...)``. A per-request
            ``APIRequest.session_hint`` still wins.
        """
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
                (detect_binary("claude", env_override) if env_override else None)
                or detect_binary("claude", None)
                or ""
            )
        self._workspace_dir = workspace_dir
        self._auth_mode = auth_mode
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
        self._strict_wire = strict_wire
        self._runner_factory = runner_factory
        self._session_hint: Optional[Dict[str, Any]] = dict(session_hint) if session_hint else None
        #: ``None`` = handshake not attempted yet; ``"unknown"`` = attempted
        #: and failed (never retried — one probe per client instance).
        self._cli_version_value: Optional[str] = None

    # ─────────────────────────────────────────────────────── helpers ─

    def _env_extras(self) -> Dict[str, str]:
        extras: Dict[str, str] = dict(self._extra_env)
        if self._api_key:
            extras["ANTHROPIC_API_KEY"] = self._api_key
        return extras

    def _make_runner(self, *, timeout_s: Optional[float] = None) -> CLIProcessRunner:
        effective_timeout = self._timeout_s if timeout_s is None else timeout_s
        # A host-supplied runner factory (e.g. a container sandbox) runs the
        # CLI elsewhere — the agent binary need not exist on this host — so the
        # host-binary check is the *default* in-process runner's concern only.
        if self._runner_factory is not None:
            return self._runner_factory(
                binary=self._binary,
                cwd=self._workspace_dir,
                env_extras=self._env_extras(),
                timeout_s=effective_timeout,
            )
        if not self._binary:
            raise CLIBinaryNotFound(
                "claude binary not found. Set binary_path=, CLAUDE_CODE_BINARY env var, "
                "or ensure 'claude' is on PATH."
            )
        return CLIProcessRunner(
            binary=self._binary,
            cwd=self._workspace_dir,
            env_extras=self._env_extras(),
            timeout_s=effective_timeout,
        )

    def _build_argv(self, request: APIRequest) -> List[str]:
        return claude_code_argv(
            request,
            bare_mode=self._bare_mode,
            auth_mode=self._auth_mode,
            has_api_key=bool(self._api_key),
            permission_mode=self._default_permission_mode,
            max_budget_usd=self._max_budget_usd,
            settings_path=self._settings_path,
            mcp_config=self._mcp_config,
            allow_tools=self._allow_tools,
            disallow_tools=self._disallow_tools,
            extra_args=self._extra_args,
        )

    def _build_request(
        self,
        *,
        model_config: ModelConfig,
        messages: List[Dict[str, Any]],
        system: Any,
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Dict[str, Any]],
        stream: bool,
    ) -> APIRequest:
        """Canonical request assembly + client-level session continuity.

        The high-level ``create_message`` / ``create_message_stream``
        surface (the only one stages call) has no ``session_hint``
        parameter, so before 2.2.0 ``supports_session_continuity=True``
        was advertised but unreachable: the argv builder knew how to emit
        ``--resume`` / ``--session-id`` and no request ever carried the
        hint. The client-level default set via the constructor or
        ``configure(session_hint=...)`` closes that gap; an explicit
        per-request hint (low-level ``_send`` callers) still wins.
        """
        request = super()._build_request(
            model_config=model_config,
            messages=messages,
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            stream=stream,
        )
        if self._session_hint and request.session_hint is None:
            request.session_hint = dict(self._session_hint)
        return request

    # ─────────────────────────────────────── version handshake ─

    async def _ensure_cli_version(self) -> str:
        """One-time ``<binary> --version`` handshake (lazy, cached).

        All four 2.1.x boundary incidents were version skew, and the
        post-mortems had to reconstruct which CLI was deployed from
        infrastructure logs because nothing in the executor recorded it.
        The probe runs once per client instance, is capped at
        ``_VERSION_PROBE_TIMEOUT_S``, and **never** fails the call — a
        broken probe caches ``"unknown"`` and moves on. The result is
        logged at INFO, attached to ``APIResponse.raw['cli_version']``,
        and appended to CLI ``APIError`` messages.
        """
        if self._cli_version_value is not None:
            return self._cli_version_value
        version = "unknown"
        try:
            runner = self._make_runner(
                timeout_s=min(self._VERSION_PROBE_TIMEOUT_S, self._timeout_s)
            )
            result = await runner.run_oneshot(["--version"])
            text = result.stdout.decode("utf-8", errors="replace").strip()
            if result.returncode == 0 and text:
                # First line only — defensive against chatty wrappers.
                version = text.splitlines()[0].strip()
        except Exception:
            # Deliberately broad: the handshake is telemetry, not a
            # precondition. Whatever broke here will resurface with a
            # proper category on the real call.
            version = "unknown"
        self._cli_version_value = version
        logger.info(
            "Claude Code CLI version handshake: %s (binary=%s)",
            version,
            self._binary,
        )
        return version

    def _with_version(self, message: str) -> str:
        """Append the handshaken CLI version to an error message."""
        if self._cli_version_value:
            return f"{message} [cli_version={self._cli_version_value}]"
        return message

    def _attach_cli_version(self, response: APIResponse) -> APIResponse:
        if isinstance(response.raw, dict) and self._cli_version_value:
            response.raw.setdefault("cli_version", self._cli_version_value)
        return response

    # ─────────────────────────────────────── wire-shape telemetry ─

    def _report_unknown_wire(
        self,
        *,
        unknown_count: int,
        malformed_count: int,
        first_unknown_type: Optional[str],
    ) -> None:
        """Forward wire-drift telemetry; optionally fail under strict_wire.

        Emitted at most once per call (the caller invokes this once,
        after the stream drains) so hosts get a single
        ``llm_client.unknown_wire_shape`` signal per request rather than
        a token-rate flood. This is the consumer the v2.1.4 masking
        channel never had: the parser produced ``cli_unknown`` tags for
        weeks and nothing read them (audit §2.2).
        """
        total = unknown_count + malformed_count
        if not total:
            return
        if self._event_sink is not None:
            self._event_sink(
                {
                    "type": "llm_client.unknown_wire_shape",
                    "provider": self.provider,
                    "unknown_type": first_unknown_type,
                    "count": total,
                    "unknown_line_count": unknown_count,
                    "malformed_line_count": malformed_count,
                    "cli_version": self._cli_version_value or "unknown",
                }
            )
        if self._strict_wire:
            raise APIError(
                self._with_version(
                    "Claude Code CLI emitted "
                    f"{total} unknown/malformed stream-json line(s) "
                    f"(first unknown type: {first_unknown_type!r}) and this "
                    "client was constructed with strict_wire=True"
                ),
                category=ErrorCategory.CLI_PROTOCOL_ERROR,
            )

    def _post_wire_checks(self, response: APIResponse) -> APIResponse:
        """Telemetry + strict enforcement for the assembler path, which
        only exposes counts through ``APIResponse.raw``."""
        raw = response.raw if isinstance(response.raw, dict) else {}
        self._report_unknown_wire(
            unknown_count=int(raw.get("unknown_line_count", 0) or 0),
            malformed_count=int(raw.get("malformed_line_count", 0) or 0),
            first_unknown_type=raw.get("first_unknown_type"),
        )
        return self._attach_cli_version(response)

    # ─────────────────────────────────────────────────────── _send ─

    async def _send(self, request: APIRequest, *, purpose: str = "") -> APIResponse:
        try:
            runner = self._make_runner()
        except CLIBinaryNotFound as e:
            raise APIError(str(e), category=ErrorCategory.CLI_NOT_FOUND) from e

        # Vision on the non-streaming surface: the ``--print`` positional
        # prompt is text-only, so a request that carries image blocks must
        # travel over the stream-json wire (which ingests base64 images
        # natively). ``create_message`` still returns one assembled
        # APIResponse — only the wire mode changes. Without this, every
        # non-stream vision call (e.g. screen-observation captioning) lost
        # its image and the model answered "I don't see an image".
        if not request.stream and messages_have_images(request.messages):
            request = replace(request, stream=True)

        cli_version = await self._ensure_cli_version()
        argv = self._build_argv(request)
        stdin = build_stream_json_stdin(request.messages) if request.stream else None

        try:
            if request.stream:
                response = await assemble_response_from_stream_json(
                    runner.stream(argv, stdin_iter=aiter_bytes(stdin)),
                    model=request.model,
                    cli_version=cli_version,
                )
                return self._post_wire_checks(response)
            result = await runner.run_oneshot(argv, stdin=stdin)
            if result.returncode != 0:
                raise _classify_cli_result(result, cli_version=cli_version)
            return self._attach_cli_version(
                parse_json_output_to_response(result.stdout, model=request.model)
            )
        except CLIBinaryNotFound as e:
            raise APIError(self._with_version(str(e)), category=ErrorCategory.CLI_NOT_FOUND) from e
        except CLITimeout as e:
            raise APIError(self._with_version(str(e)), category=ErrorCategory.CLI_TIMEOUT) from e
        except CLIAuthFailed as e:
            raise APIError(
                self._with_version(str(e)), category=ErrorCategory.CLI_AUTH_FAILED
            ) from e
        except CLIProtocolError as e:
            raise APIError(
                self._with_version(str(e)), category=ErrorCategory.CLI_PROTOCOL_ERROR
            ) from e
        except RuntimeError as e:
            # stream-json error envelope was raised by the assembler.
            raise APIError(
                self._with_version(str(e)), category=ErrorCategory.CLI_PROTOCOL_ERROR
            ) from e

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

        cli_version = await self._ensure_cli_version()
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
        accum = StreamJsonAccumulator(model=model_config.model, cli_version=cli_version)

        try:
            # ``aclosing`` finalizes the runner generator *synchronously*
            # when this generator is closed mid-answer (SSE consumer
            # disconnect → GeneratorExit at a ``yield`` below). Without
            # it the inner generator — and the kill ladder in its
            # ``finally`` — would only run whenever the GC's asyncgen
            # hook got around to it, leaving a live ``claude`` child in
            # the meantime (audit 2026-06-09 §3.7).
            async with aclosing(runner.stream(argv, stdin_iter=aiter_bytes(stdin))) as lines:
                async for raw in lines:
                    line_obj = parse_stream_json_line(raw)
                    if line_obj is None:
                        continue
                    # Surface CLI-side errors as APIError so the stage's
                    # retry/escalate path runs instead of silently producing
                    # an empty response. (Malformed lines have no ``type``
                    # key — they fall through to ``feed`` for counting.)
                    if str(line_obj.get("type", "")) == "error":
                        raise APIError(
                            self._with_version(
                                f"Claude Code CLI reported error: "
                                f"{line_obj.get('message') or line_obj!r}"
                            ),
                            category=ErrorCategory.CLI_PROTOCOL_ERROR,
                        )
                    # Surface the authentication_failed annotation that the
                    # CLI emits on the assistant frame when no credential
                    # is available — without this we'd swallow the
                    # "Not logged in" placeholder text as the assistant's
                    # answer and call the session "successful".
                    if str(line_obj.get("error", "")) == "authentication_failed":
                        raise APIError(
                            self._with_version(
                                "Claude Code CLI is not authenticated (claude --print "
                                "returned error=authentication_failed). Sign in via "
                                "Settings → LLM Backends → Claude Code (CLI)."
                            ),
                            category=ErrorCategory.CLI_AUTH_FAILED,
                        )

                    # Feed accumulator + stream canonical events to consumer.
                    for event in accum.feed(line_obj):
                        yield event

            # Wire-drift telemetry must run before the terminal envelope:
            # strict_wire failures should look like a failed call, not a
            # successful one with a footnote.
            self._report_unknown_wire(
                unknown_count=accum.unknown_line_count,
                malformed_count=accum.malformed_line_count,
                first_unknown_type=accum.first_unknown_type,
            )
            yield {
                "type": "message_complete",
                "response": self._attach_cli_version(accum.finalize()),
            }
        except CLIBinaryNotFound as e:
            raise APIError(self._with_version(str(e)), category=ErrorCategory.CLI_NOT_FOUND) from e
        except CLITimeout as e:
            raise APIError(self._with_version(str(e)), category=ErrorCategory.CLI_TIMEOUT) from e
        except CLIProtocolError as e:
            raise APIError(
                self._with_version(str(e)), category=ErrorCategory.CLI_PROTOCOL_ERROR
            ) from e


def build_container_cli_client(
    *,
    sandbox: SandboxHandle,
    workdir: str = "/workspace",
    launcher: str = "docker",
    container_binary: str = "claude",
    **client_kwargs: Any,
) -> "ClaudeCodeCLIClient":
    """Build a :class:`ClaudeCodeCLIClient` whose every process spawn — the
    per-request CLI run *and* the one-time ``--version`` handshake — happens
    inside ``sandbox``'s container via :class:`ContainerCLIRunner`.

    This is the supported, host-agnostic way to run the agent CLI in a
    sandbox; it absorbs the bespoke ``SandboxedCLIProcessRunner`` that hosts
    (GAPT) previously had to carry. The host does **not** need the agent
    binary installed — it lives in the container image — only the ``launcher``
    (``docker`` by default).

    ``client_kwargs`` are forwarded verbatim to :class:`ClaudeCodeCLIClient`
    (``api_key``, ``auth_mode``, ``mcp_config``, ``allow_tools``,
    ``workspace_dir``, ...). ``runner_factory`` must not be passed — it is set
    here.

    Example::

        client = build_container_cli_client(
            sandbox=workspace_sandbox,   # has .container_name + async .ensure()
            api_key=api_key,
            mcp_config=mcp_config,
        )
        pipeline.attach_runtime(llm_client=client, hook_runner=hook_runner)
    """
    if "runner_factory" in client_kwargs:
        raise TypeError(
            "build_container_cli_client sets runner_factory itself; "
            "do not pass it in client_kwargs"
        )

    def _factory(**runner_kwargs: Any) -> CLIProcessRunner:
        return ContainerCLIRunner(
            sandbox=sandbox,
            workdir=workdir,
            launcher=launcher,
            container_binary=container_binary,
            **runner_kwargs,
        )

    return ClaudeCodeCLIClient(**client_kwargs, runner_factory=_factory)
