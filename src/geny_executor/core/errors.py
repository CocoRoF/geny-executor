"""Error classification and exception hierarchy.

Two parallel classification dimensions
--------------------------------------

* :class:`ErrorCategory` (existing) — coarse, retry-aware buckets used by
  the executor's own retry / backoff machinery. Stable since 2.0.x; one
  value per *behavioural* class (recoverable vs fatal).

* :class:`ExecutorErrorCode` (new in 2.1.0) — fine-grained, stable string
  identifier the host (Geny, CI runners, downstream consumers) uses for
  logging, i18n, telemetry grouping, and alerting. Each code is a stable
  string in the form ``exec.<component>.<reason>``. Once published, a
  code's value MUST NOT change — see ``docs/error_codes.md``.

The two coexist on every :class:`GenyExecutorError`:

  - ``e.code`` answers *what specifically went wrong* (``exec.cli.auth_failed``)
  - ``e.category`` (on :class:`APIError`) answers *should we retry?*
    (``ErrorCategory.CLI_AUTH_FAILED.is_fatal == True``)

Backward compatibility
----------------------
Existing call sites of the form
``raise APIError("...", category=ErrorCategory.CLI_AUTH_FAILED)``
keep working unchanged — the base class now derives a reasonable default
``code`` from the ``category`` argument via :data:`_CATEGORY_TO_CODE_DEFAULT`.
Sites that want to be specific can pass ``code=ExecutorErrorCode.EXEC_CLI_AUTH_FAILED``
explicitly; the explicit code wins. The ``code`` attribute is always set
(no ``None`` checks needed downstream).
"""

from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────────────── ExecutorErrorCode ─


class ExecutorErrorCode(str, Enum):
    """Stable, fine-grained error identifiers (2.1.0+).

    Format: ``exec.<component>.<reason>``. Lowercase, dot-separated, ≤4
    segments deep. Designed for:

      - Host logging / Sentry grouping
      - Frontend i18n key lookup (Geny maps ``exec.cli.auth_failed`` →
        ``t("executor.exec.cli.auth_failed")``)
      - Telemetry alerting (route ``exec.api.*`` rate to a different
        channel than ``exec.cli.*``)
      - Operator dashboards

    Stability contract:
      - **NEVER renumber** — once shipped, a code's string value is API surface.
      - **NEVER repurpose** — if the meaning of a code drifts, deprecate
        and add a new code rather than mutating the old one.
      - Adding new codes is non-breaking.

    The full list with descriptions, recoverability, and example
    scenarios lives in ``docs/error_codes.md`` — keep it in sync.
    """

    # ── exec.api.* — vendor API surface (Anthropic/OpenAI/Google/vLLM SDK) ──
    EXEC_API_AUTH_INVALID_KEY = "exec.api.auth.invalid_key"
    EXEC_API_AUTH_EXPIRED = "exec.api.auth.expired"
    EXEC_API_RATE_LIMITED = "exec.api.rate_limited"
    EXEC_API_TIMEOUT = "exec.api.timeout"
    EXEC_API_NETWORK = "exec.api.network"
    EXEC_API_TOKEN_LIMIT = "exec.api.token_limit"
    EXEC_API_BAD_REQUEST = "exec.api.bad_request"
    EXEC_API_SERVER_ERROR = "exec.api.server_error"
    EXEC_API_TERMINAL = "exec.api.terminal"
    EXEC_API_UNKNOWN = "exec.api.unknown"
    EXEC_API_NO_CLIENT = "exec.api.no_client"
    EXEC_API_STREAM_INCOMPLETE = "exec.api.stream_incomplete"
    EXEC_API_RETRY_EXHAUSTED = "exec.api.retry_exhausted"

    # ── exec.cli.* — CLI-driven backends (claude_code_cli) ──
    EXEC_CLI_BINARY_NOT_FOUND = "exec.cli.binary_not_found"
    EXEC_CLI_AUTH_FAILED = "exec.cli.auth_failed"
    EXEC_CLI_TIMEOUT = "exec.cli.timeout"
    EXEC_CLI_PROTOCOL_ERROR = "exec.cli.protocol_error"
    EXEC_CLI_PERMISSION_DENIED = "exec.cli.permission_denied"
    EXEC_CLI_EXITED = "exec.cli.exited"

    # ── exec.pipeline.* / exec.stage.* — pipeline / stage orchestration ──
    EXEC_PIPELINE_NOT_INITIALIZED = "exec.pipeline.not_initialized"
    EXEC_PIPELINE_INVALID_MANIFEST = "exec.pipeline.invalid_manifest"
    EXEC_STAGE_FAILED = "exec.stage.failed"
    EXEC_STAGE_GUARD_REJECTED = "exec.stage.guard_rejected"

    # ── exec.tool.* — Stage 10 tool dispatch ──
    EXEC_TOOL_UNKNOWN = "exec.tool.unknown"
    EXEC_TOOL_INVALID_INPUT = "exec.tool.invalid_input"
    EXEC_TOOL_ACCESS_DENIED = "exec.tool.access_denied"
    EXEC_TOOL_CRASHED = "exec.tool.crashed"
    EXEC_TOOL_TRANSPORT = "exec.tool.transport"

    # ── exec.mutation.* — runtime config mutation ──
    EXEC_MUTATION_INVALID = "exec.mutation.invalid"
    EXEC_MUTATION_LOCKED = "exec.mutation.locked"

    # ── exec.mcp.* — MCP server lifecycle ──
    EXEC_MCP_CONNECT_FAILED = "exec.mcp.connect_failed"
    EXEC_MCP_INITIALIZE_FAILED = "exec.mcp.initialize_failed"
    EXEC_MCP_LIST_TOOLS_FAILED = "exec.mcp.list_tools_failed"
    EXEC_MCP_SDK_MISSING = "exec.mcp.sdk_missing"

    # ── exec.unknown — fallback when no code is attached ──
    EXEC_UNKNOWN = "exec.unknown"

    @classmethod
    def from_category(cls, category: "ErrorCategory") -> "ExecutorErrorCode":
        """Map a legacy ``ErrorCategory`` to a default code.

        Used by :class:`APIError` to keep existing call sites
        (``raise APIError("...", category=ErrorCategory.CLI_AUTH_FAILED)``)
        working unchanged — the resulting exception still gets a
        meaningful ``code`` attribute without the caller having to
        thread it explicitly.
        """
        return _CATEGORY_TO_CODE_DEFAULT.get(category, cls.EXEC_UNKNOWN)


# ─────────────────────────────────────────────────────── ErrorCategory ─


class ErrorCategory(str, Enum):
    """API error classification for retry decisions."""

    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    NETWORK = "network"
    TOKEN_LIMIT = "token_limit"
    AUTH = "auth"
    BAD_REQUEST = "bad_request"
    SERVER_ERROR = "server_error"
    TERMINAL = "terminal"
    UNKNOWN = "unknown"

    # --- CLI-backend specific categories ---
    CLI_NOT_FOUND = "cli_not_found"
    CLI_AUTH_FAILED = "cli_auth_failed"
    CLI_TIMEOUT = "cli_timeout"
    CLI_PROTOCOL_ERROR = "cli_protocol_error"
    CLI_PERMISSION_DENIED = "cli_permission_denied"

    @property
    def is_recoverable(self) -> bool:
        return self in {
            ErrorCategory.RATE_LIMITED,
            ErrorCategory.TIMEOUT,
            ErrorCategory.NETWORK,
            ErrorCategory.SERVER_ERROR,
            ErrorCategory.CLI_TIMEOUT,
            ErrorCategory.CLI_PROTOCOL_ERROR,
        }

    @property
    def is_fatal(self) -> bool:
        """Categories where retry has no chance of recovery."""
        return self in {
            ErrorCategory.AUTH,
            ErrorCategory.BAD_REQUEST,
            ErrorCategory.CLI_NOT_FOUND,
            ErrorCategory.CLI_AUTH_FAILED,
            ErrorCategory.CLI_PERMISSION_DENIED,
        }


# ─────────────────────────────────────────────────────── default mapping ─


# Category → default error code. Kept private — call sites should rely on
# the public ``ExecutorErrorCode.from_category(...)`` accessor so the
# mapping can evolve without churn on consumers.
_CATEGORY_TO_CODE_DEFAULT: dict = {
    ErrorCategory.RATE_LIMITED: ExecutorErrorCode.EXEC_API_RATE_LIMITED,
    ErrorCategory.TIMEOUT: ExecutorErrorCode.EXEC_API_TIMEOUT,
    ErrorCategory.NETWORK: ExecutorErrorCode.EXEC_API_NETWORK,
    ErrorCategory.TOKEN_LIMIT: ExecutorErrorCode.EXEC_API_TOKEN_LIMIT,
    ErrorCategory.AUTH: ExecutorErrorCode.EXEC_API_AUTH_INVALID_KEY,
    ErrorCategory.BAD_REQUEST: ExecutorErrorCode.EXEC_API_BAD_REQUEST,
    ErrorCategory.SERVER_ERROR: ExecutorErrorCode.EXEC_API_SERVER_ERROR,
    ErrorCategory.TERMINAL: ExecutorErrorCode.EXEC_API_TERMINAL,
    ErrorCategory.UNKNOWN: ExecutorErrorCode.EXEC_API_UNKNOWN,
    ErrorCategory.CLI_NOT_FOUND: ExecutorErrorCode.EXEC_CLI_BINARY_NOT_FOUND,
    ErrorCategory.CLI_AUTH_FAILED: ExecutorErrorCode.EXEC_CLI_AUTH_FAILED,
    ErrorCategory.CLI_TIMEOUT: ExecutorErrorCode.EXEC_CLI_TIMEOUT,
    ErrorCategory.CLI_PROTOCOL_ERROR: ExecutorErrorCode.EXEC_CLI_PROTOCOL_ERROR,
    ErrorCategory.CLI_PERMISSION_DENIED: ExecutorErrorCode.EXEC_CLI_PERMISSION_DENIED,
}


# ─────────────────────────────────────────────────────── exceptions ─


class GenyExecutorError(Exception):
    """Base exception for geny-executor.

    Every executor exception carries a stable :class:`ExecutorErrorCode`
    accessible as ``e.code``. Subclasses set a class-level default in
    :attr:`_DEFAULT_CODE` so call sites that pre-date the 2.1.0 code
    field still get a meaningful code on the resulting exception.
    """

    _DEFAULT_CODE: ExecutorErrorCode = ExecutorErrorCode.EXEC_UNKNOWN

    def __init__(
        self,
        message: str,
        *,
        code: Optional[ExecutorErrorCode] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.code: ExecutorErrorCode = code or self._DEFAULT_CODE
        self.cause = cause


class PipelineError(GenyExecutorError):
    """Pipeline-level error."""

    _DEFAULT_CODE = ExecutorErrorCode.EXEC_PIPELINE_NOT_INITIALIZED


class StageError(GenyExecutorError):
    """Stage execution error."""

    _DEFAULT_CODE = ExecutorErrorCode.EXEC_STAGE_FAILED

    def __init__(
        self,
        message: str,
        *,
        stage_name: str = "",
        stage_order: int = 0,
        code: Optional[ExecutorErrorCode] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, code=code, cause=cause)
        self.stage_name = stage_name
        self.stage_order = stage_order


class GuardRejectError(StageError):
    """Guard rejected execution."""

    _DEFAULT_CODE = ExecutorErrorCode.EXEC_STAGE_GUARD_REJECTED

    def __init__(self, message: str, *, guard_name: str = "", **kwargs):
        super().__init__(message, stage_name="guard", stage_order=4, **kwargs)
        self.guard_name = guard_name


class APIError(GenyExecutorError):
    """API call error with classification.

    Carries both a coarse :class:`ErrorCategory` (used by the executor's
    retry machinery) and a fine-grained :class:`ExecutorErrorCode` (used
    by hosts for logging / i18n / telemetry). When ``code`` is omitted,
    it is derived from ``category`` via
    :meth:`ExecutorErrorCode.from_category` so legacy call sites keep
    their existing semantics.
    """

    def __init__(
        self,
        message: str,
        *,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        status_code: Optional[int] = None,
        code: Optional[ExecutorErrorCode] = None,
        cause: Optional[Exception] = None,
    ):
        # Resolve code: explicit > derived-from-category > UNKNOWN.
        resolved_code = code or ExecutorErrorCode.from_category(category)
        super().__init__(message, code=resolved_code, cause=cause)
        self.category = category
        self.status_code = status_code


class ToolExecutionError(GenyExecutorError):
    """Tool execution failed."""

    _DEFAULT_CODE = ExecutorErrorCode.EXEC_TOOL_CRASHED

    def __init__(
        self,
        message: str,
        *,
        tool_name: str = "",
        code: Optional[ExecutorErrorCode] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, code=code, cause=cause)
        self.tool_name = tool_name


class MutationError(GenyExecutorError):
    """Invalid mutation request (bad stage/slot/impl)."""

    _DEFAULT_CODE = ExecutorErrorCode.EXEC_MUTATION_INVALID

    def __init__(
        self,
        message: str,
        *,
        stage_order: int = 0,
        slot_name: str = "",
        code: Optional[ExecutorErrorCode] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, code=code, cause=cause)
        self.stage_order = stage_order
        self.slot_name = slot_name


class MutationLocked(GenyExecutorError):
    """Mutation blocked because the target stage is currently executing."""

    _DEFAULT_CODE = ExecutorErrorCode.EXEC_MUTATION_LOCKED

    def __init__(
        self,
        message: str,
        *,
        stage_order: int = 0,
        code: Optional[ExecutorErrorCode] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, code=code, cause=cause)
        self.stage_order = stage_order
