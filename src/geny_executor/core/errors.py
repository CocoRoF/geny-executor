"""Error classification and exception hierarchy."""

from enum import Enum
from typing import Optional


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

    @property
    def is_recoverable(self) -> bool:
        return self in {
            ErrorCategory.RATE_LIMITED,
            ErrorCategory.TIMEOUT,
            ErrorCategory.NETWORK,
            ErrorCategory.SERVER_ERROR,
        }


class GenyExecutorError(Exception):
    """Base exception for geny-executor."""

    def __init__(self, message: str, *, cause: Optional[Exception] = None):
        super().__init__(message)
        self.cause = cause


class PipelineError(GenyExecutorError):
    """Pipeline-level error."""

    pass


class StageError(GenyExecutorError):
    """Stage execution error."""

    def __init__(
        self,
        message: str,
        *,
        stage_name: str = "",
        stage_order: int = 0,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, cause=cause)
        self.stage_name = stage_name
        self.stage_order = stage_order


class GuardRejectError(StageError):
    """Guard rejected execution."""

    def __init__(self, message: str, *, guard_name: str = "", **kwargs):
        super().__init__(message, stage_name="guard", stage_order=4, **kwargs)
        self.guard_name = guard_name


class APIError(GenyExecutorError):
    """API call error with classification."""

    def __init__(
        self,
        message: str,
        *,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        status_code: Optional[int] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, cause=cause)
        self.category = category
        self.status_code = status_code


class ToolExecutionError(GenyExecutorError):
    """Tool execution failed."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str = "",
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, cause=cause)
        self.tool_name = tool_name


class MutationError(GenyExecutorError):
    """Invalid mutation request (bad stage/slot/impl)."""

    def __init__(
        self,
        message: str,
        *,
        stage_order: int = 0,
        slot_name: str = "",
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, cause=cause)
        self.stage_order = stage_order
        self.slot_name = slot_name


class MutationLocked(GenyExecutorError):
    """Mutation blocked because the target stage is currently executing."""

    def __init__(
        self,
        message: str,
        *,
        stage_order: int = 0,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, cause=cause)
        self.stage_order = stage_order
