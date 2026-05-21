"""Stability regression for :class:`ExecutorErrorCode` (since 2.1.0).

Error codes are **API surface**: hosts (Geny, downstream CI runners,
log dashboards, Sentry grouping rules, frontend i18n keys) all depend
on the string values being stable across releases.

This test pins every shipped code's exact string value. Any rename,
re-purpose, or accidental delete fails CI before a release — forcing
a deliberate deprecation step (add a new code, mark the old one
deprecated) instead of a silent breaking change.

When you add a new code, add a row to ``_FROZEN`` below and to
``docs/error_codes.md``. When you intentionally retire a code (major
version bump), remove it from ``_FROZEN`` here AND from the enum AND
note the breaking change in CHANGELOG.
"""

from __future__ import annotations

from typing import Dict

import pytest

from geny_executor.core.errors import (
    APIError,
    ErrorCategory,
    ExecutorErrorCode,
    GenyExecutorError,
)


# ──────────────────────────────────────────────────────── frozen codes ─


# The canonical set of codes shipped in geny-executor ≥ 2.1.0.
# **Do not edit this dict to make a failing test pass.** If a code
# value here doesn't match the enum, fix the enum (you accidentally
# renamed a code) or — if you really mean to remove/rename a code —
# bump the major version and update both the enum and this dict.
_FROZEN: Dict[str, str] = {
    # exec.api.*
    "EXEC_API_AUTH_INVALID_KEY": "exec.api.auth.invalid_key",
    "EXEC_API_AUTH_EXPIRED": "exec.api.auth.expired",
    "EXEC_API_RATE_LIMITED": "exec.api.rate_limited",
    "EXEC_API_TIMEOUT": "exec.api.timeout",
    "EXEC_API_NETWORK": "exec.api.network",
    "EXEC_API_TOKEN_LIMIT": "exec.api.token_limit",
    "EXEC_API_BAD_REQUEST": "exec.api.bad_request",
    "EXEC_API_SERVER_ERROR": "exec.api.server_error",
    "EXEC_API_TERMINAL": "exec.api.terminal",
    "EXEC_API_UNKNOWN": "exec.api.unknown",
    "EXEC_API_NO_CLIENT": "exec.api.no_client",
    "EXEC_API_STREAM_INCOMPLETE": "exec.api.stream_incomplete",
    "EXEC_API_RETRY_EXHAUSTED": "exec.api.retry_exhausted",
    # exec.cli.*
    "EXEC_CLI_BINARY_NOT_FOUND": "exec.cli.binary_not_found",
    "EXEC_CLI_AUTH_FAILED": "exec.cli.auth_failed",
    "EXEC_CLI_TIMEOUT": "exec.cli.timeout",
    "EXEC_CLI_PROTOCOL_ERROR": "exec.cli.protocol_error",
    "EXEC_CLI_PERMISSION_DENIED": "exec.cli.permission_denied",
    "EXEC_CLI_EXITED": "exec.cli.exited",
    # exec.pipeline.* / exec.stage.*
    "EXEC_PIPELINE_NOT_INITIALIZED": "exec.pipeline.not_initialized",
    "EXEC_PIPELINE_INVALID_MANIFEST": "exec.pipeline.invalid_manifest",
    "EXEC_STAGE_FAILED": "exec.stage.failed",
    "EXEC_STAGE_GUARD_REJECTED": "exec.stage.guard_rejected",
    # exec.tool.*
    "EXEC_TOOL_UNKNOWN": "exec.tool.unknown",
    "EXEC_TOOL_INVALID_INPUT": "exec.tool.invalid_input",
    "EXEC_TOOL_ACCESS_DENIED": "exec.tool.access_denied",
    "EXEC_TOOL_CRASHED": "exec.tool.crashed",
    "EXEC_TOOL_TRANSPORT": "exec.tool.transport",
    # exec.mutation.*
    "EXEC_MUTATION_INVALID": "exec.mutation.invalid",
    "EXEC_MUTATION_LOCKED": "exec.mutation.locked",
    # exec.mcp.*
    "EXEC_MCP_CONNECT_FAILED": "exec.mcp.connect_failed",
    "EXEC_MCP_INITIALIZE_FAILED": "exec.mcp.initialize_failed",
    "EXEC_MCP_LIST_TOOLS_FAILED": "exec.mcp.list_tools_failed",
    "EXEC_MCP_SDK_MISSING": "exec.mcp.sdk_missing",
    # exec.unknown — fallback
    "EXEC_UNKNOWN": "exec.unknown",
}


def test_frozen_codes_match_enum_values_exactly() -> None:
    """Every code in ``_FROZEN`` must exist on the enum with the
    pinned string value. Catches accidental renames."""
    for member_name, expected_value in _FROZEN.items():
        member = getattr(ExecutorErrorCode, member_name, None)
        assert member is not None, (
            f"{member_name} disappeared from ExecutorErrorCode. "
            f"If this was intentional, bump the major version and "
            f"remove the row from _FROZEN."
        )
        assert member.value == expected_value, (
            f"ExecutorErrorCode.{member_name} = {member.value!r} but "
            f"_FROZEN pinned it to {expected_value!r}. "
            f"Renaming code strings is a breaking change — restore "
            f"the old value or bump the major version."
        )


def test_enum_has_no_codes_missing_from_frozen() -> None:
    """Every enum member must appear in ``_FROZEN``. Catches additions
    that weren't recorded in the stability pin — usually a forgotten
    docstring / docs/error_codes.md update."""
    enum_names = {m.name for m in ExecutorErrorCode}
    frozen_names = set(_FROZEN.keys())
    new = enum_names - frozen_names
    assert not new, (
        f"ExecutorErrorCode added member(s) not yet pinned in "
        f"_FROZEN: {sorted(new)}. Add a row to _FROZEN here AND a "
        f"row to docs/error_codes.md so the new code is documented."
    )


def test_all_code_values_match_canonical_format() -> None:
    """``exec.<component>.<reason>`` — lowercase, dot-separated,
    ≤4 segments, ASCII-only."""
    for code in ExecutorErrorCode:
        v = code.value
        assert v == v.lower(), f"{code.name} value {v!r} contains uppercase"
        assert v.startswith("exec."), f"{code.name} value {v!r} missing exec.* prefix"
        segments = v.split(".")
        assert 2 <= len(segments) <= 4, (
            f"{code.name} value {v!r} has {len(segments)} segments — "
            f"expected 2–4 (exec.<component>.<reason>[.<sub>])"
        )
        for seg in segments:
            assert seg, f"{code.name} value {v!r} has an empty segment"
            assert seg.replace("_", "").isalnum(), (
                f"{code.name} segment {seg!r} contains non-alphanumeric "
                f"chars beyond underscores"
            )


# ──────────────────────────────────── default category → code mapping ─


def test_every_error_category_has_a_default_code() -> None:
    """``ExecutorErrorCode.from_category()`` must return a meaningful
    code for every ``ErrorCategory`` value — never the generic fallback
    ``EXEC_UNKNOWN``. Otherwise legacy ``APIError(category=…)`` raises
    would all degrade to ``exec.unknown`` and hosts couldn't tell them
    apart."""
    for cat in ErrorCategory:
        code = ExecutorErrorCode.from_category(cat)
        assert code is not ExecutorErrorCode.EXEC_UNKNOWN, (
            f"ErrorCategory.{cat.name} has no specific default code — "
            f"add it to _CATEGORY_TO_CODE_DEFAULT in core/errors.py."
        )


# ──────────────────────────────────────────────────── exception wiring ─


def test_api_error_default_code_derives_from_category() -> None:
    """``APIError("...", category=ErrorCategory.CLI_AUTH_FAILED)`` —
    a legacy call site that pre-dates 2.1.0 — must still get a
    sensible ``code`` attribute without the caller having to thread
    one explicitly."""
    err = APIError("nope", category=ErrorCategory.CLI_AUTH_FAILED)
    assert err.code is ExecutorErrorCode.EXEC_CLI_AUTH_FAILED


def test_api_error_explicit_code_wins_over_category_default() -> None:
    """When a caller passes both ``category`` and ``code``, the
    explicit ``code`` is preserved — important for sites that want
    finer-grained classification than the broad category provides."""
    err = APIError(
        "rate limit retries exhausted",
        category=ErrorCategory.RATE_LIMITED,
        code=ExecutorErrorCode.EXEC_API_RETRY_EXHAUSTED,
    )
    assert err.code is ExecutorErrorCode.EXEC_API_RETRY_EXHAUSTED
    assert err.category is ErrorCategory.RATE_LIMITED


def test_base_exception_has_default_code() -> None:
    """``GenyExecutorError("...")`` with no code argument falls back
    to the subclass's ``_DEFAULT_CODE``, never ``None``. Downstream
    consumers can rely on ``e.code`` being set."""
    err = GenyExecutorError("test")
    assert err.code is ExecutorErrorCode.EXEC_UNKNOWN


@pytest.mark.parametrize(
    "kwargs, expected_code",
    [
        ({}, ExecutorErrorCode.EXEC_API_UNKNOWN),
        ({"category": ErrorCategory.CLI_NOT_FOUND}, ExecutorErrorCode.EXEC_CLI_BINARY_NOT_FOUND),
        ({"category": ErrorCategory.RATE_LIMITED}, ExecutorErrorCode.EXEC_API_RATE_LIMITED),
        ({"code": ExecutorErrorCode.EXEC_API_NO_CLIENT}, ExecutorErrorCode.EXEC_API_NO_CLIENT),
    ],
)
def test_api_error_code_resolution_matrix(
    kwargs: dict, expected_code: ExecutorErrorCode,
) -> None:
    """End-to-end: confirms the resolution rules
    (explicit code wins, otherwise category-derived, otherwise default)."""
    err = APIError("test", **kwargs)
    assert err.code is expected_code
