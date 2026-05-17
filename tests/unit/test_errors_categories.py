"""Tests for ErrorCategory after Phase A1 — new CLI categories + helpers."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor.core.errors import APIError, ErrorCategory


# ---------------------------------------------------------------------------
# Enum membership
# ---------------------------------------------------------------------------


def test_new_cli_categories_exist() -> None:
    for name in (
        "CLI_NOT_FOUND",
        "CLI_AUTH_FAILED",
        "CLI_TIMEOUT",
        "CLI_PROTOCOL_ERROR",
        "CLI_PERMISSION_DENIED",
    ):
        assert hasattr(ErrorCategory, name), name


def test_existing_categories_still_present() -> None:
    for name in (
        "RATE_LIMITED",
        "TIMEOUT",
        "NETWORK",
        "TOKEN_LIMIT",
        "AUTH",
        "BAD_REQUEST",
        "SERVER_ERROR",
        "TERMINAL",
        "UNKNOWN",
    ):
        assert hasattr(ErrorCategory, name), name


# ---------------------------------------------------------------------------
# is_recoverable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cat",
    [
        ErrorCategory.RATE_LIMITED,
        ErrorCategory.TIMEOUT,
        ErrorCategory.NETWORK,
        ErrorCategory.SERVER_ERROR,
        ErrorCategory.CLI_TIMEOUT,
        ErrorCategory.CLI_PROTOCOL_ERROR,
    ],
)
def test_is_recoverable_true(cat: ErrorCategory) -> None:
    assert cat.is_recoverable is True


@pytest.mark.parametrize(
    "cat",
    [
        ErrorCategory.AUTH,
        ErrorCategory.BAD_REQUEST,
        ErrorCategory.TOKEN_LIMIT,
        ErrorCategory.TERMINAL,
        ErrorCategory.UNKNOWN,
        ErrorCategory.CLI_NOT_FOUND,
        ErrorCategory.CLI_AUTH_FAILED,
        ErrorCategory.CLI_PERMISSION_DENIED,
    ],
)
def test_is_recoverable_false(cat: ErrorCategory) -> None:
    assert cat.is_recoverable is False


# ---------------------------------------------------------------------------
# is_fatal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cat",
    [
        ErrorCategory.AUTH,
        ErrorCategory.BAD_REQUEST,
        ErrorCategory.CLI_NOT_FOUND,
        ErrorCategory.CLI_AUTH_FAILED,
        ErrorCategory.CLI_PERMISSION_DENIED,
    ],
)
def test_is_fatal_true(cat: ErrorCategory) -> None:
    assert cat.is_fatal is True


@pytest.mark.parametrize(
    "cat",
    [
        ErrorCategory.RATE_LIMITED,
        ErrorCategory.TIMEOUT,
        ErrorCategory.NETWORK,
        ErrorCategory.SERVER_ERROR,
        ErrorCategory.CLI_TIMEOUT,
        ErrorCategory.CLI_PROTOCOL_ERROR,
        ErrorCategory.TOKEN_LIMIT,
        ErrorCategory.TERMINAL,
        ErrorCategory.UNKNOWN,
    ],
)
def test_is_fatal_false(cat: ErrorCategory) -> None:
    assert cat.is_fatal is False


# ---------------------------------------------------------------------------
# APIError wraps the new categories
# ---------------------------------------------------------------------------


def test_apierror_can_carry_cli_category() -> None:
    e = APIError("claude not on PATH", category=ErrorCategory.CLI_NOT_FOUND)
    assert e.category is ErrorCategory.CLI_NOT_FOUND
    assert e.category.is_fatal is True
    assert e.category.is_recoverable is False


def test_apierror_default_category_unchanged() -> None:
    e = APIError("?")
    assert e.category is ErrorCategory.UNKNOWN
