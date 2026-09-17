"""Which failures another account can survive.

This is the judgement the whole route rests on. Get it wrong in one direction
and a turn dies on an account that simply ran out of money while a working one
sits next in the list; wrong in the other and a malformed request is replayed
against every account the user owns.

The first direction is not hypothetical. A production Claude subscription
answered ``You've hit your monthly spend limit`` — wording that matched none
of the rate-limit patterns — so it was read as a bad request and the turn
failed outright with an OpenAI account waiting behind it.
"""

from __future__ import annotations

import pytest

from geny_executor.core.errors import APIError, ErrorCategory
from geny_executor.llm_client import _failover as failover


@pytest.mark.parametrize("message", [
    # The one that cost a turn
    "You've hit your monthly spend limit · raise it at claude.ai/settings/usage?from=cc_cli_limit_message",
    "Claude usage limit reached. Your limit will reset at 5pm.",
    "Your credit balance is too low to access the Anthropic API",
    "insufficient_quota: You exceeded your current quota, please check your plan and billing details",
    "Billing hard limit has been reached",
    "Rate limit exceeded",
    "Too many requests",
    "The engine is currently overloaded, please try again later",
])
def test_an_account_that_cannot_pay_is_a_rate_limit(message: str) -> None:
    """To a router these are one thing: the account is alive, the request is
    fine, and the next hop should take it."""
    assert failover.classify_text(message) == ErrorCategory.RATE_LIMITED


@pytest.mark.parametrize("message", [
    "Not logged in. Please run /login.",
    "Invalid API key provided",
    "OAuth token has expired",
    "invalid_grant",
])
def test_a_dead_login_is_auth(message: str) -> None:
    assert failover.classify_text(message) == ErrorCategory.AUTH


@pytest.mark.parametrize("message", [
    "schema validation failed for tool input",
    "prompt is too long: 250000 tokens > 200000 maximum",
    "unknown model 'claude-sonnet-4-6'",
])
def test_the_requests_own_fault_is_not_a_reason_to_try_elsewhere(message: str) -> None:
    """Replaying these against every account the user owns would burn all of
    them on a request that cannot succeed anywhere."""
    assert failover.classify_text(message) != ErrorCategory.RATE_LIMITED
    assert failover.classify_text(message) != ErrorCategory.AUTH


class TestStatusCodesWin:
    def test_429_is_a_rate_limit_whatever_it_says(self) -> None:
        assert failover.classify_text("something went wrong", 429) == ErrorCategory.RATE_LIMITED

    def test_401_is_auth(self) -> None:
        assert failover.classify_text("nope", 401) == ErrorCategory.AUTH

    def test_500_is_the_server(self) -> None:
        assert failover.classify_text("boom", 500) == ErrorCategory.SERVER_ERROR

    def test_a_plain_400_is_the_requests_fault(self) -> None:
        assert failover.classify_text("bad", 400) == ErrorCategory.BAD_REQUEST


class TestWhatFailsOver:
    @pytest.mark.parametrize("category", [
        ErrorCategory.RATE_LIMITED,
        ErrorCategory.AUTH,
        ErrorCategory.SERVER_ERROR,
        ErrorCategory.NETWORK,
        ErrorCategory.TIMEOUT,
        ErrorCategory.CLI_NOT_FOUND,
        ErrorCategory.CLI_TIMEOUT,
    ])
    def test_another_account_may_well_succeed(self, category: ErrorCategory) -> None:
        assert category in failover.FAILOVER

    @pytest.mark.parametrize("category", [
        ErrorCategory.BAD_REQUEST,
        ErrorCategory.TOKEN_LIMIT,
    ])
    def test_another_account_would_fail_the_same_way(self, category: ErrorCategory) -> None:
        assert category not in failover.FAILOVER


class TestFromAnException:
    def test_a_spend_limit_raised_as_an_api_error_still_fails_over(self) -> None:
        """The CLI client raises it as an APIError it could not classify
        itself; the router must not inherit that guess."""
        error = APIError(
            "Claude Code [personal]: You've hit your monthly spend limit",
            category=ErrorCategory.UNKNOWN,
        )
        assert failover.category_of(error) == ErrorCategory.RATE_LIMITED

    def test_a_cli_auth_failure_is_an_auth_failure(self) -> None:
        error = APIError("not logged in", category=ErrorCategory.CLI_AUTH_FAILED)
        assert failover.category_of(error) == ErrorCategory.AUTH

    def test_an_already_classified_error_is_believed(self) -> None:
        error = APIError("whatever", category=ErrorCategory.RATE_LIMITED)
        assert failover.category_of(error) == ErrorCategory.RATE_LIMITED
