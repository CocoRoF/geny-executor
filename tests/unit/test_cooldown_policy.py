"""A bench should last as long as the outage does — no longer, no shorter.

A flat fifteen minutes is a guess in both directions. Too short and the next
turn walks into the same wall; too long and an account that recovered five
minutes ago sits idle while its sibling carries everything. Three policies
close that gap, and each is a real failure someone has had:

  * the provider usually SAYS when to come back ("try again in 42 seconds",
    ``retry-after``) — believe it over a constant;
  * benching the ONLY account is not protection, it is downtime;
  * rate limits are per model, so a 429 on opus says nothing about haiku on
    the same login.
"""

from __future__ import annotations

import time

import pytest

from geny_executor.core.errors import ErrorCategory
from geny_executor.llm_client import _failover as failover
from geny_executor.llm_client.router import RouterClient


@pytest.fixture(autouse=True)
def _clean():
    for acct in ("a", "b"):
        failover.clear_cooldown(acct)
        failover.clear_model_cooldowns(acct)
        failover._LAST_USED.pop(acct, None)
    yield
    for acct in ("a", "b"):
        failover.clear_cooldown(acct)
        failover.clear_model_cooldowns(acct)
        failover._LAST_USED.pop(acct, None)


class TestTheProviderIsBelievedAboutWhenToComeBack:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Rate limited. Please try again in 42 seconds.", 42),
            ("Slow down — retry in 2 minutes", 120),
            ("retry-after: 90", 90),
            ("Quota exceeded, come back in 1 hour", 3600),
            ("usage limit reached, try again in 30s", 30),
        ],
    )
    def test_it_is_read_out_of_the_message(self, text: str, expected: int) -> None:
        """Vendors put this in the text far more often than in a field."""
        at = failover.reset_at_from(text)
        assert at is not None, f"no reset read from {text!r}"
        assert abs((at - time.time()) - expected) < 2

    @pytest.mark.parametrize(
        "fields",
        [
            {"reset_at": "2026-09-19T12:00:00Z"},
            {"resets_at": 1789819200},
            {"resets_at": 1789819200000},  # milliseconds
        ],
    )
    def test_an_explicit_field_outranks_the_text(self, fields: dict) -> None:
        at = failover.reset_at_from("try again in 5 seconds", fields=fields)
        assert at == pytest.approx(1789819200, abs=1)

    def test_silence_means_fall_back_to_the_category(self) -> None:
        """No guess dressed up as an answer."""
        assert failover.reset_at_from("Claude usage limit reached.") is None


class TestBenchingTheOnlyAccount:
    def test_a_throttle_is_a_short_sit_down(self) -> None:
        assert failover.cooldown_for(ErrorCategory.RATE_LIMITED) == failover.RATE_LIMIT_COOLDOWN_S
        assert (
            failover.cooldown_for(ErrorCategory.RATE_LIMITED, sole_account=True)
            == failover.SOLE_ACCOUNT_COOLDOWN_S
        )

    def test_a_dead_login_is_not_shortened(self) -> None:
        """Waiting cannot fix it, and retrying every minute only spams."""
        assert (
            failover.cooldown_for(ErrorCategory.AUTH, sole_account=True)
            == failover.AUTH_COOLDOWN_S
        )


class TestRateLimitsArePerModel:
    def _route(self, *models: str) -> list[dict]:
        return [
            {"accountId": "a", "engineProvider": "anthropic", "model": m, "label": m}
            for m in models
        ]

    def test_a_429_on_one_model_leaves_its_sibling_usable(self) -> None:
        client = RouterClient(targets=self._route("opus", "haiku"))
        client._failed(0, RuntimeError("429 rate limit exceeded"), remaining=1)

        order = [client.targets[i]["model"] for i in client._order()]
        assert order[0] == "haiku", "the sibling model was benched with its twin"
        assert failover.model_cooling("a", "opus") is not None
        assert failover.model_cooling("a", "haiku") is None

    def test_the_account_itself_is_not_benched(self) -> None:
        """Only the model. An account-wide bench would take every model out."""
        client = RouterClient(targets=self._route("opus", "haiku"))
        client._failed(0, RuntimeError("429 rate limit exceeded"), remaining=1)
        assert failover.cooling("a") is None

    def test_an_unscoped_caller_stays_conservative(self) -> None:
        """Not naming a model could land on the one that just failed."""
        failover.cool_down_model("a", "opus", time.time() + 300, "429")
        assert failover.model_cooling("a", None) is not None

    def test_an_auth_failure_still_benches_the_whole_account(self) -> None:
        """A dead login is not a property of one model."""
        client = RouterClient(targets=self._route("opus", "haiku"))
        client._failed(0, RuntimeError("Not logged in. Please run /login."), remaining=1)
        assert failover.cooling("a") is not None

    def test_a_success_clears_the_model_bench(self) -> None:
        client = RouterClient(targets=self._route("opus", "haiku"))
        client._failed(0, RuntimeError("429 rate limit exceeded"), remaining=1)
        client._chosen(0)
        assert failover.model_cooling("a", "opus") is None


class TestTheProviderSReleaseTimeWins:
    def test_a_short_retry_hint_beats_the_flat_bench(self) -> None:
        """The whole point: 42 seconds, not fifteen minutes."""
        client = RouterClient(
            targets=[
                {"accountId": "a", "engineProvider": "anthropic", "model": "opus"},
                {"accountId": "b", "engineProvider": "anthropic", "model": "opus"},
            ]
        )
        client._failed(0, RuntimeError("429 — please try again in 42 seconds"), remaining=1)
        benched = failover.model_cooling("a", "opus")
        assert benched is not None
        assert abs((benched[0] - time.time()) - 42) < 3, "the hint was ignored"


class TestNothingIsBenchedInvisibly:
    """``snapshot()`` is the one view of what cannot answer right now — the
    host renders it and callers clear from it. A bench missing from it is a
    bench nobody can see or lift, and it leaks into the next turn."""

    def test_a_model_bench_is_visible(self) -> None:
        failover.cool_down_model("a", "opus", time.time() + 300, "429")
        snap = failover.snapshot()
        assert "a" in snap, "a benched account vanished from the view"
        assert snap["a"]["scope"] == "model"
        assert "opus" in snap["a"]["models"]

    def test_clearing_an_account_lifts_its_model_benches_too(self) -> None:
        """'this account is fine now' is ONE fact — leaving a model bench
        behind is how an account stays half-benched with no explanation."""
        failover.cool_down("a", 300, "auth")
        failover.cool_down_model("a", "opus", time.time() + 300, "429")
        failover.clear_cooldown("a")
        assert failover.snapshot() == {}

    def test_an_account_bench_and_a_model_bench_coexist(self) -> None:
        failover.cool_down("a", 300, "server error")
        failover.cool_down_model("a", "opus", time.time() + 120, "429")
        entry = failover.snapshot()["a"]
        assert entry["scope"] == "account"
        assert entry["models"]["opus"]["remaining"] == pytest.approx(120, abs=2)


class TestSubscriptionsMeterTheWholeAccount:
    """The distinction that makes per-model benching safe: a metered API
    limits per model, a subscription limits the plan. Getting this backwards
    sends the next turn into the same wall on a sibling model."""

    @pytest.mark.parametrize("provider", ["geny_claude_code", "geny_codex"])
    def test_a_plan_usage_limit_benches_the_account(self, provider: str) -> None:
        client = RouterClient(
            targets=[
                {"accountId": "a", "engineProvider": provider, "model": "sonnet"},
                {"accountId": "b", "engineProvider": provider, "model": "sonnet"},
            ]
        )
        client._failed(0, RuntimeError("Claude usage limit reached."), remaining=1)
        assert failover.cooling("a") is not None, "a plan limit was scoped to one model"

    @pytest.mark.parametrize("provider", ["anthropic", "openai"])
    def test_a_metered_api_benches_the_model(self, provider: str) -> None:
        client = RouterClient(
            targets=[
                {"accountId": "a", "engineProvider": provider, "model": "opus"},
                {"accountId": "a", "engineProvider": provider, "model": "haiku"},
            ]
        )
        client._failed(0, RuntimeError("429 rate limit exceeded"), remaining=1)
        assert failover.cooling("a") is None
        assert failover.model_cooling("a", "opus") is not None


class TestTheHintSurvivesRewording:
    """A machine-readable answer — Codex's ``resets_in_seconds``, a
    ``Retry-After`` header — rides on the exception, not only in the message.
    Carrying it only in the text means the next person who rewords an error
    string quietly turns the hint back into a guess."""

    def test_an_attached_retry_after_is_used(self) -> None:
        from geny_executor.llm_client.router import _retry_fields

        exc = RuntimeError("HTTP 429: the usage limit has been reached")
        exc.retry_after = 90  # type: ignore[attr-defined]
        at = failover.reset_at_from(str(exc), fields=_retry_fields(exc))
        assert at is not None and abs((at - time.time()) - 90) < 2

    def test_it_wins_over_a_number_in_the_text(self) -> None:
        from geny_executor.llm_client.router import _retry_fields

        exc = RuntimeError("try again in 5 seconds")
        exc.retry_after = 300  # type: ignore[attr-defined]
        at = failover.reset_at_from(str(exc), fields=_retry_fields(exc))
        assert abs((at or 0) - time.time() - 300) < 2

    def test_an_error_carrying_nothing_falls_back_to_the_text(self) -> None:
        from geny_executor.llm_client.router import _retry_fields

        exc = RuntimeError("please retry in 7 seconds")
        at = failover.reset_at_from(str(exc), fields=_retry_fields(exc))
        assert at is not None and abs((at - time.time()) - 7) < 2


class TestTheCodexWireKeepsWhatItWasTold:
    def test_resets_in_seconds_reaches_the_error(self) -> None:
        """It was already read for a notification and then dropped — the one
        number that says exactly when the account is usable again."""
        import json

        from geny_executor.llm_client.codex import CodexResponsesClient

        client = CodexResponsesClient(api_key="k")
        body = json.dumps(
            {"error": {"message": "usage limit reached", "resets_in_seconds": 120}}
        )
        error = client._http_error(429, body)
        assert getattr(error, "retry_after", None) == 120

    def test_a_retry_after_header_reaches_the_error(self) -> None:
        from geny_executor.llm_client.codex import CodexResponsesClient

        client = CodexResponsesClient(api_key="k")
        error = client._http_error(429, "too many requests", {"retry-after": "45"})
        assert getattr(error, "retry_after", None) == 45

    def test_no_hint_attaches_nothing(self) -> None:
        from geny_executor.llm_client.codex import CodexResponsesClient

        client = CodexResponsesClient(api_key="k")
        error = client._http_error(429, "too many requests", {})
        assert getattr(error, "retry_after", None) is None
