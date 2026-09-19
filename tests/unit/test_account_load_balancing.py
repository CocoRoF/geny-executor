"""Two accounts should share the work, not take turns being exhausted.

Failover is a reaction: it moves off an account once that account has
already hit its wall. For subscription plans that is backwards — the whole
reason to hold two Claude logins is that neither should reach its cap. A
route of healthy accounts sent EVERY turn to the first one until it 429'd,
and the second sat idle until then.

``balance`` orders the healthy hops least-recently-used. It is opt-in: a
route's order is a choice someone made, and silently round-robining it
would be a different product.
"""

from __future__ import annotations

import pytest

from geny_executor.llm_client import _failover as failover
from geny_executor.llm_client.router import RouterClient


def _targets(*ids: str) -> list[dict]:
    return [
        {"accountId": i, "engineProvider": "anthropic", "model": "claude-sonnet-5", "label": i}
        for i in ids
    ]


@pytest.fixture(autouse=True)
def _clean():
    for acct in ("a", "b", "c"):
        failover.clear_cooldown(acct)
        failover._LAST_USED.pop(acct, None)
    yield
    for acct in ("a", "b", "c"):
        failover.clear_cooldown(acct)
        failover._LAST_USED.pop(acct, None)


def _order(client: RouterClient) -> list[str]:
    return [client.targets[i]["accountId"] for i in client._order()]


class TestWithoutBalancing:
    def test_the_route_order_is_kept(self) -> None:
        """The default is unchanged: hop 0 answers until it cannot."""
        client = RouterClient(targets=_targets("a", "b", "c"))
        assert _order(client) == ["a", "b", "c"]
        client._chosen(0)
        assert _order(client) == ["a", "b", "c"], "a used account must not move"


class TestWithBalancing:
    def test_a_never_used_account_goes_first(self) -> None:
        client = RouterClient(targets=_targets("a", "b"), balance=True)
        client._chosen(0)
        assert _order(client)[0] == "b", "the idle account should take the next turn"

    def test_turns_alternate_instead_of_piling_onto_one_account(self) -> None:
        client = RouterClient(targets=_targets("a", "b"), balance=True)
        answered = []
        for _ in range(6):
            first = client._order()[0]
            client._chosen(first)
            answered.append(client.targets[first]["accountId"])
        assert answered.count("a") == 3 and answered.count("b") == 3, answered

    def test_route_position_breaks_ties(self) -> None:
        """Equal load is not a reason to shuffle: a deliberate ordering
        still holds among accounts that have done the same amount."""
        client = RouterClient(targets=_targets("a", "b", "c"), balance=True)
        assert _order(client) == ["a", "b", "c"]

    def test_a_cooling_account_is_still_last(self) -> None:
        """Balancing changes who is asked first among the healthy; it does
        not promote an account that is rate-limited."""
        client = RouterClient(targets=_targets("a", "b"), balance=True)
        failover.cool_down("b", 300, "usage limit")
        client._chosen(0)  # a is now the most recently used
        assert _order(client) == ["a", "b"], "a cooling hop must not be preferred"

    def test_a_failed_hop_does_not_count_as_its_turn(self) -> None:
        """Only the account that actually answered has taken a turn — a hop
        that was tried and failed over has not."""
        client = RouterClient(targets=_targets("a", "b"), balance=True)
        client._failed(0, RuntimeError("429 usage limit"), remaining=1)
        client._chosen(1)
        failover.clear_cooldown("a")
        assert _order(client)[0] == "a", "the hop that never answered should be next"


class TestTheLoadRecordIsShared:
    def test_two_agents_on_the_same_accounts_do_not_both_pick_first(self) -> None:
        """Sibling agents share the accounts, so the record has to be
        process-wide — a per-client counter would let every agent
        'balance' onto the same account."""
        one = RouterClient(targets=_targets("a", "b"), balance=True)
        two = RouterClient(targets=_targets("a", "b"), balance=True)
        one._chosen(one._order()[0])
        assert _order(two)[0] == "b", "the second agent repeated the first one's choice"


class TestPoolsNotTheWholeRoute:
    """A route is a preference; only interchangeable hops are a pool."""

    def _mixed(self) -> list[dict]:
        return [
            {"accountId": "claude1", "engineProvider": "geny_claude_code",
             "model": "sonnet", "label": "Claude 1"},
            {"accountId": "claude2", "engineProvider": "geny_claude_code",
             "model": "sonnet", "label": "Claude 2"},
            {"accountId": "codex1", "engineProvider": "geny_codex",
             "model": "gpt-5.6-terra", "label": "Codex"},
        ]

    def test_a_conversation_does_not_ping_pong_between_providers(self) -> None:
        """The failure this guards: blind round-robin over a route would
        answer as Claude, then as GPT, then as Claude — one agent visibly
        changing who it is every turn."""
        client = RouterClient(targets=self._mixed(), balance=True)
        answered = []
        try:
            for _ in range(6):
                first = client._order()[0]
                client._chosen(first)
                answered.append(client.targets[first]["engineProvider"])
        finally:
            for acct in ("claude1", "claude2", "codex1"):
                failover._LAST_USED.pop(acct, None)
        assert set(answered) == {"geny_claude_code"}, answered

    def test_the_two_equivalent_logins_still_share(self) -> None:
        client = RouterClient(targets=self._mixed(), balance=True)
        answered = []
        try:
            for _ in range(4):
                first = client._order()[0]
                client._chosen(first)
                answered.append(client.targets[first]["accountId"])
        finally:
            for acct in ("claude1", "claude2", "codex1"):
                failover._LAST_USED.pop(acct, None)
        assert answered.count("claude1") == 2 and answered.count("claude2") == 2, answered

    def test_the_fallback_pool_is_reached_when_the_primary_pool_is_cooling(self) -> None:
        client = RouterClient(targets=self._mixed(), balance=True)
        try:
            failover.cool_down("claude1", 300, "usage limit")
            failover.cool_down("claude2", 300, "usage limit")
            assert client.targets[client._order()[0]]["engineProvider"] == "geny_codex"
        finally:
            failover.clear_cooldown("claude1")
            failover.clear_cooldown("claude2")


class TestTheFlagReachesTheClient:
    """A host sets ``balance`` in the route's credential extras; the
    pipeline turns extras into constructor kwargs. If that chain breaks,
    balancing is silently off and nothing fails — the first account just
    quietly absorbs every turn again."""

    def test_extras_become_the_constructor_kwarg(self) -> None:
        from geny_executor.core.pipeline import _creds_to_client_kwargs
        from geny_executor.llm_client import ProviderCredentials

        creds = ProviderCredentials(
            extras={"targets": _targets("a", "b"), "balance": True}
        )
        kwargs = _creds_to_client_kwargs("geny_router", creds)
        assert kwargs["balance"] is True

        client = RouterClient(**kwargs)
        try:
            client._chosen(0)
            assert _order(client)[0] == "b", "balancing did not take effect"
        finally:
            for acct in ("a", "b"):
                failover._LAST_USED.pop(acct, None)

    def test_a_route_without_the_flag_keeps_its_order(self) -> None:
        from geny_executor.core.pipeline import _creds_to_client_kwargs
        from geny_executor.llm_client import ProviderCredentials

        creds = ProviderCredentials(extras={"targets": _targets("a", "b")})
        client = RouterClient(**_creds_to_client_kwargs("geny_router", creds))
        try:
            client._chosen(0)
            assert _order(client) == ["a", "b"]
        finally:
            for acct in ("a", "b"):
                failover._LAST_USED.pop(acct, None)
