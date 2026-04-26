"""FrequencyPolicy tests (PR-B.2.1)."""

from __future__ import annotations

from typing import Optional

import pytest

from geny_executor.stages.s19_summarize import (
    EveryNTurnsPolicy,
    FrequencyAwareSummarizerProxy,
    FrequencyContext,
    NeverPolicy,
    NoSummarizer,
    OnContextFillPolicy,
    RuleBasedSummarizer,
)
from geny_executor.stages.s19_summarize.interface import Summarizer


def _ctx(
    iteration=1,
    last_fired=None,
    used=0,
    max_ctx=200_000,
):
    return FrequencyContext(
        iteration=iteration,
        last_fired_iteration=last_fired,
        input_tokens=used // 2,
        output_tokens=used - (used // 2),
        max_context_tokens=max_ctx,
    )


# ── NeverPolicy / EveryNTurnsPolicy ──────────────────────────────────


class TestNever:
    def test_never_fires(self):
        assert NeverPolicy().should_fire(_ctx(iteration=100)) is False


class TestEveryN:
    def test_invalid_n_rejected(self):
        with pytest.raises(ValueError):
            EveryNTurnsPolicy(0)

    def test_iteration_zero_no_fire(self):
        assert EveryNTurnsPolicy(5).should_fire(_ctx(iteration=0)) is False

    def test_fires_at_multiples(self):
        p = EveryNTurnsPolicy(5)
        assert p.should_fire(_ctx(iteration=5)) is True
        assert p.should_fire(_ctx(iteration=10)) is True
        assert p.should_fire(_ctx(iteration=4)) is False
        assert p.should_fire(_ctx(iteration=11)) is False


# ── OnContextFill ────────────────────────────────────────────────────


class TestOnContextFill:
    def test_invalid_threshold_rejected(self):
        with pytest.raises(ValueError):
            OnContextFillPolicy(threshold=0)
        with pytest.raises(ValueError):
            OnContextFillPolicy(threshold=1.5)

    def test_invalid_min_between_rejected(self):
        with pytest.raises(ValueError):
            OnContextFillPolicy(min_turns_between=-1)

    def test_zero_max_context_no_fire(self):
        # division-by-zero guard
        assert OnContextFillPolicy().should_fire(_ctx(used=100, max_ctx=0)) is False

    def test_below_threshold(self):
        p = OnContextFillPolicy(threshold=0.8)
        # 50% used → no fire
        assert p.should_fire(_ctx(used=100_000, max_ctx=200_000)) is False

    def test_at_or_above_threshold(self):
        p = OnContextFillPolicy(threshold=0.8, min_turns_between=0)
        assert p.should_fire(_ctx(used=160_000, max_ctx=200_000)) is True
        assert p.should_fire(_ctx(used=180_000, max_ctx=200_000)) is True

    def test_min_turns_between_blocks(self):
        p = OnContextFillPolicy(threshold=0.8, min_turns_between=5)
        # 80% full but only 2 turns since last fire → blocked
        ctx = _ctx(iteration=10, last_fired=8, used=160_000)
        assert p.should_fire(ctx) is False
        # 5 turns since last → allowed
        ctx = _ctx(iteration=15, last_fired=10, used=160_000)
        assert p.should_fire(ctx) is True


# ── FrequencyAwareSummarizerProxy ────────────────────────────────────


_SENTINEL = object()


class _MockSummarizer(Summarizer):
    def __init__(self, response=_SENTINEL):
        self.response = "summary" if response is _SENTINEL else response
        self.calls = 0

    @property
    def name(self) -> str:
        return "mock"

    @property
    def description(self) -> str:
        return "mock"

    async def summarize(self, state):
        self.calls += 1
        return self.response


class _FakeState:
    def __init__(self, iteration=0, used=0, max_ctx=200_000):
        self.iteration = iteration
        self.session_id = "s1"
        self.messages = []
        self.shared = {
            "input_tokens": used // 2,
            "output_tokens": used - (used // 2),
            "max_context_tokens": max_ctx,
        }


class TestProxy:
    @pytest.mark.asyncio
    async def test_proxy_skips_when_policy_says_no(self):
        inner = _MockSummarizer()
        proxy = FrequencyAwareSummarizerProxy(inner, NeverPolicy())
        result = await proxy.summarize(_FakeState(iteration=5))
        assert result is None
        assert inner.calls == 0

    @pytest.mark.asyncio
    async def test_proxy_calls_inner_when_policy_says_yes(self):
        inner = _MockSummarizer(response="hi")
        proxy = FrequencyAwareSummarizerProxy(inner, EveryNTurnsPolicy(1))
        result = await proxy.summarize(_FakeState(iteration=1))
        assert result == "hi"
        assert inner.calls == 1

    @pytest.mark.asyncio
    async def test_proxy_tracks_last_fired(self):
        inner = _MockSummarizer(response="hi")
        proxy = FrequencyAwareSummarizerProxy(
            inner, OnContextFillPolicy(threshold=0.8, min_turns_between=3),
        )
        # Iteration 1: fires (no last_fired yet)
        await proxy.summarize(_FakeState(iteration=1, used=200_000))
        assert inner.calls == 1
        # Iteration 2: blocked (only 1 turn since)
        await proxy.summarize(_FakeState(iteration=2, used=200_000))
        assert inner.calls == 1
        # Iteration 4: fires (3 turns since)
        await proxy.summarize(_FakeState(iteration=4, used=200_000))
        assert inner.calls == 2

    @pytest.mark.asyncio
    async def test_proxy_does_not_stamp_on_none_result(self):
        inner = _MockSummarizer(response=None)
        proxy = FrequencyAwareSummarizerProxy(inner, EveryNTurnsPolicy(1))
        await proxy.summarize(_FakeState(iteration=1))
        # Inner returned None → no stamp → next call not blocked.
        assert proxy._last_fired_iteration is None

    def test_proxy_name_includes_inner(self):
        inner = NoSummarizer()
        proxy = FrequencyAwareSummarizerProxy(inner, NeverPolicy())
        assert "no_summary" in proxy.name
