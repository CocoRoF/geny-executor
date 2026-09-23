"""The token estimate, as measured against Claude — and the cut that uses it.

``len // 4`` read a Korean conversation at about a quarter of its size
(Claude spends roughly one token per Hangul syllable), so compaction fired
late and any budget stated in characters meant different things in different
languages. The estimator now prices character-class runs with weights fitted
to differential ``usage`` measurements; these tests pin the properties the
rest of the pipeline relies on, not the fitted constants.
"""

from __future__ import annotations

import random

import pytest

from geny_executor.core.token_estimate import chars_within_tokens, estimate_text_tokens


class TestTheEstimate:
    def test_korean_costs_about_a_token_per_syllable(self) -> None:
        text = "오늘 회의에서 결정된 사항을 정리해서 공유드립니다" * 10
        syllables = sum(1 for ch in text if "가" <= ch <= "힣")
        assert 1.2 * syllables <= estimate_text_tokens(text) <= 1.8 * syllables

    def test_english_costs_far_less_per_character(self) -> None:
        text = "The meeting decided to ship the release on Friday. " * 10
        assert len(text) / 6 <= estimate_text_tokens(text) <= len(text) / 3

    def test_the_old_ratio_is_gone(self) -> None:
        """The same character count is not the same cost."""
        ko, en = "가" * 1_000, "a" * 1_000
        assert estimate_text_tokens(ko) > 3 * estimate_text_tokens(en)

    def test_empty_is_zero(self) -> None:
        assert estimate_text_tokens("") == 0


class TestTheCut:
    """``chars_within_tokens`` finds the cut in one pass; what makes that
    safe is that the piece it names never prices over the budget."""

    ALPHABET = list("abcXYZ012  \n\n,.{}\"가나다漢字é€😀\t") + [
        "hello ", "세계 ", "  ", "12345", "\n",
    ]

    @pytest.mark.parametrize("from_end", [False, True])
    def test_the_piece_never_costs_more_than_asked(self, from_end: bool) -> None:
        rng = random.Random(7)
        for _ in range(2_000):
            text = "".join(rng.choice(self.ALPHABET) for _ in range(rng.randint(0, 200)))
            tokens = rng.randint(0, 200)
            n = chars_within_tokens(text, tokens, from_end=from_end)
            piece = text[len(text) - n :] if from_end else text[:n]
            assert estimate_text_tokens(piece) <= tokens, (text, tokens, n)

    @pytest.mark.parametrize("from_end", [False, True])
    def test_and_it_is_not_needlessly_short(self, from_end: bool) -> None:
        """Within a character of the longest piece that fits: a lone space
        prices at zero, so "one more" can occasionally still fit for free."""
        rng = random.Random(11)
        for _ in range(2_000):
            text = "".join(rng.choice(self.ALPHABET) for _ in range(rng.randint(1, 200)))
            tokens = rng.randint(1, 200)
            n = chars_within_tokens(text, tokens, from_end=from_end)
            if n >= len(text) - 1:
                continue
            two_more = text[len(text) - n - 2 :] if from_end else text[: n + 2]
            assert estimate_text_tokens(two_more) > tokens, (text, tokens, n)

    def test_a_text_that_fits_is_taken_whole(self) -> None:
        text = "짧은 문장입니다."
        assert chars_within_tokens(text, 1_000) == len(text)

    def test_the_same_budget_takes_fewer_korean_characters(self) -> None:
        assert chars_within_tokens("가" * 10_000, 1_000) < chars_within_tokens("a" * 10_000, 1_000)

    def test_zero_budget_takes_nothing(self) -> None:
        assert chars_within_tokens("anything", 0) == 0
