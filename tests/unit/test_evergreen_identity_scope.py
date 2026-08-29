"""``## Identity`` records who the agent IS, not what it is like.

The evergreen is distilled from transcripts, so "identity" drifted into a
description of how the agent had been behaving — temperament, tone,
manner. That reads as a durable fact and is injected every turn, so when
a host later changed the agent's configured character the memory went on
asserting the old one, in the user's own words, with more specificity and
more recency than the character section far above it.

Observed 2026-08-29: a session moved onto a warm, playful ESFP preset and
kept answering "차갑고 조용해. 감정 잘 안 드러내" — quoting its own
evergreen line, "cold analytical clinical counselor,
structure-before-empathy-FIXED".

Character is configuration. A transcript is a record of how the agent
behaved, not a rule for how it must behave.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor.memory.rollup import (
    build_evergreen_instruction,
    build_evergreen_instruction_structured,
)


BUILDERS = (build_evergreen_instruction, build_evergreen_instruction_structured)


@pytest.mark.parametrize("build", BUILDERS)
def test_the_merge_is_told_identity_is_not_a_personality(build):
    text = build(current="(none yet)", recent_digest="(none)", max_chars=4000)
    lowered = text.lower()
    assert "not a personality description" in lowered
    # The words it must actually rule out, named so the model can match them.
    for word in ("temperament", "tone", "warmth", "manner"):
        assert word in lowered, f"the rule never mentions {word}"


@pytest.mark.parametrize("build", BUILDERS)
def test_it_says_what_identity_IS_for(build):
    """A prohibition with no positive scope invites the model to drop the
    name and the role too."""
    lowered = build(current="", recent_digest="", max_chars=4000).lower()
    assert "your name" in lowered
    assert "role" in lowered
    assert "taboo" in lowered


@pytest.mark.parametrize("build", BUILDERS)
def test_existing_lines_are_dropped_not_merely_not_added(build):
    """A vault already carrying "cold analytical" must shed it on the next
    merge, not keep it forever because the rule only covered new writes."""
    lowered = build(current="- Ellen — cold analytical",
                    recent_digest="", max_chars=4000).lower()
    assert "drop such lines when merging" in lowered


@pytest.mark.parametrize("build", BUILDERS)
def test_the_preserve_clause_still_stands(build):
    """Scoping identity must not weaken the rule that durable facts —
    names, preferences, commitments — are never lost."""
    lowered = build(current="", recent_digest="", max_chars=4000).lower()
    assert "never lose" in lowered
    assert "preference" in lowered
