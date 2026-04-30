"""Phase 10.6 — killer bundled skills (simplify / skillify / loop).

These three are the higher-effort workflows the executor ships in
addition to the operational five from 10.4. Each has specific
metadata expectations the host catalog UI relies on.
"""

from __future__ import annotations

import pytest

from geny_executor.skills.bundled_skills import load_bundled_skills
from geny_executor.skills.types import Skill


@pytest.fixture(scope="module")
def loaded() -> dict:
    return {s.id: s for s in load_bundled_skills(strict=True).loaded}


def test_simplify_metadata(loaded) -> None:
    s: Skill = loaded["simplify"]
    assert s.metadata.category == "workflow"
    assert s.metadata.effort == "high"
    assert "Read" in s.metadata.allowed_tools
    assert "Bash" in s.metadata.allowed_tools
    # Body should describe a multi-pass review.
    body = s.body.lower()
    assert "reuse" in body
    assert "quality" in body
    assert "efficiency" in body
    # Expects a target arg.
    assert "target" in s.metadata.arguments


def test_skillify_metadata(loaded) -> None:
    s: Skill = loaded["skillify"]
    assert s.metadata.category == "meta"
    assert "Write" in s.metadata.allowed_tools
    # Body must mention the skill output path so the model knows
    # where to write.
    body = s.body.lower()
    assert "skill.md" in body
    # Self-referential safeguard: the body warns against runaway
    # interview loops.
    assert "stop" in body or "stop." in body


def test_loop_metadata(loaded) -> None:
    s: Skill = loaded["loop"]
    assert s.metadata.category == "workflow"
    assert "interval" in s.metadata.arguments
    assert "task" in s.metadata.arguments
    assert s.metadata.argument_hint is not None
    # Body must mention cron — it's the canonical translation target.
    assert "cron" in s.body.lower()


def test_killer_skills_are_user_invocable(loaded) -> None:
    for skill_id in ("simplify", "skillify", "loop"):
        s = loaded[skill_id]
        assert s.metadata.user_invocable is True
        assert s.metadata.disable_model_invocation is False


def test_killer_skills_have_when_to_use(loaded) -> None:
    """Higher-effort workflows are noisy if invoked at the wrong
    time — they all carry explicit ``when_to_use`` copy."""
    for skill_id in ("simplify", "skillify", "loop"):
        s = loaded[skill_id]
        assert s.metadata.when_to_use is not None
        assert len(s.metadata.when_to_use) > 30  # not just a label
