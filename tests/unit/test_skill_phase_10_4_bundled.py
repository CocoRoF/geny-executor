"""Phase 10.4 — bundled skill catalog.

Validates the executor's shipped operational skills load cleanly and
each carries the metadata callers will rely on. The skills
themselves are markdown — these tests are the contract that says
"this is what we ship".
"""

from __future__ import annotations

from typing import Set

import pytest

from geny_executor.skills.bundled_skills import (
    bundled_skill_ids,
    bundled_skills_dir,
    load_bundled_skills,
)
from geny_executor.skills.registry import SkillRegistry
from geny_executor.skills.types import Skill


# ── Catalog inventory ───────────────────────────────────────────────


# The locked set of skills that ship with 1.6.x. Adding to this
# requires bumping the patch version *and* updating the test;
# removing requires a deprecation note in CHANGELOG.
EXPECTED_BUNDLED: Set[str] = {
    "verify",
    "debug",
    "lorem-ipsum",
    "stuck",
    "batch",
}


def test_bundled_dir_exists() -> None:
    assert bundled_skills_dir().is_dir()


def test_bundled_skill_ids_match_expected() -> None:
    """Locked inventory — adding / removing a bundled skill must
    update this test deliberately."""
    assert set(bundled_skill_ids()) == EXPECTED_BUNDLED


def test_load_bundled_skills_returns_each_id() -> None:
    report = load_bundled_skills()
    assert report.errors == []
    loaded_ids = {s.id for s in report.loaded}
    assert loaded_ids == EXPECTED_BUNDLED


# ── Per-skill metadata ──────────────────────────────────────────────


@pytest.fixture(scope="module")
def loaded() -> dict:
    """Load once per module; index by id."""
    report = load_bundled_skills(strict=True)
    return {s.id: s for s in report.loaded}


def test_lorem_ipsum_metadata(loaded) -> None:
    s: Skill = loaded["lorem-ipsum"]
    assert s.metadata.category == "utility"
    assert s.metadata.effort == "low"
    assert "count" in s.metadata.arguments
    assert "style" in s.metadata.arguments
    assert s.metadata.when_to_use is not None
    assert s.metadata.argument_hint is not None
    # No shell blocks — pure prompt template.
    assert "```!" not in s.body
    assert "!`" not in s.body


def test_verify_metadata(loaded) -> None:
    s: Skill = loaded["verify"]
    assert s.metadata.category == "diagnostic"
    assert s.metadata.shell == "bash"
    assert s.metadata.shell_timeout_s == 15.0
    # Verify uses shell blocks.
    assert "```!" in s.body or "!`" in s.body


def test_debug_metadata(loaded) -> None:
    s: Skill = loaded["debug"]
    assert s.metadata.category == "diagnostic"
    assert s.metadata.shell_timeout_s == 20.0
    assert "```!" in s.body or "!`" in s.body


def test_stuck_metadata(loaded) -> None:
    s: Skill = loaded["stuck"]
    assert s.metadata.category == "meta"
    assert s.metadata.effort == "low"
    # Stuck is pure prompt — explicitly no shell blocks (the skill
    # tells the model to stop calling tools).
    assert "```!" not in s.body
    assert "!`" not in s.body


def test_batch_metadata(loaded) -> None:
    s: Skill = loaded["batch"]
    assert s.metadata.category == "workflow"
    assert "items" in s.metadata.arguments
    assert "operation" in s.metadata.arguments


# ── Registration roundtrip ──────────────────────────────────────────


def test_bundled_skills_register_into_registry() -> None:
    """End-to-end: load the catalog and register every skill into a
    fresh SkillRegistry. No collisions, no malformed names."""
    report = load_bundled_skills(strict=True)
    registry = SkillRegistry()
    registry.register_many(report.loaded)
    assert len(registry) == len(EXPECTED_BUNDLED)
    for skill_id in EXPECTED_BUNDLED:
        assert registry.get(skill_id) is not None


def test_bundled_skill_ids_alphabetical() -> None:
    """Convention: directory iteration is alphabetised so the catalog
    listing is stable across platforms."""
    ids = bundled_skill_ids()
    assert ids == sorted(ids)


# ── No bundled skill is invocation-locked ───────────────────────────


def test_bundled_skills_are_user_invocable_by_default(loaded) -> None:
    """All ops bundled skills should be reachable via the user's
    slash-command palette. None should set ``user_invocable: false``
    by default — that's a flag for host-customised lockdowns."""
    for skill in loaded.values():
        assert skill.metadata.user_invocable is True
        assert skill.metadata.disable_model_invocation is False
