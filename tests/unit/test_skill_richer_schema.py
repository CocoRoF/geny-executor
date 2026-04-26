"""Richer SKILL.md schema tests (PR-B.4.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from geny_executor.skills.loader import SkillLoadError, parse_skill_file


def _write_skill(tmp_path: Path, *, body_extras: str = "") -> Path:
    skill_dir = tmp_path / "test-skill"
    skill_dir.mkdir(exist_ok=True)
    md = skill_dir / "SKILL.md"
    md.write_text(
        f"---\n"
        f"name: Test\n"
        f"description: A test skill\n"
        f"{body_extras}"
        f"---\n\n"
        f"body content here\n",
        encoding="utf-8",
    )
    return md


# ── New fields ───────────────────────────────────────────────────────


class TestRicherSchema:
    def test_category_loaded(self, tmp_path: Path):
        path = _write_skill(tmp_path, body_extras="category: writing\n")
        skill = parse_skill_file(path)
        assert skill.metadata.category == "writing"

    def test_category_absent_is_none(self, tmp_path: Path):
        path = _write_skill(tmp_path)
        skill = parse_skill_file(path)
        assert skill.metadata.category is None

    def test_effort_loaded(self, tmp_path: Path):
        path = _write_skill(tmp_path, body_extras="effort: medium\n")
        skill = parse_skill_file(path)
        assert skill.metadata.effort == "medium"

    def test_examples_list(self, tmp_path: Path):
        path = _write_skill(
            tmp_path,
            body_extras="examples:\n  - 'do X'\n  - 'do Y'\n",
        )
        skill = parse_skill_file(path)
        assert skill.metadata.examples == ("do X", "do Y")

    def test_examples_single_string(self, tmp_path: Path):
        path = _write_skill(tmp_path, body_extras="examples: 'just one'\n")
        skill = parse_skill_file(path)
        assert skill.metadata.examples == ("just one",)

    def test_examples_invalid_type_rejected(self, tmp_path: Path):
        path = _write_skill(
            tmp_path,
            body_extras="examples:\n  - 'good'\n  - 42\n",
        )
        with pytest.raises(SkillLoadError):
            parse_skill_file(path)

    def test_empty_string_fields_become_none(self, tmp_path: Path):
        path = _write_skill(
            tmp_path,
            body_extras="category: ''\neffort: ''\n",
        )
        skill = parse_skill_file(path)
        assert skill.metadata.category is None
        assert skill.metadata.effort is None

    def test_unknown_keys_still_in_extras(self, tmp_path: Path):
        path = _write_skill(
            tmp_path,
            body_extras="category: x\nmy_custom_key: hello\n",
        )
        skill = parse_skill_file(path)
        # category is consumed, my_custom_key falls through.
        assert "my_custom_key" in skill.metadata.extras
        assert "category" not in skill.metadata.extras


# ── Backward compat ──────────────────────────────────────────────────


def test_old_skill_md_still_loads(tmp_path: Path):
    """A SKILL.md from before PR-B.4.1 with no new fields must still load."""
    path = _write_skill(tmp_path)
    skill = parse_skill_file(path)
    # All new fields default cleanly.
    assert skill.metadata.category is None
    assert skill.metadata.effort is None
    assert skill.metadata.examples == ()
