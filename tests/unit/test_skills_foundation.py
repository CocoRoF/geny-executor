"""Phase 4 Week 7 — Skills foundation tests.

Covers the type system, frontmatter parser, loader, and registry.
SkillTool (the Tool wrapper) lands in Phase 4 Week 8 with its own tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from geny_executor.skills import (
    Skill,
    SkillContext,
    SkillLoadError,
    SkillLoadReport,
    SkillMetadata,
    SkillRegistry,
    load_skills_dir,
    parse_frontmatter,
    parse_skill_file,
    validate_execution_mode,
)


# ─────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────


class TestTypes:
    def test_metadata_shortcut_properties(self):
        meta = SkillMetadata(name="X", description="y")
        skill = Skill(id="x", metadata=meta, body="body")
        assert skill.name == "X"
        assert skill.description == "y"

    def test_skill_is_frozen(self):
        skill = Skill(id="x", metadata=SkillMetadata("X", "y"), body="")
        with pytest.raises(Exception):
            skill.id = "y"  # type: ignore[misc]

    def test_validate_execution_mode_ok(self):
        assert validate_execution_mode("inline") == "inline"
        assert validate_execution_mode("fork") == "fork"

    def test_validate_execution_mode_rejects_unknown(self):
        with pytest.raises(ValueError):
            validate_execution_mode("weird")

    def test_skill_context_fields(self):
        skill = Skill(id="x", metadata=SkillMetadata("X", "y"), body="")
        ctx = SkillContext(skill=skill, session_id="sess")
        assert ctx.skill is skill
        assert ctx.session_id == "sess"
        assert ctx.invoke_args == {}


# ─────────────────────────────────────────────────────────────────
# Frontmatter parser
# ─────────────────────────────────────────────────────────────────


class TestFrontmatter:
    def test_basic_parse(self):
        text = "---\nname: A\ndescription: B\n---\n\nBody text here.\n"
        meta, body = parse_frontmatter(text)
        assert meta == {"name": "A", "description": "B"}
        assert body == "Body text here.\n"

    def test_no_frontmatter_returns_original(self):
        text = "Just some text without delimiters.\n"
        meta, body = parse_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_opening_delimiter_without_closer_treated_as_body(self):
        text = "---\nname: A\n(no closing)\nMore text\n"
        meta, body = parse_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_invalid_yaml_treated_as_body(self):
        text = "---\nname: : : broken\n---\nBody\n"
        meta, body = parse_frontmatter(text)
        assert meta == {}

    def test_empty_frontmatter(self):
        text = "---\n---\nBody\n"
        meta, body = parse_frontmatter(text)
        assert meta == {}
        assert body == "Body\n"

    def test_complex_yaml_list(self):
        text = (
            "---\n"
            "name: X\n"
            "allowed_tools:\n"
            "  - Read\n"
            "  - Grep\n"
            "---\n"
            "body\n"
        )
        meta, body = parse_frontmatter(text)
        assert meta["allowed_tools"] == ["Read", "Grep"]

    def test_leading_whitespace_before_delimiter(self):
        text = "\n\n---\nname: A\ndescription: B\n---\nbody\n"
        meta, body = parse_frontmatter(text)
        assert meta == {"name": "A", "description": "B"}

    def test_non_dict_yaml_rejected(self):
        # Top-level YAML that's a list — not valid frontmatter
        text = "---\n- 1\n- 2\n---\nbody\n"
        meta, body = parse_frontmatter(text)
        assert meta == {}


# ─────────────────────────────────────────────────────────────────
# parse_skill_file
# ─────────────────────────────────────────────────────────────────


class TestParseSkillFile:
    def _write_skill(self, tmp_path: Path, dirname: str, contents: str) -> Path:
        skill_dir = tmp_path / dirname
        skill_dir.mkdir()
        path = skill_dir / "SKILL.md"
        path.write_text(contents, encoding="utf-8")
        return path

    def test_minimal_valid(self, tmp_path):
        path = self._write_skill(
            tmp_path,
            "refactor",
            "---\nname: Refactor\ndescription: Help refactor code\n---\nBody\n",
        )
        skill = parse_skill_file(path)
        assert skill.id == "refactor"
        assert skill.name == "Refactor"
        assert skill.description == "Help refactor code"
        assert skill.body == "Body\n"
        assert skill.source == path
        assert skill.metadata.execution_mode == "inline"  # default
        assert skill.metadata.allowed_tools == ()

    def test_full_metadata(self, tmp_path):
        path = self._write_skill(
            tmp_path,
            "heavy",
            "---\n"
            "name: Heavy thinker\n"
            "description: Deep reasoning\n"
            "version: 1.2.3\n"
            "allowed_tools:\n"
            "  - Read\n"
            "  - Grep\n"
            "model_override: claude-opus-4-7\n"
            "execution_mode: fork\n"
            "custom_key: custom_value\n"
            "---\nbody\n",
        )
        skill = parse_skill_file(path)
        assert skill.metadata.version == "1.2.3"
        assert skill.metadata.allowed_tools == ("Read", "Grep")
        assert skill.metadata.model_override == "claude-opus-4-7"
        assert skill.metadata.execution_mode == "fork"
        assert skill.metadata.extras == {"custom_key": "custom_value"}

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(SkillLoadError, match="not found"):
            parse_skill_file(tmp_path / "missing" / "SKILL.md")

    def test_missing_name_raises(self, tmp_path):
        path = self._write_skill(
            tmp_path, "s", "---\ndescription: d\n---\nbody\n"
        )
        with pytest.raises(SkillLoadError, match="'name' is required"):
            parse_skill_file(path)

    def test_missing_description_raises(self, tmp_path):
        path = self._write_skill(tmp_path, "s", "---\nname: n\n---\nbody\n")
        with pytest.raises(SkillLoadError, match="'description' is required"):
            parse_skill_file(path)

    def test_missing_frontmatter_raises(self, tmp_path):
        path = self._write_skill(tmp_path, "s", "just a markdown body\n")
        with pytest.raises(SkillLoadError, match="missing or invalid YAML"):
            parse_skill_file(path)

    def test_invalid_execution_mode_raises(self, tmp_path):
        path = self._write_skill(
            tmp_path,
            "s",
            "---\nname: n\ndescription: d\nexecution_mode: weird\n---\nbody\n",
        )
        with pytest.raises(SkillLoadError, match="execution_mode"):
            parse_skill_file(path)

    def test_non_list_allowed_tools_raises(self, tmp_path):
        path = self._write_skill(
            tmp_path,
            "s",
            "---\nname: n\ndescription: d\nallowed_tools: not_a_list\n---\nbody\n",
        )
        with pytest.raises(SkillLoadError, match="allowed_tools"):
            parse_skill_file(path)

    def test_non_string_allowed_tool_entry_raises(self, tmp_path):
        path = self._write_skill(
            tmp_path,
            "s",
            "---\n"
            "name: n\n"
            "description: d\n"
            "allowed_tools:\n"
            "  - Read\n"
            "  - 42\n"
            "---\nbody\n",
        )
        with pytest.raises(SkillLoadError, match="allowed_tools"):
            parse_skill_file(path)

    def test_empty_model_override_treated_as_none(self, tmp_path):
        path = self._write_skill(
            tmp_path,
            "s",
            "---\nname: n\ndescription: d\nmodel_override: '   '\n---\nbody\n",
        )
        skill = parse_skill_file(path)
        assert skill.metadata.model_override is None

    def test_non_string_model_override_raises(self, tmp_path):
        path = self._write_skill(
            tmp_path,
            "s",
            "---\nname: n\ndescription: d\nmodel_override: 42\n---\nbody\n",
        )
        with pytest.raises(SkillLoadError, match="model_override"):
            parse_skill_file(path)


# ─────────────────────────────────────────────────────────────────
# load_skills_dir
# ─────────────────────────────────────────────────────────────────


class TestLoadSkillsDir:
    def _write_skill(self, root: Path, dirname: str, contents: str):
        skill_dir = root / dirname
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(contents, encoding="utf-8")

    def test_loads_every_valid_skill(self, tmp_path):
        self._write_skill(
            tmp_path, "a", "---\nname: A\ndescription: da\n---\nbody a\n"
        )
        self._write_skill(
            tmp_path, "b", "---\nname: B\ndescription: db\n---\nbody b\n"
        )
        report = load_skills_dir(tmp_path)
        assert isinstance(report, SkillLoadReport)
        assert [s.id for s in report.loaded] == ["a", "b"]
        assert report.errors == []

    def test_missing_dir_returns_empty_report(self, tmp_path):
        report = load_skills_dir(tmp_path / "does-not-exist")
        assert report.loaded == []
        assert report.errors == []

    def test_ignores_non_directory_entries(self, tmp_path):
        (tmp_path / "README.md").write_text("not a skill", encoding="utf-8")
        self._write_skill(
            tmp_path, "real", "---\nname: R\ndescription: d\n---\n"
        )
        report = load_skills_dir(tmp_path)
        assert [s.id for s in report.loaded] == ["real"]

    def test_ignores_dirs_without_skill_md(self, tmp_path):
        (tmp_path / "not-a-skill").mkdir()
        report = load_skills_dir(tmp_path)
        assert report.loaded == []
        assert report.errors == []

    def test_collects_errors_by_default(self, tmp_path, caplog):
        caplog.set_level("WARNING")
        # Valid
        self._write_skill(
            tmp_path, "good", "---\nname: G\ndescription: gd\n---\n"
        )
        # Broken — missing name
        self._write_skill(tmp_path, "broken", "---\ndescription: bad\n---\n")
        report = load_skills_dir(tmp_path)
        assert [s.id for s in report.loaded] == ["good"]
        assert len(report.errors) == 1
        err_path, err_obj = report.errors[0]
        assert err_path.parent.name == "broken"
        assert isinstance(err_obj, SkillLoadError)
        assert any("broken" in r.message for r in caplog.records)

    def test_strict_raises_on_first_error(self, tmp_path):
        self._write_skill(tmp_path, "bad", "---\ndescription: d\n---\n")
        with pytest.raises(SkillLoadError):
            load_skills_dir(tmp_path, strict=True)


# ─────────────────────────────────────────────────────────────────
# SkillRegistry
# ─────────────────────────────────────────────────────────────────


def _make_skill(sid: str, name: str = "S", description: str = "d") -> Skill:
    return Skill(id=sid, metadata=SkillMetadata(name=name, description=description), body="")


class TestSkillRegistry:
    def test_register_and_get(self):
        reg = SkillRegistry()
        s = _make_skill("a")
        reg.register(s)
        assert reg.get("a") is s
        assert reg.get("missing") is None

    def test_duplicate_rejected(self):
        reg = SkillRegistry()
        reg.register(_make_skill("a"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(_make_skill("a"))

    def test_unregister_allows_reregister(self):
        reg = SkillRegistry()
        reg.register(_make_skill("a", name="first"))
        reg.unregister("a")
        reg.register(_make_skill("a", name="second"))
        assert reg.get("a").name == "second"

    def test_unregister_missing_is_noop(self):
        reg = SkillRegistry()
        reg.unregister("nothing")

    def test_register_many(self):
        reg = SkillRegistry()
        reg.register_many([_make_skill("a"), _make_skill("b")])
        assert len(reg) == 2

    def test_list_all_sorted(self):
        reg = SkillRegistry()
        reg.register(_make_skill("c"))
        reg.register(_make_skill("a"))
        reg.register(_make_skill("b"))
        assert [s.id for s in reg.list_all()] == ["a", "b", "c"]

    def test_contains_and_len(self):
        reg = SkillRegistry()
        reg.register(_make_skill("a"))
        assert "a" in reg
        assert "b" not in reg
        assert len(reg) == 1
