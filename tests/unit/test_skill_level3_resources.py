"""Level 3 progressive disclosure — bundled resources load on demand.

A skill folder may ship extra files (REFERENCE.md, scripts/…). They are NOT in
context at rest (L1) nor when the body is returned (L2) — only when the caller
asks for one via the SkillTool ``resource`` arg (L3). Uniform across backends.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from geny_executor.skills.loader import parse_skill_file
from geny_executor.skills.skill_tool import SkillTool
from geny_executor.tools.base import ToolContext


def _ctx() -> ToolContext:
    return ToolContext(working_dir="")


def _make_skill(tmp_path: Path) -> Path:
    d = tmp_path / "pdf-processing"
    (d / "scripts").mkdir(parents=True)
    (d / "SKILL.md").write_text(
        "---\nname: pdf-processing\ndescription: Work with PDFs.\n---\n\n"
        "# PDF\n\nFor advanced forms, load REFERENCE.md.\n",
        encoding="utf-8",
    )
    (d / "REFERENCE.md").write_text("# Detailed API\n\nlots of detail here\n", encoding="utf-8")
    (d / "scripts" / "fill.py").write_text("print('fill')\n", encoding="utf-8")
    (d / ".gitkeep").write_text("", encoding="utf-8")
    return d / "SKILL.md"


def test_list_resources_excludes_skill_md_and_dotfiles(tmp_path: Path) -> None:
    skill = parse_skill_file(_make_skill(tmp_path))
    res = skill.list_resources()
    assert "REFERENCE.md" in res
    assert "scripts/fill.py" in res
    assert "SKILL.md" not in res
    assert ".gitkeep" not in res


def test_schema_advertises_resource_only_when_present(tmp_path: Path) -> None:
    skill = parse_skill_file(_make_skill(tmp_path))
    props = SkillTool(skill).input_schema["properties"]
    assert "resource" in props
    # A skill with no bundled files keeps the minimal schema.
    bare = tmp_path / "bare" / "SKILL.md"
    bare.parent.mkdir()
    bare.write_text("---\nname: bare\ndescription: x\n---\n\nbody\n", encoding="utf-8")
    assert "resource" not in SkillTool(parse_skill_file(bare)).input_schema["properties"]


@pytest.mark.asyncio
async def test_l2_body_lists_resources_l3_loads_on_demand(tmp_path: Path) -> None:
    tool = SkillTool(parse_skill_file(_make_skill(tmp_path)))

    # L2 — running the skill returns the body + lists resources, NOT contents.
    l2 = await tool.execute({"args": {}}, _ctx())
    assert not l2.is_error
    assert "For advanced forms" in l2.content          # body
    assert "REFERENCE.md" in l2.content                # listed
    assert "lots of detail here" not in l2.content     # content NOT pulled yet

    # L3 — explicitly load the resource → its content, not the body.
    l3 = await tool.execute({"resource": "REFERENCE.md"}, _ctx())
    assert not l3.is_error
    assert "lots of detail here" in l3.content
    assert "For advanced forms" not in l3.content
    assert l3.metadata.get("resource") == "REFERENCE.md"


@pytest.mark.asyncio
async def test_traversal_guard_and_missing(tmp_path: Path) -> None:
    tool = SkillTool(parse_skill_file(_make_skill(tmp_path)))
    esc = await tool.execute({"resource": "../../etc/passwd"}, _ctx())
    assert esc.is_error
    missing = await tool.execute({"resource": "NOPE.md"}, _ctx())
    assert missing.is_error and "not found" in missing.content
