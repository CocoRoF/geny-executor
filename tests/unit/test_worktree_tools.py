"""Worktree tool tests (PR-A.3.4)."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from geny_executor.tools.base import ToolContext
from geny_executor.tools.built_in import (
    BUILT_IN_TOOL_CLASSES,
    EnterWorktreeTool,
    ExitWorktreeTool,
)


def _git_init(repo: Path) -> None:
    """Initialise a small git repo for tests."""
    subprocess.run(["git", "init", "-q", "--initial-branch=main", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test"],
        check=True,
    )
    (repo / "README.md").write_text("hello\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "initial"],
        check=True,
    )


def test_both_registered():
    assert "EnterWorktree" in BUILT_IN_TOOL_CLASSES
    assert "ExitWorktree" in BUILT_IN_TOOL_CLASSES


# ── EnterWorktree ────────────────────────────────────────────────────


class TestEnter:
    @pytest.mark.asyncio
    async def test_creates_worktree_with_new_branch(self, tmp_path: Path):
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        ctx = ToolContext(working_dir=str(repo), extras={})
        result = await EnterWorktreeTool().execute(
            {"branch": "feature-x", "base": "main"}, ctx,
        )
        assert result.is_error is False
        assert "feature-x" in result.content["branch"]
        assert Path(result.content["worktree_path"]).exists()
        assert result.content["depth"] == 1

    @pytest.mark.asyncio
    async def test_pushes_onto_stack(self, tmp_path: Path):
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        ctx = ToolContext(working_dir=str(repo), extras={})
        await EnterWorktreeTool().execute({"branch": "a", "base": "main"}, ctx)
        await EnterWorktreeTool().execute({"branch": "b", "base": "main"}, ctx)
        stack = ctx.extras["worktree_stack"]
        assert len(stack) == 2
        assert stack[1]["branch"] == "b"

    @pytest.mark.asyncio
    async def test_not_a_git_repo(self, tmp_path: Path):
        ctx = ToolContext(working_dir=str(tmp_path), extras={})
        result = await EnterWorktreeTool().execute({"branch": "x"}, ctx)
        assert result.is_error is True
        assert result.content["error"]["code"] == "NOT_A_GIT_REPO"

    @pytest.mark.asyncio
    async def test_git_command_failure(self, tmp_path: Path):
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        ctx = ToolContext(working_dir=str(repo), extras={})
        # Worktree to a path that already has a non-empty dir → git fails.
        existing = repo / "blocker"
        existing.mkdir()
        (existing / "file").write_text("x")
        result = await EnterWorktreeTool().execute(
            {"branch": "fail", "path": str(existing), "base": "main"}, ctx,
        )
        assert result.is_error is True
        assert result.content["error"]["code"] == "GIT_WORKTREE_FAILED"


# ── ExitWorktree ─────────────────────────────────────────────────────


class TestExit:
    @pytest.mark.asyncio
    async def test_pops_without_remove(self, tmp_path: Path):
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        ctx = ToolContext(working_dir=str(repo), extras={})
        await EnterWorktreeTool().execute({"branch": "x", "base": "main"}, ctx)
        result = await ExitWorktreeTool().execute({}, ctx)
        assert result.is_error is False
        assert result.content["removed"] is False
        assert ctx.extras["worktree_stack"] == []
        # Worktree dir still exists.
        assert Path(result.content["exited"]).exists()

    @pytest.mark.asyncio
    async def test_pops_with_remove(self, tmp_path: Path):
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        ctx = ToolContext(working_dir=str(repo), extras={})
        await EnterWorktreeTool().execute({"branch": "x", "base": "main"}, ctx)
        result = await ExitWorktreeTool().execute({"remove": True}, ctx)
        assert result.is_error is False
        assert result.content["removed"] is True

    @pytest.mark.asyncio
    async def test_empty_stack(self, tmp_path: Path):
        ctx = ToolContext(working_dir=str(tmp_path), extras={})
        result = await ExitWorktreeTool().execute({}, ctx)
        assert result.is_error is True
        assert result.content["error"]["code"] == "NO_WORKTREE"
