"""Workspace + Worktree/LSP tools integration (PR-D.4.2).

Confirms the bridge between the legacy worktree_stack dict and the
new WorkspaceStack: pushing a worktree updates the workspace; popping
restores it; LSP cwd resolution prefers the workspace.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from geny_executor.tools.base import ToolContext
from geny_executor.tools.built_in import (
    EnterWorktreeTool,
    ExitWorktreeTool,
    LSPTool,
)
from geny_executor.workspace import WorkspaceStack


def _git_init(repo: Path) -> None:
    subprocess.run(["git", "init", "-q", "--initial-branch=main", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    (repo / "README.md").write_text("hi")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "init"], check=True)


# ── EnterWorktree pushes onto WorkspaceStack ────────────────────────


class TestEnterPushesWorkspace:
    @pytest.mark.asyncio
    async def test_enter_creates_workspace_stack(self, tmp_path: Path):
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        ctx = ToolContext(working_dir=str(repo), extras={})
        await EnterWorktreeTool().execute({"branch": "feat-a", "base": "main"}, ctx)
        ws_stack = ctx.extras.get("workspace_stack")
        assert isinstance(ws_stack, WorkspaceStack)
        # Initial seed (cwd) + push (worktree) → depth 2.
        assert ws_stack.depth() == 2
        current = ws_stack.current()
        assert current.git_branch == "feat-a"

    @pytest.mark.asyncio
    async def test_exit_pops_workspace(self, tmp_path: Path):
        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)
        ctx = ToolContext(working_dir=str(repo), extras={})
        await EnterWorktreeTool().execute({"branch": "x", "base": "main"}, ctx)
        await ExitWorktreeTool().execute({}, ctx)
        ws_stack = ctx.extras["workspace_stack"]
        # Back to seed.
        assert ws_stack.depth() == 1
        assert ws_stack.current().git_branch is None


# ── LSPTool prefers workspace cwd ───────────────────────────────────


class TestLSPUsesWorkspace:
    @pytest.mark.asyncio
    async def test_lsp_cwd_from_workspace_when_present(self, tmp_path: Path):
        captured = {}

        async def adapter(*, action, file, line, col, cwd):
            captured["cwd"] = cwd
            return {"diagnostics": []}

        repo = tmp_path / "repo"
        repo.mkdir()
        _git_init(repo)

        ctx = ToolContext(
            working_dir=str(repo),
            extras={"lsp_adapters": {"python": adapter}},
        )
        await EnterWorktreeTool().execute({"branch": "feat-b", "base": "main"}, ctx)
        await LSPTool().execute(
            {"language": "python", "action": "diagnostics", "file": "x.py"},
            ctx,
        )
        # cwd should be the worktree path (under .worktrees/feat-b),
        # not the original repo.
        assert "feat-b" in captured["cwd"]

    @pytest.mark.asyncio
    async def test_lsp_falls_back_to_working_dir_without_workspace(self, tmp_path: Path):
        captured = {}

        async def adapter(*, action, file, line, col, cwd):
            captured["cwd"] = cwd
            return {}

        ctx = ToolContext(
            working_dir=str(tmp_path),
            extras={"lsp_adapters": {"python": adapter}},
        )
        await LSPTool().execute(
            {"language": "python", "action": "hover", "file": "x.py"},
            ctx,
        )
        # No workspace_stack in extras → working_dir.
        assert captured["cwd"] == str(tmp_path)
