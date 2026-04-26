"""Git worktree tools — EnterWorktree / ExitWorktree (PR-A.3.4).

Lets sub-agents (or the main agent) work in an isolated branch
without changing the host process's cwd. Worktree state is tracked
on ``ToolContext.extras["worktree_stack"]`` (a list of dicts) so
EnterWorktree push and ExitWorktree pop are paired.

The actual file operations downstream (Read / Write / Edit / Bash)
will see the regular ``context.working_dir`` — host integrations
that want subsequent file ops to resolve under the worktree should
either re-bind ``context.working_dir`` (preferred) or treat the
returned path as the new cwd.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from geny_executor.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult


_DEFAULT_WORKTREE_BASE = ".worktrees"


def _stack(ctx: ToolContext) -> List[Dict[str, Any]]:
    stack = ctx.extras.get("worktree_stack")
    if not isinstance(stack, list):
        stack = []
        ctx.extras["worktree_stack"] = stack
    return stack


def _err(code: str, message: str) -> ToolResult:
    return ToolResult(
        content={"error": {"code": code, "message": message}},
        is_error=True,
    )


async def _git(*args: str, cwd: str) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        "git", *args,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")


class EnterWorktreeTool(Tool):
    @property
    def name(self) -> str:
        return "EnterWorktree"

    @property
    def description(self) -> str:
        return (
            "Create a git worktree for the given branch and push it onto the "
            "session's worktree stack. Subsequent ExitWorktree pops back."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "path": {"type": "string", "description": "Optional explicit worktree dir."},
                "base": {"type": "string", "description": "Optional base ref for new branch."},
            },
            "required": ["branch"],
        }

    def capabilities(self, input):
        return ToolCapabilities(concurrency_safe=False, destructive=False)

    async def execute(self, input, context):
        cwd = context.working_dir or "."
        if not (Path(cwd) / ".git").exists() and not Path(cwd).joinpath(".git").is_file():
            return _err("NOT_A_GIT_REPO", f"{cwd} is not a git repository")
        branch = input["branch"]
        path = input.get("path")
        if path is None:
            base_dir = Path(cwd) / _DEFAULT_WORKTREE_BASE
            base_dir.mkdir(parents=True, exist_ok=True)
            path = str(base_dir / branch.replace("/", "_"))
        cmd = ["worktree", "add"]
        if input.get("base"):
            cmd += ["-b", branch, path, input["base"]]
        else:
            cmd += [path, branch]
        rc, stdout, stderr = await _git(*cmd, cwd=cwd)
        if rc != 0:
            return _err("GIT_WORKTREE_FAILED", stderr.strip()[:500] or stdout.strip()[:500])
        _stack(context).append({"path": path, "branch": branch})
        return ToolResult(content={"worktree_path": path, "branch": branch, "depth": len(_stack(context))})


class ExitWorktreeTool(Tool):
    @property
    def name(self) -> str:
        return "ExitWorktree"

    @property
    def description(self) -> str:
        return "Pop the most recent worktree from the session stack. Optionally remove it from disk."

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {"remove": {"type": "boolean", "default": False}},
        }

    def capabilities(self, input):
        return ToolCapabilities(concurrency_safe=False, destructive=True)

    async def execute(self, input, context):
        stack = _stack(context)
        if not stack:
            return _err("NO_WORKTREE", "session stack is empty")
        entry = stack.pop()
        if input.get("remove", False):
            cwd = context.working_dir or "."
            rc, _stdout, stderr = await _git("worktree", "remove", entry["path"], cwd=cwd)
            if rc != 0:
                # Non-fatal — pop already happened. Surface the error
                # so the LLM can decide whether to clean up manually.
                return ToolResult(
                    content={
                        "exited": entry["path"],
                        "branch": entry["branch"],
                        "removed": False,
                        "warning": stderr.strip()[:500],
                    },
                )
            return ToolResult(content={
                "exited": entry["path"],
                "branch": entry["branch"],
                "removed": True,
            })
        return ToolResult(content={
            "exited": entry["path"],
            "branch": entry["branch"],
            "removed": False,
        })


__all__ = ["EnterWorktreeTool", "ExitWorktreeTool"]
