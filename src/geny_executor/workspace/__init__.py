"""Workspace abstraction (PR-D.4.1).

A :class:`Workspace` bundles three currently-independent concepts:

- ``cwd`` — sandbox-rooted working directory
- ``git_branch`` — worktree-aware branch context
- ``lsp_session_id`` — language-server context handle
- ``env_vars`` — workspace-scoped env overrides
- ``metadata`` — host-supplied free-form bag

WorkspaceStack lets nested tools (EnterWorktree etc.) push a new
Workspace and pop on cleanup. ToolContext gains a ``workspace``
view (default seeded from working_dir) so workspace-aware tools
can read it instead of duplicating cwd plumbing.

Backward compatible — every Workspace field is optional with
sensible defaults; tools that don't read ``ctx.workspace`` keep
working unchanged.
"""

from geny_executor.workspace.stack import WorkspaceStack
from geny_executor.workspace.types import Workspace


# ── Snapshot helpers for cross-pipeline propagation (PR-D.4.3) ───────


def workspace_stack_to_snapshot(stack: WorkspaceStack) -> list:
    """Serialize a WorkspaceStack for cross-pipeline propagation.

    The orchestrator stashes this under ``state.shared["workspace_snapshot"]``
    before spawning a sub-pipeline; the sub's host calls
    :func:`workspace_stack_from_snapshot` to rehydrate.
    """
    return [ws.to_dict() for ws in stack.snapshot()]


def workspace_stack_from_snapshot(snapshot: list) -> WorkspaceStack:
    """Inverse of :func:`workspace_stack_to_snapshot`. Tolerates dict
    rows missing keys — defaults from Workspace() apply."""
    from pathlib import Path

    stack = WorkspaceStack()
    if not isinstance(snapshot, list):
        return stack
    for row in snapshot:
        if not isinstance(row, dict):
            continue
        cwd = row.get("cwd")
        ws = Workspace(
            cwd=Path(cwd) if cwd else Path("."),
            git_branch=row.get("git_branch"),
            lsp_session_id=row.get("lsp_session_id"),
            env_vars=dict(row.get("env_vars") or {}),
            metadata=dict(row.get("metadata") or {}),
        )
        stack.push(ws)
    return stack


__all__ = [
    "Workspace",
    "WorkspaceStack",
    "workspace_stack_from_snapshot",
    "workspace_stack_to_snapshot",
]
