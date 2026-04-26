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

__all__ = ["Workspace", "WorkspaceStack"]
