"""WorkspaceStack — push/pop workspace for nested tool contexts."""

from __future__ import annotations

from typing import List, Optional

from geny_executor.workspace.types import Workspace


class WorkspaceStack:
    """LIFO of :class:`Workspace` values.

    Push a new workspace when a tool enters a scope (worktree branch,
    LSP session, etc.); pop when leaving. ``current()`` returns the
    top of the stack or ``None`` when empty.

    Not thread-safe — sessions are single-coroutine; if you need a
    workspace stack across concurrent sub-agents, give each its own
    instance (typically seeded from the parent's current workspace).
    """

    def __init__(self, initial: Optional[Workspace] = None) -> None:
        self._stack: List[Workspace] = []
        if initial is not None:
            self._stack.append(initial)

    def push(self, ws: Workspace) -> None:
        self._stack.append(ws)

    def pop(self) -> Workspace:
        if not self._stack:
            raise IndexError("WorkspaceStack: pop from empty stack")
        return self._stack.pop()

    def current(self) -> Optional[Workspace]:
        return self._stack[-1] if self._stack else None

    def depth(self) -> int:
        return len(self._stack)

    def snapshot(self) -> List[Workspace]:
        """Frozen copy of the stack — newest last. Useful for debug
        dumps and AgentTool spawn (sub-agent inherits the parent's
        chain so it can compose its own pushes on top)."""
        return list(self._stack)


__all__ = ["WorkspaceStack"]
