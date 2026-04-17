"""Per-stage tool binding — filtered view of the global tool registry.

A :class:`StageToolBinding` describes which tools a particular Stage may use.
``None`` on either ``allowed`` or ``blocked`` means "inherit everything" for
that axis, giving a simple hierarchy:

- ``allowed = None``: any tool not explicitly blocked is permitted.
- ``allowed = {...}``: only the listed tools are permitted.
- ``blocked`` always wins over ``allowed``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from geny_executor.tools.base import Tool
from geny_executor.tools.registry import ToolRegistry


@dataclass
class StageToolBinding:
    """Per-stage view of the global tool registry."""

    stage_order: int
    allowed: Optional[Set[str]] = None
    blocked: Optional[Set[str]] = None
    extra_context: Dict[str, Any] = field(default_factory=dict)

    def is_allowed(self, tool_name: str) -> bool:
        """Return whether *tool_name* is visible to the bound stage."""
        if self.blocked and tool_name in self.blocked:
            return False
        if self.allowed is not None and tool_name not in self.allowed:
            return False
        return True

    def filter(self, registry: ToolRegistry) -> List[Tool]:
        """Return the subset of *registry* visible to the bound stage."""
        include = self.allowed
        exclude = self.blocked
        return registry.filter(include=include, exclude=exclude)

    def allow(self, tool_name: str) -> None:
        """Add *tool_name* to the allow-list (promotes ``None`` to ``set``)."""
        if self.allowed is None:
            self.allowed = set()
        self.allowed.add(tool_name)
        if self.blocked:
            self.blocked.discard(tool_name)

    def block(self, tool_name: str) -> None:
        """Add *tool_name* to the block-list."""
        if self.blocked is None:
            self.blocked = set()
        self.blocked.add(tool_name)
        if self.allowed is not None:
            self.allowed.discard(tool_name)

    def clear(self) -> None:
        """Reset to inherit-everything."""
        self.allowed = None
        self.blocked = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready representation of this binding."""
        return {
            "stage_order": self.stage_order,
            "allowed": sorted(self.allowed) if self.allowed is not None else None,
            "blocked": sorted(self.blocked) if self.blocked is not None else None,
            "extra_context": dict(self.extra_context),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StageToolBinding":
        """Rehydrate a :class:`StageToolBinding` from :meth:`to_dict` output."""
        allowed_raw = data.get("allowed")
        blocked_raw = data.get("blocked")
        return cls(
            stage_order=int(data["stage_order"]),
            allowed=set(allowed_raw) if allowed_raw is not None else None,
            blocked=set(blocked_raw) if blocked_raw is not None else None,
            extra_context=dict(data.get("extra_context", {})),
        )


class ToolAccessDenied(Exception):
    """Raised when a stage attempts to use a tool outside its binding scope."""

    def __init__(self, tool_name: str, stage_order: int) -> None:
        super().__init__(f"Tool '{tool_name}' is not bound to stage {stage_order}")
        self.tool_name = tool_name
        self.stage_order = stage_order
