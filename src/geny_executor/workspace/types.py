"""Workspace dataclass — immutable value object."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class Workspace:
    """A snapshot of the host filesystem context a tool runs against.

    Immutable on purpose — composition uses :meth:`with_cwd` /
    :meth:`with_branch` etc. to derive new Workspaces. The sandbox
    enforces the actual access boundaries; Workspace is the *intent*.
    """

    cwd: Path = field(default_factory=lambda: Path.cwd())
    git_branch: Optional[str] = None
    lsp_session_id: Optional[str] = None
    env_vars: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def with_cwd(self, new_cwd: Path) -> "Workspace":
        return replace(self, cwd=Path(new_cwd))

    def with_branch(self, branch: Optional[str]) -> "Workspace":
        return replace(self, git_branch=branch)

    def with_lsp(self, session_id: Optional[str]) -> "Workspace":
        return replace(self, lsp_session_id=session_id)

    def with_env(self, env: Mapping[str, str]) -> "Workspace":
        merged = {**dict(self.env_vars), **dict(env)}
        return replace(self, env_vars=merged)

    def with_metadata(self, **extras: Any) -> "Workspace":
        merged = {**dict(self.metadata), **extras}
        return replace(self, metadata=merged)

    def to_dict(self) -> dict:
        return {
            "cwd": str(self.cwd),
            "git_branch": self.git_branch,
            "lsp_session_id": self.lsp_session_id,
            "env_vars": dict(self.env_vars),
            "metadata": dict(self.metadata),
        }


__all__ = ["Workspace"]
