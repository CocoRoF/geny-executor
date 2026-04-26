"""Workspace + WorkspaceStack tests (PR-D.4.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from geny_executor.workspace import Workspace, WorkspaceStack


# ── Workspace dataclass ──────────────────────────────────────────────


class TestWorkspace:
    def test_default_cwd_is_current_dir(self):
        ws = Workspace()
        assert ws.cwd.exists()

    def test_with_cwd_returns_new_instance(self):
        ws = Workspace(cwd=Path("/tmp"))
        ws2 = ws.with_cwd(Path("/var"))
        assert ws.cwd == Path("/tmp")        # unchanged
        assert ws2.cwd == Path("/var")
        assert ws is not ws2

    def test_with_branch(self):
        ws = Workspace().with_branch("feature-x")
        assert ws.git_branch == "feature-x"

    def test_with_branch_none_clears(self):
        ws = Workspace().with_branch("x").with_branch(None)
        assert ws.git_branch is None

    def test_with_lsp(self):
        ws = Workspace().with_lsp("session-42")
        assert ws.lsp_session_id == "session-42"

    def test_with_env_merges(self):
        ws = Workspace(env_vars={"A": "1"}).with_env({"B": "2"})
        assert ws.env_vars == {"A": "1", "B": "2"}

    def test_with_env_later_wins(self):
        ws = Workspace(env_vars={"A": "1"}).with_env({"A": "2"})
        assert ws.env_vars["A"] == "2"

    def test_with_metadata_merges(self):
        ws = Workspace(metadata={"x": 1}).with_metadata(y=2)
        assert ws.metadata == {"x": 1, "y": 2}

    def test_to_dict(self):
        ws = Workspace(
            cwd=Path("/work"),
            git_branch="main",
            env_vars={"K": "V"},
        )
        d = ws.to_dict()
        assert d["cwd"] == "/work"
        assert d["git_branch"] == "main"
        assert d["env_vars"] == {"K": "V"}

    def test_immutability(self):
        ws = Workspace(cwd=Path("/x"))
        with pytest.raises(Exception):
            # frozen dataclass — direct attribute set raises
            ws.cwd = Path("/y")  # type: ignore[misc]


# ── WorkspaceStack ───────────────────────────────────────────────────


class TestWorkspaceStack:
    def test_empty_current_is_none(self):
        s = WorkspaceStack()
        assert s.current() is None
        assert s.depth() == 0

    def test_initial_seeds(self):
        ws = Workspace(cwd=Path("/seed"))
        s = WorkspaceStack(initial=ws)
        assert s.current() is ws
        assert s.depth() == 1

    def test_push_pop_round_trip(self):
        s = WorkspaceStack()
        ws = Workspace(cwd=Path("/a"))
        s.push(ws)
        assert s.current() is ws
        popped = s.pop()
        assert popped is ws
        assert s.current() is None

    def test_pop_empty_raises(self):
        with pytest.raises(IndexError):
            WorkspaceStack().pop()

    def test_nested_push(self):
        s = WorkspaceStack()
        ws1 = Workspace(cwd=Path("/a"))
        ws2 = Workspace(cwd=Path("/b"))
        s.push(ws1)
        s.push(ws2)
        assert s.current() is ws2
        s.pop()
        assert s.current() is ws1

    def test_snapshot_returns_copy(self):
        s = WorkspaceStack()
        ws = Workspace(cwd=Path("/a"))
        s.push(ws)
        snap = s.snapshot()
        assert snap == [ws]
        # Mutating the snapshot does not affect the stack.
        snap.clear()
        assert s.depth() == 1
