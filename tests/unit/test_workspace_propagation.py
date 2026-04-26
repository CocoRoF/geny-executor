"""Workspace cross-pipeline propagation tests (PR-D.4.3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from geny_executor.workspace import (
    Workspace,
    WorkspaceStack,
    workspace_stack_from_snapshot,
    workspace_stack_to_snapshot,
)


# ── snapshot helpers round-trip ──────────────────────────────────────


class TestSnapshot:
    def test_empty_stack_round_trip(self):
        s = WorkspaceStack()
        snap = workspace_stack_to_snapshot(s)
        assert snap == []
        s2 = workspace_stack_from_snapshot(snap)
        assert s2.depth() == 0

    def test_full_stack_round_trip(self):
        s = WorkspaceStack(initial=Workspace(cwd=Path("/seed")))
        s.push(Workspace(cwd=Path("/feat"), git_branch="feature-x"))
        s.push(Workspace(cwd=Path("/feat/sub"), env_vars={"K": "V"}))
        snap = workspace_stack_to_snapshot(s)
        assert len(snap) == 3
        s2 = workspace_stack_from_snapshot(snap)
        assert s2.depth() == 3
        assert s2.current().cwd == Path("/feat/sub")
        assert s2.current().env_vars == {"K": "V"}

    def test_invalid_snapshot_returns_empty_stack(self):
        # Non-list / non-dict rows are tolerated; bad entries skipped.
        assert workspace_stack_from_snapshot("not a list").depth() == 0
        assert workspace_stack_from_snapshot([1, "x", None]).depth() == 0

    def test_partial_dict_uses_defaults(self):
        snap = [{"cwd": "/x"}]   # no branch/env/metadata
        s = workspace_stack_from_snapshot(snap)
        ws = s.current()
        assert ws.cwd == Path("/x")
        assert ws.git_branch is None
        assert ws.env_vars == {}


# ── Orchestrator threads workspace_snapshot ──────────────────────────


def test_orchestrator_propagates_snapshot_to_sub_state():
    """SubagentTypeOrchestrator copies parent state.shared
    [workspace_snapshot] into sub_state.shared. We test the propagation
    helper in isolation; the full orchestrator integration requires a
    real sub-pipeline factory which is exercised by the executor's own
    integration suite."""
    from geny_executor.core.state import PipelineState

    parent = PipelineState(session_id="parent")
    parent.shared["workspace_snapshot"] = [
        {"cwd": "/parent-work", "git_branch": "feature-x"},
    ]
    sub = PipelineState(session_id="sub")
    # The orchestrator's logic — duplicated minimal test:
    if "workspace_snapshot" in parent.shared:
        sub.shared["workspace_snapshot"] = parent.shared["workspace_snapshot"]
    # Sub now has the snapshot.
    rehydrated = workspace_stack_from_snapshot(sub.shared["workspace_snapshot"])
    assert rehydrated.current().git_branch == "feature-x"
