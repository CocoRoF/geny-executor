"""Regression test for the 2.0.1 fix: SubagentTypeOrchestrator must
accept zero-arg construction so the StrategySlot restore path that
runs *before* Pipeline._wire_subagent_orchestrator doesn't crash.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor.core.state import PipelineState
from geny_executor.stages.s12_agent.subagent_type import (
    SubagentTypeDescriptor,
    SubagentTypeOrchestrator,
    SubagentTypeRegistry,
)


def test_orchestrator_accepts_zero_arg() -> None:
    """The StrategySlot machinery instantiates the strategy with no
    constructor args during PipelineMutator.restore. 2.0.0 raised
    TypeError here; 2.0.1 returns a usable instance with an empty
    registry."""
    orch = SubagentTypeOrchestrator()
    assert orch._registry is not None
    assert len(orch._registry) == 0


def test_orchestrator_explicit_registry_unchanged() -> None:
    reg = SubagentTypeRegistry()
    reg.register(SubagentTypeDescriptor(
        agent_type="x", factory=lambda ctx: None, description="x"
    ))
    orch = SubagentTypeOrchestrator(reg)
    assert orch._registry is reg
    assert len(orch._registry) == 1


@pytest.mark.asyncio
async def test_zero_arg_orchestrate_treats_unknown_agent_as_structured_failure() -> None:
    """Before _wire_subagent_orchestrator runs, the orchestrator has an
    empty registry. Any delegate request must land as a structured
    failure, not a crash."""
    orch = SubagentTypeOrchestrator()
    state = PipelineState(session_id="s")
    state.delegate_requests = [{"agent_type": "missing", "task": "t"}]
    result = await orch.orchestrate(state)
    assert result.delegated is True
    assert len(result.sub_results) == 1
    assert result.sub_results[0]["success"] is False
    assert "unknown_agent_type" in (result.sub_results[0].get("error") or "")
