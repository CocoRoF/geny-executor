"""2.2.0 Wave 1 — GuardStage fail_fast / max_chain_length wiring.

Audit "validated-but-inert" table: both knobs passed schema validation
and were never read in ``execute`` — the operator saw a green check and
no behaviour change. These tests pin the wiring:

  * fail_fast=True (default) keeps the historical first-failure
    short-circuit untouched.
  * fail_fast=False runs EVERY guard, aggregates the violations into one
    event + one error, and still honours warn-action results.
  * max_chain_length rejects oversized chains with a message naming the
    knob so the fix is obvious from the error alone.
"""

from __future__ import annotations

import pytest

from geny_executor.core.errors import GuardRejectError
from geny_executor.core.state import PipelineState
from geny_executor.stages.s04_guard.artifact.default.stage import GuardStage
from geny_executor.stages.s04_guard.interface import Guard
from geny_executor.stages.s04_guard.types import GuardResult


class _StubGuard(Guard):
    """Records whether it ran; returns a canned result."""

    def __init__(self, label: str, *, passed: bool, action: str = "reject"):
        self._label = label
        self._passed = passed
        self._action = action
        self.calls = 0

    @property
    def name(self) -> str:
        return self._label

    def check(self, state: PipelineState) -> GuardResult:
        self.calls += 1
        return GuardResult(
            passed=self._passed,
            guard_name=self._label,
            message=f"{self._label} says no" if not self._passed else "",
            action=self._action,
        )


def _state() -> PipelineState:
    return PipelineState(session_id="s")


def _events(state: PipelineState, event_type: str) -> list:
    return [e for e in state.events if e.get("type") == event_type]


class TestFailFastDefault:
    @pytest.mark.asyncio
    async def test_first_failure_short_circuits(self):
        first = _StubGuard("g1", passed=False)
        second = _StubGuard("g2", passed=True)
        stage = GuardStage(guards=[first, second])  # fail_fast defaults True

        with pytest.raises(GuardRejectError) as exc_info:
            await stage.execute("in", _state())

        assert exc_info.value.guard_name == "g1"
        assert second.calls == 0  # historical short-circuit preserved

    @pytest.mark.asyncio
    async def test_all_pass_returns_input(self):
        stage = GuardStage(guards=[_StubGuard("g1", passed=True)])
        assert await stage.execute("in", _state()) == "in"


class TestFailFastDisabled:
    @pytest.mark.asyncio
    async def test_every_guard_runs_and_violations_aggregate(self):
        g1 = _StubGuard("g1", passed=False)
        g2 = _StubGuard("g2", passed=True)
        g3 = _StubGuard("g3", passed=False)
        stage = GuardStage(guards=[g1, g2, g3], fail_fast=False)
        state = _state()

        with pytest.raises(GuardRejectError) as exc_info:
            await stage.execute("in", state)

        # Every guard was consulted — the point of the aggregate mode.
        assert (g1.calls, g2.calls, g3.calls) == (1, 1, 1)
        # The error names all failing guards, not just the first.
        assert "g1" in str(exc_info.value)
        assert "g3" in str(exc_info.value)
        assert exc_info.value.guard_name == "g1, g3"
        # The event carries the structured violation list for dashboards.
        checks = _events(state, "guard.check")
        assert len(checks) == 1
        violations = checks[0]["data"]["violations"]
        assert [v["guard_name"] for v in violations] == ["g1", "g3"]

    @pytest.mark.asyncio
    async def test_warn_only_failures_do_not_raise(self):
        g1 = _StubGuard("g1", passed=False, action="warn")
        g2 = _StubGuard("g2", passed=False, action="warn")
        stage = GuardStage(guards=[g1, g2], fail_fast=False)
        state = _state()

        result = await stage.execute("in", state)

        assert result == "in"
        warns = _events(state, "guard.warn")
        assert len(warns) == 2

    @pytest.mark.asyncio
    async def test_mixed_warn_and_reject_raises_with_rejects_only(self):
        warn = _StubGuard("warner", passed=False, action="warn")
        reject = _StubGuard("rejector", passed=False)
        stage = GuardStage(guards=[warn, reject], fail_fast=False)
        state = _state()

        with pytest.raises(GuardRejectError) as exc_info:
            await stage.execute("in", state)

        assert exc_info.value.guard_name == "rejector"
        assert len(_events(state, "guard.warn")) == 1

    @pytest.mark.asyncio
    async def test_via_update_config(self):
        """The manifest path sets the knob through update_config."""
        g1 = _StubGuard("g1", passed=False)
        g2 = _StubGuard("g2", passed=False)
        stage = GuardStage(guards=[g1, g2])
        stage.update_config({"fail_fast": False})

        with pytest.raises(GuardRejectError):
            await stage.execute("in", _state())
        assert g2.calls == 1


class TestMaxChainLength:
    @pytest.mark.asyncio
    async def test_oversized_chain_rejected_with_clear_error(self):
        guards = [_StubGuard(f"g{i}", passed=True) for i in range(3)]
        stage = GuardStage(guards=guards, max_chain_length=2)

        with pytest.raises(GuardRejectError, match="max_chain_length=2"):
            await stage.execute("in", _state())
        # No guard ran — the configuration itself is invalid.
        assert all(g.calls == 0 for g in guards)

    @pytest.mark.asyncio
    async def test_chain_at_limit_is_fine(self):
        guards = [_StubGuard(f"g{i}", passed=True) for i in range(2)]
        stage = GuardStage(guards=guards, max_chain_length=2)
        assert await stage.execute("in", _state()) == "in"

    @pytest.mark.asyncio
    async def test_via_update_config(self):
        guards = [_StubGuard(f"g{i}", passed=True) for i in range(3)]
        stage = GuardStage(guards=guards)
        stage.update_config({"max_chain_length": 1})

        with pytest.raises(GuardRejectError, match="max_chain_length"):
            await stage.execute("in", _state())
