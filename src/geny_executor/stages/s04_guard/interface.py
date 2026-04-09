"""Stage 4: Guard — interface definitions."""

from __future__ import annotations

from abc import abstractmethod
from typing import List, Optional

from geny_executor.core.stage import Strategy
from geny_executor.core.state import PipelineState
from geny_executor.stages.s04_guard.types import GuardResult


class Guard(Strategy):
    """Base interface for individual guard checks."""

    @abstractmethod
    def check(self, state: PipelineState) -> GuardResult:
        """Run guard check. Returns GuardResult."""


class GuardChain:
    """Chain of guards — all must pass for execution to proceed."""

    def __init__(self, guards: Optional[List[Guard]] = None):
        self._guards = list(guards or [])

    def add(self, guard: Guard) -> GuardChain:
        self._guards.append(guard)
        return self

    def check_all(self, state: PipelineState) -> GuardResult:
        """Run all guards. First failure stops the chain."""
        for guard in self._guards:
            result = guard.check(state)
            if not result.passed:
                return result
        return GuardResult(passed=True, guard_name="chain", message="All guards passed")

    @property
    def guards(self) -> List[Guard]:
        return list(self._guards)
