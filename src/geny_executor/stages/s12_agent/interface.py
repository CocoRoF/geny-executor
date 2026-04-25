"""Stage 11: Agent — interface definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from geny_executor.core.stage import Strategy
from geny_executor.core.state import PipelineState
from geny_executor.stages.s12_agent.types import AgentResult

if TYPE_CHECKING:
    from geny_executor.core.pipeline import Pipeline


class SubPipelineFactory(ABC):
    """Creates sub-pipelines for agent delegation."""

    @abstractmethod
    def create(self, agent_type: str) -> Pipeline:
        """Create a pipeline for the given agent type."""
        ...


class AgentOrchestrator(Strategy, ABC):
    """Level 2 strategy: multi-agent orchestration pattern."""

    @abstractmethod
    async def orchestrate(self, state: PipelineState) -> AgentResult:
        """Orchestrate agent delegation. Return result."""
        ...
