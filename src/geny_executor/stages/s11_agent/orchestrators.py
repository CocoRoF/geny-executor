"""Agent stage — orchestrator strategies (Level 2)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from geny_executor.core.stage import Strategy
from geny_executor.core.state import PipelineState

if TYPE_CHECKING:
    from geny_executor.core.pipeline import Pipeline


@dataclass
class AgentResult:
    """Result of agent orchestration."""

    delegated: bool = False
    sub_results: List[Dict[str, Any]] = field(default_factory=list)
    evaluation_input: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SubPipelineFactory(ABC):
    """Creates sub-pipelines for agent delegation."""

    @abstractmethod
    def create(self, agent_type: str) -> Pipeline:
        """Create a pipeline for the given agent type."""
        ...


class DefaultSubPipelineFactory(SubPipelineFactory):
    """Factory that uses registered pipeline creators."""

    def __init__(self):
        self._creators: Dict[str, Callable[[], Pipeline]] = {}

    def register(self, agent_type: str, creator: Callable[[], Pipeline]) -> None:
        self._creators[agent_type] = creator

    def create(self, agent_type: str) -> Pipeline:
        if agent_type not in self._creators:
            raise ValueError(f"Unknown agent type: {agent_type}")
        return self._creators[agent_type]()


class AgentOrchestrator(Strategy, ABC):
    """Level 2 strategy: multi-agent orchestration pattern."""

    @abstractmethod
    async def orchestrate(
        self,
        state: PipelineState,
    ) -> AgentResult:
        """Orchestrate agent delegation. Return result."""
        ...


class SingleAgentOrchestrator(AgentOrchestrator):
    """No delegation — single agent passes through.

    Default for most use cases. The main pipeline handles everything.
    """

    @property
    def name(self) -> str:
        return "single_agent"

    async def orchestrate(self, state: PipelineState) -> AgentResult:
        return AgentResult(delegated=False)


class DelegateOrchestrator(AgentOrchestrator):
    """Delegates to sub-pipelines when [DELEGATE: agent_type] signal detected.

    Creates a new pipeline session for the delegated task,
    runs it, and integrates the result back.
    """

    def __init__(self, factory: Optional[SubPipelineFactory] = None):
        self._factory = factory or DefaultSubPipelineFactory()

    @property
    def name(self) -> str:
        return "delegate"

    async def orchestrate(self, state: PipelineState) -> AgentResult:
        if not state.delegate_requests:
            return AgentResult(delegated=False)

        sub_results = []
        for request in state.delegate_requests:
            agent_type = request.get("agent_type", "default")
            task = request.get("task", "")

            try:
                sub_pipeline = self._factory.create(agent_type)
                sub_state = PipelineState(
                    session_id=f"{state.session_id}-sub-{agent_type}",
                )
                result = await sub_pipeline.run(task, sub_state)
                sub_results.append({
                    "agent_type": agent_type,
                    "task": task,
                    "success": result.success,
                    "text": result.text,
                    "error": result.error,
                })
            except Exception as e:
                sub_results.append({
                    "agent_type": agent_type,
                    "task": task,
                    "success": False,
                    "text": "",
                    "error": str(e),
                })

        # Clear processed requests
        state.delegate_requests = []

        return AgentResult(
            delegated=True,
            sub_results=sub_results,
        )


class EvaluatorOrchestrator(AgentOrchestrator):
    """Generator/Evaluator pattern from Anthropic.

    Main pipeline = Generator. A separate lightweight pipeline = Evaluator.
    Evaluator assesses the generator's output and provides feedback
    that flows to Stage 12 (Evaluate).
    """

    def __init__(
        self,
        evaluator_factory: Optional[SubPipelineFactory] = None,
        evaluator_type: str = "evaluator",
    ):
        self._factory = evaluator_factory or DefaultSubPipelineFactory()
        self._evaluator_type = evaluator_type

    @property
    def name(self) -> str:
        return "evaluator"

    async def orchestrate(self, state: PipelineState) -> AgentResult:
        if not state.final_text:
            return AgentResult(delegated=False)

        try:
            evaluator = self._factory.create(self._evaluator_type)
            eval_state = PipelineState(
                session_id=f"{state.session_id}-eval",
            )

            eval_prompt = (
                f"Evaluate the following response for quality, accuracy, and completeness.\n"
                f"Provide a score (0-100) and brief feedback.\n\n"
                f"Response to evaluate:\n{state.final_text}"
            )

            result = await evaluator.run(eval_prompt, eval_state)

            evaluation_input = {
                "evaluator_response": result.text,
                "evaluator_success": result.success,
            }
        except Exception as e:
            evaluation_input = {
                "evaluator_response": "",
                "evaluator_success": False,
                "evaluator_error": str(e),
            }

        return AgentResult(
            delegated=True,
            evaluation_input=evaluation_input,
        )
