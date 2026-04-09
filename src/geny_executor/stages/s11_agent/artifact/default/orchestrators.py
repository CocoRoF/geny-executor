"""Default artifact orchestrators for Stage 11: Agent."""

from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from geny_executor.core.state import PipelineState
from geny_executor.stages.s11_agent.interface import AgentOrchestrator, SubPipelineFactory
from geny_executor.stages.s11_agent.types import AgentResult

if TYPE_CHECKING:
    from geny_executor.core.pipeline import Pipeline


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


class SingleAgentOrchestrator(AgentOrchestrator):
    """No delegation — single agent passes through."""

    @property
    def name(self) -> str:
        return "single_agent"

    async def orchestrate(self, state: PipelineState) -> AgentResult:
        return AgentResult(delegated=False)


class DelegateOrchestrator(AgentOrchestrator):
    """Delegates to sub-pipelines when [DELEGATE: agent_type] signal detected."""

    def __init__(self, factory: Optional[SubPipelineFactory] = None):
        self._factory = factory or DefaultSubPipelineFactory()

    @property
    def name(self) -> str:
        return "delegate"

    async def orchestrate(self, state: PipelineState) -> AgentResult:
        if not state.delegate_requests:
            return AgentResult(delegated=False)

        sub_results: List[Dict[str, Any]] = []
        for request in state.delegate_requests:
            agent_type = request.get("agent_type", "default")
            task = request.get("task", "")

            try:
                sub_pipeline = self._factory.create(agent_type)
                sub_state = PipelineState(
                    session_id=f"{state.session_id}-sub-{agent_type}-{uuid.uuid4().hex[:8]}",
                )
                result = await sub_pipeline.run(task, sub_state)
                sub_results.append(
                    {
                        "agent_type": agent_type,
                        "task": task,
                        "success": result.success,
                        "text": result.text,
                        "error": result.error,
                    }
                )
            except Exception as e:
                sub_results.append(
                    {
                        "agent_type": agent_type,
                        "task": task,
                        "success": False,
                        "text": "",
                        "error": str(e),
                    }
                )

        state.delegate_requests = []

        return AgentResult(delegated=True, sub_results=sub_results)


class EvaluatorOrchestrator(AgentOrchestrator):
    """Generator/Evaluator pattern from Anthropic."""

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
                session_id=f"{state.session_id}-eval-{uuid.uuid4().hex[:8]}",
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

        return AgentResult(delegated=True, evaluation_input=evaluation_input)
