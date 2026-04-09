"""Default implementation of Stage 11: Agent."""

from __future__ import annotations

from typing import Any, List, Optional

from geny_executor.core.stage import Stage, StrategyInfo
from geny_executor.core.state import PipelineState
from geny_executor.stages.s11_agent.interface import AgentOrchestrator
from geny_executor.stages.s11_agent.artifact.default.orchestrators import (
    SingleAgentOrchestrator,
)


class AgentStage(Stage[Any, Any]):
    """Stage 11: Agent.

    Multi-agent orchestration — delegates tasks to sub-pipelines
    when appropriate, based on the configured orchestrator strategy.
    """

    def __init__(
        self,
        orchestrator: Optional[AgentOrchestrator] = None,
    ):
        self._orchestrator = orchestrator or SingleAgentOrchestrator()

    @property
    def name(self) -> str:
        return "agent"

    @property
    def order(self) -> int:
        return 11

    @property
    def category(self) -> str:
        return "execution"

    def should_bypass(self, state: PipelineState) -> bool:
        if isinstance(self._orchestrator, SingleAgentOrchestrator):
            return not state.delegate_requests
        return False

    async def execute(self, input: Any, state: PipelineState) -> Any:
        state.add_event(
            "agent.orchestrate_start",
            {
                "orchestrator": self._orchestrator.name,
                "delegate_count": len(state.delegate_requests),
            },
        )

        result = await self._orchestrator.orchestrate(state)

        if result.delegated:
            for sub in result.sub_results:
                state.agent_results.append(sub)

            if result.evaluation_input:
                state.metadata["evaluation_input"] = result.evaluation_input

            if result.sub_results:
                summary_parts = []
                for sub in result.sub_results:
                    status = "success" if sub.get("success") else "failed"
                    summary_parts.append(
                        f"[Agent:{sub['agent_type']}] ({status}) {sub.get('text', '')[:200]}"
                    )
                if summary_parts:
                    state.add_message(
                        "user",
                        [
                            {
                                "type": "text",
                                "text": "Sub-agent results:\n" + "\n".join(summary_parts),
                            }
                        ],
                    )
                    state.loop_decision = "continue"

        state.add_event(
            "agent.orchestrate_complete",
            {
                "delegated": result.delegated,
                "sub_result_count": len(result.sub_results),
            },
        )

        return input

    def list_strategies(self) -> List[StrategyInfo]:
        return [
            StrategyInfo(
                slot_name="orchestrator",
                current_impl=type(self._orchestrator).__name__,
                available_impls=[
                    "SingleAgentOrchestrator",
                    "DelegateOrchestrator",
                    "EvaluatorOrchestrator",
                ],
            ),
        ]
