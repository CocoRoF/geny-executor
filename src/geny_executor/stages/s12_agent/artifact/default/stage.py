"""Default implementation of Stage 11: Agent."""

from __future__ import annotations

from typing import Any, Dict, Optional

from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.core.slot import StrategySlot
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState
from geny_executor.stages.s12_agent.interface import AgentOrchestrator
from geny_executor.stages.s12_agent.artifact.default.orchestrators import (
    DelegateOrchestrator,
    EvaluatorOrchestrator,
    SingleAgentOrchestrator,
)
from geny_executor.stages.s12_agent.subagent_type import SubagentTypeOrchestrator


class AgentStage(Stage[Any, Any]):
    """Stage 11: Agent.

    Multi-agent orchestration — delegates tasks to sub-pipelines
    when appropriate, based on the configured orchestrator strategy.
    """

    def __init__(
        self,
        orchestrator: Optional[AgentOrchestrator] = None,
        *,
        max_delegations: int = 4,
    ):
        self._slots: Dict[str, StrategySlot] = {
            "orchestrator": StrategySlot(
                name="orchestrator",
                strategy=orchestrator or SingleAgentOrchestrator(),
                registry={
                    "single_agent": SingleAgentOrchestrator,
                    "delegate": DelegateOrchestrator,
                    "evaluator": EvaluatorOrchestrator,
                    # Phase 7 S7.5 — typed subagent dispatch via
                    # SubagentTypeRegistry. Manifests can name the
                    # strategy; the registry instance arrives via
                    # ``Pipeline.attach_runtime`` (orchestrator
                    # construction needs the registry arg).
                    "subagent_type": SubagentTypeOrchestrator,
                },
                description="Agent orchestration strategy",
            ),
        }
        self._max_delegations = int(max_delegations)

    @property
    def _orchestrator(self) -> AgentOrchestrator:
        return self._slots["orchestrator"].strategy  # type: ignore[return-value]

    @property
    def name(self) -> str:
        return "agent"

    @property
    def order(self) -> int:
        return 11

    @property
    def category(self) -> str:
        return "execution"

    def get_strategy_slots(self) -> Dict[str, StrategySlot]:
        return self._slots

    def get_config_schema(self) -> ConfigSchema:
        return ConfigSchema(
            name="agent",
            fields=[
                ConfigField(
                    name="max_delegations",
                    type="integer",
                    label="Max Delegations",
                    description="Maximum number of sub-agent delegations per turn.",
                    default=4,
                    min_value=0,
                ),
            ],
        )

    def get_config(self) -> Dict[str, Any]:
        return {"max_delegations": self._max_delegations}

    def update_config(self, config: Dict[str, Any]) -> None:
        if "max_delegations" in config:
            self._max_delegations = int(config["max_delegations"])

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
