"""Stage 11: Agent — Multi-Agent orchestration."""

from geny_executor.stages.s11_agent.interface import AgentOrchestrator, SubPipelineFactory
from geny_executor.stages.s11_agent.subagent_type import (
    SubagentTypeDescriptor,
    SubagentTypeOrchestrator,
    SubagentTypeRegistry,
)
from geny_executor.stages.s11_agent.types import AgentResult
from geny_executor.stages.s11_agent.artifact.default.stage import AgentStage
from geny_executor.stages.s11_agent.artifact.default.orchestrators import (
    SingleAgentOrchestrator,
    DelegateOrchestrator,
    EvaluatorOrchestrator,
    DefaultSubPipelineFactory,
)

__all__ = [
    "AgentStage",
    "AgentOrchestrator",
    "SingleAgentOrchestrator",
    "DelegateOrchestrator",
    "EvaluatorOrchestrator",
    "SubPipelineFactory",
    "DefaultSubPipelineFactory",
    "SubagentTypeDescriptor",
    "SubagentTypeOrchestrator",
    "SubagentTypeRegistry",
    "AgentResult",
]
