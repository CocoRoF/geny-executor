"""Stage 11: Agent — Multi-Agent orchestration."""

from geny_executor.stages.s12_agent.interface import AgentOrchestrator, SubPipelineFactory
from geny_executor.stages.s12_agent.subagent_type import (
    ManifestSubagentPipelineFactory,
    PipelineFactory,
    SubAgentBuildContext,
    SubagentTypeDescriptor,
    SubagentTypeOrchestrator,
    SubagentTypeRegistry,
    compile_subagent_descriptors,
    resolve_subagent_provider,
)
from geny_executor.stages.s12_agent.types import AgentResult
from geny_executor.stages.s12_agent.artifact.default.stage import AgentStage
from geny_executor.stages.s12_agent.artifact.default.orchestrators import (
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
    "ManifestSubagentPipelineFactory",
    "PipelineFactory",
    "SubAgentBuildContext",
    "SubagentTypeDescriptor",
    "SubagentTypeOrchestrator",
    "SubagentTypeRegistry",
    "AgentResult",
    "compile_subagent_descriptors",
    "resolve_subagent_provider",
]
