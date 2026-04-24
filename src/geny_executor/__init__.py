"""geny-executor: Harness-engineered agent pipeline library.

Usage:
    from geny_executor import Pipeline, PipelineConfig
    from geny_executor.stages.s01_input import InputStage
    from geny_executor.stages.s06_api import APIStage, MockProvider
    from geny_executor.stages.s09_parse import ParseStage
    from geny_executor.stages.s16_yield import YieldStage

    pipeline = Pipeline(PipelineConfig(name="my-agent"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(api_key="..."))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())

    result = await pipeline.run("Hello!")
"""

from geny_executor.core.pipeline import Pipeline
from geny_executor.core.config import PipelineConfig, ModelConfig
from geny_executor.core.state import PipelineState, TokenUsage, CacheMetrics
from geny_executor.core.result import PipelineResult
from geny_executor.core.stage import Stage, Strategy, StageDescription, StrategyInfo
from geny_executor.core.errors import (
    GenyExecutorError,
    PipelineError,
    StageError,
    GuardRejectError,
    APIError,
    ToolExecutionError,
    ErrorCategory,
    MutationError,
    MutationLocked,
)
from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.core.slot import SlotChain, StrategySlot
from geny_executor.core.snapshot import PipelineSnapshot, StageSnapshot
from geny_executor.core.mutation import (
    PipelineMutator,
    MutationKind,
    MutationRecord,
    MutationResult,
)
from geny_executor.core.builder import PipelineBuilder
from geny_executor.core.presets import (
    PipelinePresets,
    PresetInfo,
    PresetManager,
    PresetRegistry,
    register_preset,
)
from geny_executor.core.diff import DiffEntry, EnvironmentDiff
from geny_executor.core.environment import (
    EnvironmentManifest,
    EnvironmentManager,
    EnvironmentMetadata,
    EnvironmentResolver,
    EnvironmentSanitizer,
    EnvironmentSummary,
    StageManifestEntry,
    ToolsSnapshot,
)
from geny_executor.core.artifact import (
    ArtifactInfo,
    create_stage,
    describe_artifact,
    get_artifact_map,
    list_artifacts,
    list_artifacts_with_meta,
)
from geny_executor.core.introspection import (
    ChainIntrospection,
    IntrospectionUnsupported,
    SlotIntrospection,
    StageIntrospection,
    introspect_all,
    introspect_stage,
)
from geny_executor.events import EventBus, PipelineEvent
from geny_executor.llm_client import (
    APIRequest,
    APIResponse,
    BaseClient,
    ClientCapabilities,
    ClientRegistry,
    ContentBlock,
    ProviderBackedClient,
)
from geny_executor.memory import (
    GenyMemoryRetriever,
    GenyMemoryStrategy,
    GenyPersistence,
    GenyPresets,
)

__version__ = "0.34.0"

__all__ = [
    # Core
    "Pipeline",
    "PipelineConfig",
    "PipelineState",
    "PipelineResult",
    "ModelConfig",
    "TokenUsage",
    "CacheMetrics",
    # Abstractions
    "Stage",
    "Strategy",
    "StageDescription",
    "StrategyInfo",
    # Builder & Presets
    "PipelineBuilder",
    "PipelinePresets",
    "PresetInfo",
    "PresetManager",
    "PresetRegistry",
    "register_preset",
    # Environment & Diff
    "EnvironmentManifest",
    "EnvironmentManager",
    "EnvironmentMetadata",
    "EnvironmentResolver",
    "EnvironmentSanitizer",
    "EnvironmentSummary",
    "StageManifestEntry",
    "ToolsSnapshot",
    "DiffEntry",
    "EnvironmentDiff",
    # Artifact system
    "ArtifactInfo",
    "create_stage",
    "describe_artifact",
    "get_artifact_map",
    "list_artifacts",
    "list_artifacts_with_meta",
    # Introspection
    "ChainIntrospection",
    "IntrospectionUnsupported",
    "SlotIntrospection",
    "StageIntrospection",
    "introspect_all",
    "introspect_stage",
    # Events
    "EventBus",
    "PipelineEvent",
    # LLM clients (unified)
    "APIRequest",
    "APIResponse",
    "BaseClient",
    "ClientCapabilities",
    "ClientRegistry",
    "ContentBlock",
    "ProviderBackedClient",
    # Errors
    "GenyExecutorError",
    "PipelineError",
    "StageError",
    "GuardRejectError",
    "APIError",
    "ToolExecutionError",
    "ErrorCategory",
    "MutationError",
    "MutationLocked",
    # Schema & Mutation
    "ConfigField",
    "ConfigSchema",
    "StrategySlot",
    "SlotChain",
    "PipelineSnapshot",
    "StageSnapshot",
    "PipelineMutator",
    "MutationKind",
    "MutationRecord",
    "MutationResult",
    # Geny Memory Integration
    "GenyMemoryRetriever",
    "GenyMemoryStrategy",
    "GenyPersistence",
    "GenyPresets",
]
