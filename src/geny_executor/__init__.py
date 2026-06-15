"""geny-executor: Harness-engineered agent pipeline library.

Usage:
    from geny_executor import Pipeline, PipelineConfig
    from geny_executor.stages.s01_input import InputStage
    from geny_executor.stages.s06_api import APIStage, MockProvider
    from geny_executor.stages.s09_parse import ParseStage
    from geny_executor.stages.s21_yield import YieldStage

    pipeline = Pipeline(PipelineConfig(name="my-agent"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(api_key="..."))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())

    result = await pipeline.run("Hello!")
"""

from geny_executor.core.pipeline import Pipeline
from geny_executor.core.config import PipelineConfig, ModelConfig, ModelOverrides
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
    ExecutorErrorCode,
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
    HostSelections,
    ManifestIssue,
    StageManifestEntry,
    ToolsSnapshot,
    validate_manifest,
)
from geny_executor.core.manifest_factory import (
    PresetDescriptor,
    build_manifest,
    build_manifest_for,
    get_preset_descriptor,
    known_manifest_presets,
    preset_catalog,
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
from geny_executor.events import (
    EVENT_CATALOG_VERSION,
    EventBus,
    EventTypes,
    PipelineEvent,
    known_event_types,
)
from geny_executor.llm_client import (
    APIRequest,
    APIResponse,
    BaseClient,
    ClientCapabilities,
    ClientRegistry,
    ConfigError,
    ContentBlock,
    CredentialBundle,
    ProviderCredentials,
)
from geny_executor.memory import (
    GenyPresets,
    MemoryAwareRetriever,
    MemoryProviderFactory,
    ProviderDrivenStrategy,
)
from geny_executor.memory.factory import provider_from_manifest_memory
from geny_executor.stages.s12_agent.subagent_type import (
    ManifestSubagentPipelineFactory,
    SubAgentBuildContext,
    SubagentTypeDescriptor,
    SubagentTypeOrchestrator,
    SubagentTypeRegistry,
    compile_subagent_descriptors,
    resolve_subagent_provider,
)

# Single source of truth: read the installed distribution version so
# ``__version__`` can never drift from ``pyproject.toml`` again.
try:
    from importlib.metadata import PackageNotFoundError as _PkgNotFound
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("geny-executor")
except Exception:  # noqa: BLE001 — not installed (e.g. source checkout)
    __version__ = "0.0.0+local"

__all__ = [
    # Core
    "Pipeline",
    "PipelineConfig",
    "PipelineState",
    "PipelineResult",
    "ModelConfig",
    "ModelOverrides",
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
    "HostSelections",
    "ManifestIssue",
    "StageManifestEntry",
    "ToolsSnapshot",
    "validate_manifest",
    "build_manifest",
    "build_manifest_for",
    "known_manifest_presets",
    "preset_catalog",
    "get_preset_descriptor",
    "PresetDescriptor",
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
    "EVENT_CATALOG_VERSION",
    "EventBus",
    "EventTypes",
    "PipelineEvent",
    "known_event_types",
    # LLM clients (unified)
    "APIRequest",
    "APIResponse",
    "BaseClient",
    "ClientCapabilities",
    "ClientRegistry",
    "ConfigError",
    "ContentBlock",
    "CredentialBundle",
    "ProviderCredentials",
    # Errors
    "GenyExecutorError",
    "PipelineError",
    "StageError",
    "GuardRejectError",
    "APIError",
    "ToolExecutionError",
    "ErrorCategory",
    "ExecutorErrorCode",
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
    # Memory plumbing (provider-driven)
    "MemoryAwareRetriever",
    "MemoryProviderFactory",
    "ProviderDrivenStrategy",
    "GenyPresets",
    "provider_from_manifest_memory",
    # Sub-agent types (manifest-expressible since 2.2.0 Wave 3)
    "ManifestSubagentPipelineFactory",
    "SubAgentBuildContext",
    "SubagentTypeDescriptor",
    "SubagentTypeOrchestrator",
    "SubagentTypeRegistry",
    "compile_subagent_descriptors",
    "resolve_subagent_provider",
]
