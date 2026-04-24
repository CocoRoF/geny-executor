"""Stage 3: System — assemble system prompt."""

from geny_executor.stages.s03_system.interface import PromptBuilder, PromptBlock
from geny_executor.stages.s03_system.artifact.default import (
    SystemStage,
    StaticPromptBuilder,
    ComposablePromptBuilder,
    PersonaBlock,
    RulesBlock,
    DateTimeBlock,
    MemoryContextBlock,
    ToolInstructionsBlock,
    CustomBlock,
)
from geny_executor.stages.s03_system.persona import (
    DynamicPersonaPromptBuilder,
    PersonaProvider,
    PersonaResolution,
)

__all__ = [
    "SystemStage",
    "PromptBuilder",
    "PromptBlock",
    "StaticPromptBuilder",
    "ComposablePromptBuilder",
    "PersonaBlock",
    "RulesBlock",
    "DateTimeBlock",
    "MemoryContextBlock",
    "ToolInstructionsBlock",
    "CustomBlock",
    "DynamicPersonaPromptBuilder",
    "PersonaProvider",
    "PersonaResolution",
]
