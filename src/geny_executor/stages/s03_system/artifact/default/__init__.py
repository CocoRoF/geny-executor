"""Stage 3: System — default artifact."""

from geny_executor.stages.s03_system.artifact.default.stage import SystemStage
from geny_executor.stages.s03_system.artifact.default.builders import (
    StaticPromptBuilder,
    ComposablePromptBuilder,
    PersonaBlock,
    RulesBlock,
    DateTimeBlock,
    MemoryContextBlock,
    ToolInstructionsBlock,
    CustomBlock,
)

Stage = SystemStage

__all__ = [
    "Stage",
    "SystemStage",
    "StaticPromptBuilder",
    "ComposablePromptBuilder",
    "PersonaBlock",
    "RulesBlock",
    "DateTimeBlock",
    "MemoryContextBlock",
    "ToolInstructionsBlock",
    "CustomBlock",
]
