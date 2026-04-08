"""Stage 3: System — assemble system prompt."""

from geny_executor.stages.s03_system.stage import SystemStage
from geny_executor.stages.s03_system.builders import (
    PromptBuilder,
    StaticPromptBuilder,
    ComposablePromptBuilder,
    PromptBlock,
    PersonaBlock,
    RulesBlock,
    DateTimeBlock,
    MemoryContextBlock,
    ToolInstructionsBlock,
)

__all__ = [
    "SystemStage",
    "PromptBuilder",
    "StaticPromptBuilder",
    "ComposablePromptBuilder",
    "PromptBlock",
    "PersonaBlock",
    "RulesBlock",
    "DateTimeBlock",
    "MemoryContextBlock",
    "ToolInstructionsBlock",
]
