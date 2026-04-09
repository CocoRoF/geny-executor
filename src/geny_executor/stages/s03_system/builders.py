"""Prompt builders — backward-compatible re-export wrapper."""

from geny_executor.stages.s03_system.interface import PromptBuilder, PromptBlock
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

__all__ = [
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
]
