"""Prompt builders — concrete implementations for system prompt assembly."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from geny_executor.core.state import PipelineState
from geny_executor.stages.s03_system.interface import PromptBlock, PromptBuilder


class StaticPromptBuilder(PromptBuilder):
    """Returns a fixed system prompt."""

    def __init__(self, prompt: str):
        self._prompt = prompt

    @property
    def name(self) -> str:
        return "static"

    @property
    def description(self) -> str:
        return "Fixed system prompt"

    def build(self, state: PipelineState) -> str:
        return self._prompt


class PersonaBlock(PromptBlock):
    """Character/role persona."""

    def __init__(self, persona: str):
        self._persona = persona

    @property
    def name(self) -> str:
        return "persona"

    def render(self, state: PipelineState) -> str:
        return self._persona


class RulesBlock(PromptBlock):
    """Rules and constraints."""

    def __init__(self, rules: List[str]):
        self._rules = rules

    @property
    def name(self) -> str:
        return "rules"

    def render(self, state: PipelineState) -> str:
        lines = ["# Rules"]
        for i, rule in enumerate(self._rules, 1):
            lines.append(f"{i}. {rule}")
        return "\n".join(lines)


class DateTimeBlock(PromptBlock):
    """Current date/time injection."""

    @property
    def name(self) -> str:
        return "datetime"

    def render(self, state: PipelineState) -> str:
        now = datetime.now(timezone.utc)
        return f"Current date: {now.strftime('%Y-%m-%d %H:%M UTC')}"


class MemoryContextBlock(PromptBlock):
    """Inject memory context from state."""

    @property
    def name(self) -> str:
        return "memory_context"

    def render(self, state: PipelineState) -> str:
        memory_ctx = state.metadata.get("memory_context", "")
        if not memory_ctx:
            return ""
        return f"# Relevant Knowledge\n{memory_ctx}"


class ToolInstructionsBlock(PromptBlock):
    """Tool usage instructions."""

    def __init__(self, instructions: str = ""):
        self._instructions = instructions

    @property
    def name(self) -> str:
        return "tool_instructions"

    def render(self, state: PipelineState) -> str:
        if self._instructions:
            return f"# Tool Usage\n{self._instructions}"
        if state.tools:
            return (
                "# Tool Usage\n"
                "You have access to tools. Use them when appropriate to accomplish tasks."
            )
        return ""


class CustomBlock(PromptBlock):
    """User-defined custom block."""

    def __init__(self, block_name: str, content: str):
        self._name = block_name
        self._content = content

    @property
    def name(self) -> str:
        return self._name

    def render(self, state: PipelineState) -> str:
        return self._content


class ComposablePromptBuilder(PromptBuilder):
    """Composable builder — assembles blocks in order.

    Supports two output modes:
      - String mode: concatenates all blocks with separators
      - Content blocks mode: wraps each block as a content block with cache_control
    """

    def __init__(
        self,
        blocks: Optional[List[PromptBlock]] = None,
        separator: str = "\n\n",
        use_content_blocks: bool = False,
    ):
        self._blocks = list(blocks or [])
        self._separator = separator
        self._use_content_blocks = use_content_blocks

    @property
    def name(self) -> str:
        return "composable"

    @property
    def description(self) -> str:
        names = [b.name for b in self._blocks]
        return f"Composable blocks: {', '.join(names)}"

    def add_block(self, block: PromptBlock) -> ComposablePromptBuilder:
        """Append a block and return self for chaining."""
        self._blocks.append(block)
        return self

    def build(self, state: PipelineState) -> Union[str, List[Dict[str, Any]]]:
        rendered = []
        for block in self._blocks:
            text = block.render(state)
            if text:
                rendered.append((block, text))

        if not rendered:
            return ""

        if self._use_content_blocks:
            content_blocks: List[Dict[str, Any]] = []
            for block, text in rendered:
                cb: Dict[str, Any] = {"type": "text", "text": text}
                if block.cache_control:
                    cb["cache_control"] = block.cache_control
                content_blocks.append(cb)
            return content_blocks

        return self._separator.join(text for _, text in rendered)
