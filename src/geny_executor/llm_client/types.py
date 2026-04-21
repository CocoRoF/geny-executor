"""Canonical LLM request / response types.

These mirror the Anthropic Messages API shape and serve as the single
provider-neutral format flowing through every ``BaseClient`` subclass.
Formerly lived at ``geny_executor.stages.s06_api.types``; that module
now re-exports from here during the PR-3→PR-4 migration window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from geny_executor.core.state import TokenUsage


@dataclass
class APIRequest:
    """Canonical request bundle (Anthropic-shaped)."""

    model: str
    messages: List[Dict[str, Any]]
    max_tokens: int = 8192
    system: Any = ""  # str or List[content blocks]
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    stop_sequences: Optional[List[str]] = None
    stream: bool = False

    thinking: Optional[Dict[str, Any]] = None

    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ContentBlock:
    """A single content block in an API response."""

    type: str  # "text", "tool_use", "thinking"
    text: Optional[str] = None

    tool_use_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None

    thinking_text: Optional[str] = None

    raw: Optional[Dict[str, Any]] = None


@dataclass
class APIResponse:
    """Canonical response bundle."""

    content: List[ContentBlock] = field(default_factory=list)
    stop_reason: str = ""

    usage: TokenUsage = field(default_factory=TokenUsage)

    model: str = ""
    message_id: str = ""

    raw: Optional[Any] = None

    @property
    def text(self) -> str:
        parts = []
        for block in self.content:
            if block.type == "text" and block.text:
                parts.append(block.text)
        return "\n".join(parts) if parts else ""

    @property
    def tool_calls(self) -> List[ContentBlock]:
        return [b for b in self.content if b.type == "tool_use"]

    @property
    def thinking_blocks(self) -> List[ContentBlock]:
        return [b for b in self.content if b.type == "thinking"]

    @property
    def has_tool_calls(self) -> bool:
        return self.stop_reason == "tool_use" or bool(self.tool_calls)
