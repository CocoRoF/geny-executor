"""API stage data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from geny_executor.core.state import TokenUsage


@dataclass
class APIRequest:
    """Request to Anthropic Messages API."""

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

    # Extended thinking
    thinking: Optional[Dict[str, Any]] = None

    # Metadata
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ContentBlock:
    """A single content block in the response."""

    type: str  # "text", "tool_use", "thinking"
    text: Optional[str] = None

    # tool_use fields
    tool_use_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None

    # thinking fields
    thinking_text: Optional[str] = None

    # raw
    raw: Optional[Dict[str, Any]] = None


@dataclass
class APIResponse:
    """Parsed response from Anthropic Messages API."""

    # Content
    content: List[ContentBlock] = field(default_factory=list)
    stop_reason: str = ""  # end_turn, tool_use, max_tokens, stop_sequence

    # Usage
    usage: TokenUsage = field(default_factory=TokenUsage)

    # Model info
    model: str = ""
    message_id: str = ""

    # Raw response (for debugging)
    raw: Optional[Any] = None

    @property
    def text(self) -> str:
        """Extract concatenated text from text blocks."""
        parts = []
        for block in self.content:
            if block.type == "text" and block.text:
                parts.append(block.text)
        return "\n".join(parts) if parts else ""

    @property
    def tool_calls(self) -> List[ContentBlock]:
        """Extract tool_use blocks."""
        return [b for b in self.content if b.type == "tool_use"]

    @property
    def thinking_blocks(self) -> List[ContentBlock]:
        """Extract thinking blocks."""
        return [b for b in self.content if b.type == "thinking"]

    @property
    def has_tool_calls(self) -> bool:
        return self.stop_reason == "tool_use" or bool(self.tool_calls)
