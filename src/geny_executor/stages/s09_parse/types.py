"""Parse stage data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from geny_executor.stages.s06_api.types import APIResponse


@dataclass
class ToolCall:
    """A parsed tool call."""

    tool_use_id: str
    tool_name: str
    tool_input: Dict[str, Any]


@dataclass
class ParsedResponse:
    """Fully parsed and classified API response."""

    # Text output
    text: str = ""

    # Tool calls
    tool_calls: List[ToolCall] = field(default_factory=list)

    # Completion signal
    signal: Optional[str] = None  # "continue", "complete", "blocked", "error", "delegate"
    signal_detail: Optional[str] = None

    # Stop reason
    stop_reason: str = ""  # end_turn, tool_use, max_tokens

    # Thinking
    thinking_texts: List[str] = field(default_factory=list)

    # Structured output (if parsed)
    structured_output: Optional[Any] = None

    # Phase 7 S7.3: surfaces JSON-parse OR JSON-Schema validation
    # failures so downstream stages (Stage 11 Agent, Stage 13 Loop)
    # can branch on a structured contract failure without re-parsing.
    # ``None`` when the structured output was either absent (no
    # ``StructuredOutputParser`` configured) or validated cleanly.
    structured_output_error: Optional[str] = None

    # Original response ref
    api_response: Optional[APIResponse] = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def is_complete(self) -> bool:
        return self.signal == "complete" or (
            self.stop_reason == "end_turn" and not self.has_tool_calls
        )

    @property
    def needs_tool_execution(self) -> bool:
        return self.has_tool_calls
