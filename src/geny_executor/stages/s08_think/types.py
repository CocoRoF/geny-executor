"""Think stage data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ThinkingBlock:
    """A single thinking content block from the API response."""

    text: str
    budget_tokens_used: int = 0


@dataclass
class ThinkingResult:
    """Result of thinking processing."""

    thinking_blocks: List[ThinkingBlock] = field(default_factory=list)
    response_blocks: List[Dict[str, Any]] = field(default_factory=list)
    total_thinking_tokens: int = 0
