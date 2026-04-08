"""Input stage data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class NormalizedInput:
    """Validated and normalized user input."""

    text: str
    role: str = "user"

    # Multimodal content
    images: List[Dict[str, Any]] = field(default_factory=list)
    files: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    source: str = "user"  # "user", "system", "agent", "broadcast"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Original raw input (before normalization)
    raw_input: Optional[Any] = None

    def to_message_content(self) -> Any:
        """Convert to Anthropic API message content format."""
        if not self.images:
            return self.text

        # Multimodal: content blocks
        blocks: List[Dict[str, Any]] = []
        for img in self.images:
            blocks.append(img)
        blocks.append({"type": "text", "text": self.text})
        return blocks
