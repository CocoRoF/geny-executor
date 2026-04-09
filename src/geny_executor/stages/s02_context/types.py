"""Stage 2: Context — shared data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class MemoryChunk:
    """A piece of retrieved memory."""

    key: str
    content: str
    source: str = ""  # "long_term", "short_term", "vector", "file"
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
