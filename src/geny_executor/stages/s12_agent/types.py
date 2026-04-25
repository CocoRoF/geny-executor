"""Agent stage data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentResult:
    """Result of agent orchestration."""

    delegated: bool = False
    sub_results: List[Dict[str, Any]] = field(default_factory=list)
    evaluation_input: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
