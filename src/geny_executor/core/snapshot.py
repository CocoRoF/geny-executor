"""PipelineSnapshot — save / restore pipeline configuration state.

A snapshot captures the *configuration surface* of a pipeline (which stages
are registered, which strategy implementations are selected, their configs,
and the PipelineConfig) so it can be serialized and later restored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List


@dataclass
class StageSnapshot:
    """Configuration state of a single stage."""

    order: int
    name: str
    is_active: bool
    strategies: Dict[str, str] = field(default_factory=dict)  # slot_name → impl_name
    strategy_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # slot_name → config
    stage_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineSnapshot:
    """Serializable snapshot of the full pipeline configuration."""

    pipeline_name: str
    stages: List[StageSnapshot] = field(default_factory=list)
    pipeline_config: Dict[str, Any] = field(default_factory=dict)
    model_config: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    description: str = ""
    version: str = "1.0"

    # ── Serialization ──────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return {
            "version": self.version,
            "pipeline_name": self.pipeline_name,
            "created_at": self.created_at,
            "description": self.description,
            "pipeline_config": self.pipeline_config,
            "model_config": self.model_config,
            "stages": [
                {
                    "order": s.order,
                    "name": s.name,
                    "is_active": s.is_active,
                    "strategies": s.strategies,
                    "strategy_configs": s.strategy_configs,
                    "stage_config": s.stage_config,
                }
                for s in self.stages
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PipelineSnapshot:
        """Reconstruct a snapshot from a dict."""
        stages = [
            StageSnapshot(
                order=s["order"],
                name=s["name"],
                is_active=s["is_active"],
                strategies=s.get("strategies", {}),
                strategy_configs=s.get("strategy_configs", {}),
                stage_config=s.get("stage_config", {}),
            )
            for s in data.get("stages", [])
        ]
        return cls(
            pipeline_name=data.get("pipeline_name", ""),
            stages=stages,
            pipeline_config=data.get("pipeline_config", {}),
            model_config=data.get("model_config", {}),
            created_at=data.get("created_at", ""),
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
        )

    @classmethod
    def from_json(cls, text: str) -> PipelineSnapshot:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(text))
