"""PipelineSnapshot — save / restore pipeline configuration state.

A snapshot captures the *configuration surface* of a pipeline (which stages
are registered, which strategy implementations are selected, their configs,
and the PipelineConfig) so it can be serialized and later restored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


DEFAULT_ARTIFACT_NAME = "default"


@dataclass
class StageSnapshot:
    """Configuration state of a single stage.

    v2 additions (Environment Builder):
        * ``artifact`` — which artifact produced this stage (default ``"default"``).
        * ``tool_binding`` — ``StageToolBinding.to_dict()`` payload or ``None``.
        * ``model_override`` — ``ModelConfig.to_dict()`` payload or ``None``.
        * ``chain_order`` — chain_name → ordered item names, for chain stages.

    These fields default to safe "nothing overridden" values so v1 snapshots
    that lack them continue to load unchanged.
    """

    order: int
    name: str
    is_active: bool
    strategies: Dict[str, str] = field(default_factory=dict)  # slot_name → impl_name
    strategy_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # slot_name → config
    stage_config: Dict[str, Any] = field(default_factory=dict)
    artifact: str = DEFAULT_ARTIFACT_NAME
    tool_binding: Optional[Dict[str, Any]] = None
    model_override: Optional[Dict[str, Any]] = None
    chain_order: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class PipelineSnapshot:
    """Serializable snapshot of the full pipeline configuration."""

    pipeline_name: str
    stages: List[StageSnapshot] = field(default_factory=list)
    pipeline_config: Dict[str, Any] = field(default_factory=dict)
    model_config: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    description: str = ""
    version: str = "2.0"

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
                    "artifact": s.artifact,
                    "tool_binding": s.tool_binding,
                    "model_override": s.model_override,
                    "chain_order": s.chain_order,
                }
                for s in self.stages
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PipelineSnapshot:
        """Reconstruct a snapshot from a dict.

        Missing v2 fields (from a v1 payload) are filled with safe defaults:
        ``artifact="default"``, ``tool_binding=None``, ``model_override=None``,
        ``chain_order={}``. No warning is emitted — the migration is silent.
        """
        stages = [
            StageSnapshot(
                order=s["order"],
                name=s["name"],
                is_active=s["is_active"],
                strategies=s.get("strategies", {}),
                strategy_configs=s.get("strategy_configs", {}),
                stage_config=s.get("stage_config", {}),
                artifact=s.get("artifact", DEFAULT_ARTIFACT_NAME),
                tool_binding=s.get("tool_binding"),
                model_override=s.get("model_override"),
                chain_order=s.get("chain_order", {}),
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
            version=data.get("version", "2.0"),
        )

    @classmethod
    def from_json(cls, text: str) -> PipelineSnapshot:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(text))
