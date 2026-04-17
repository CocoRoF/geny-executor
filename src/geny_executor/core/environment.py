"""Environment system — serialize, manage, and apply pipeline environments.

An *environment* is a complete, portable description of a pipeline configuration:
model settings, stage strategies, tool setup, and pipeline parameters. It wraps
a PipelineSnapshot with rich metadata, variable references, and tool definitions.
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from geny_executor.core.diff import EnvironmentDiff
from geny_executor.core.snapshot import PipelineSnapshot, StageSnapshot


# ═══════════════════════════════════════════════════════════
#  Data classes
# ═══════════════════════════════════════════════════════════


@dataclass
class EnvironmentMetadata:
    """Metadata about an environment."""

    id: str = ""
    name: str = ""
    description: str = ""
    author: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    base_preset: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "tags": list(self.tags),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "base_preset": self.base_preset,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EnvironmentMetadata:
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            author=data.get("author", ""),
            tags=data.get("tags", []),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            base_preset=data.get("base_preset", ""),
        )


@dataclass
class ToolsSnapshot:
    """Snapshot of the tool configuration."""

    built_in: List[str] = field(default_factory=list)
    adhoc: List[Dict[str, Any]] = field(default_factory=list)
    mcp_servers: List[Dict[str, Any]] = field(default_factory=list)
    scope: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "built_in": list(self.built_in),
            "adhoc": list(self.adhoc),
            "mcp_servers": list(self.mcp_servers),
            "scope": dict(self.scope),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ToolsSnapshot:
        return cls(
            built_in=data.get("built_in", []),
            adhoc=data.get("adhoc", []),
            mcp_servers=data.get("mcp_servers", []),
            scope=data.get("scope", {}),
        )


@dataclass
class EnvironmentManifest:
    """Complete environment definition — the .geny-env.json format."""

    version: str = "1.0"
    metadata: EnvironmentMetadata = field(default_factory=EnvironmentMetadata)
    model: Dict[str, Any] = field(default_factory=dict)
    pipeline: Dict[str, Any] = field(default_factory=dict)
    stages: List[Dict[str, Any]] = field(default_factory=list)
    tools: ToolsSnapshot = field(default_factory=ToolsSnapshot)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "metadata": self.metadata.to_dict(),
            "model": dict(self.model),
            "pipeline": dict(self.pipeline),
            "stages": list(self.stages),
            "tools": self.tools.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EnvironmentManifest:
        return cls(
            version=data.get("version", "1.0"),
            metadata=EnvironmentMetadata.from_dict(data.get("metadata", {})),
            model=data.get("model", {}),
            pipeline=data.get("pipeline", {}),
            stages=data.get("stages", []),
            tools=ToolsSnapshot.from_dict(data.get("tools", {})),
        )

    @classmethod
    def from_snapshot(
        cls,
        snapshot: PipelineSnapshot,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        tools: Optional[ToolsSnapshot] = None,
    ) -> EnvironmentManifest:
        """Create a manifest from a PipelineSnapshot."""
        env_id = f"env_{uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()

        stages = []
        for s in snapshot.stages:
            stages.append(
                {
                    "order": s.order,
                    "name": s.name,
                    "active": s.is_active,
                    "strategies": s.strategies,
                    "strategy_configs": s.strategy_configs,
                    "config": s.stage_config,
                }
            )

        return cls(
            version="1.0",
            metadata=EnvironmentMetadata(
                id=env_id,
                name=name,
                description=description,
                tags=tags or [],
                created_at=now,
                updated_at=now,
                base_preset=snapshot.pipeline_name,
            ),
            model=dict(snapshot.model_config),
            pipeline=dict(snapshot.pipeline_config),
            stages=stages,
            tools=tools or ToolsSnapshot(),
        )

    def to_snapshot(self) -> PipelineSnapshot:
        """Convert back to a PipelineSnapshot for restoration."""
        stages = []
        for s in self.stages:
            stages.append(
                StageSnapshot(
                    order=s.get("order", 0),
                    name=s.get("name", ""),
                    is_active=s.get("active", True),
                    strategies=s.get("strategies", {}),
                    strategy_configs=s.get("strategy_configs", {}),
                    stage_config=s.get("config", {}),
                )
            )

        return PipelineSnapshot(
            pipeline_name=self.metadata.base_preset or self.metadata.name,
            stages=stages,
            pipeline_config=dict(self.pipeline),
            model_config=dict(self.model),
            created_at=self.metadata.created_at,
            description=self.metadata.description,
        )

    def update(self, changes: Dict[str, Any]) -> None:
        """Apply partial updates."""
        if "metadata" in changes:
            meta = changes["metadata"]
            if "name" in meta:
                self.metadata.name = meta["name"]
            if "description" in meta:
                self.metadata.description = meta["description"]
            if "tags" in meta:
                self.metadata.tags = meta["tags"]
            if "author" in meta:
                self.metadata.author = meta["author"]
        if "model" in changes:
            self.model.update(changes["model"])
        if "pipeline" in changes:
            self.pipeline.update(changes["pipeline"])
        self.metadata.updated_at = datetime.now(timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════════
#  EnvironmentResolver — ${VAR} expansion
# ═══════════════════════════════════════════════════════════


class EnvironmentResolver:
    """Resolves ${VAR_NAME} references in environment data."""

    PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")

    @classmethod
    def resolve(
        cls, data: Dict[str, Any], env_vars: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Replace all ${VAR} references with actual values."""
        env = {**os.environ, **(env_vars or {})}
        return cls._walk(data, env)

    @classmethod
    def _walk(cls, obj: Any, env: Dict[str, str]) -> Any:
        if isinstance(obj, str):
            return cls.PATTERN.sub(lambda m: env.get(m.group(1), m.group(0)), obj)
        elif isinstance(obj, dict):
            return {k: cls._walk(v, env) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [cls._walk(item, env) for item in obj]
        return obj

    @classmethod
    def extract_variables(cls, data: Dict[str, Any]) -> Set[str]:
        """Extract all referenced variable names from an environment."""
        variables: Set[str] = set()

        def walk(obj: Any) -> None:
            if isinstance(obj, str):
                variables.update(cls.PATTERN.findall(obj))
            elif isinstance(obj, dict):
                for v in obj.values():
                    walk(v)
            elif isinstance(obj, list):
                for item in obj:
                    walk(item)

        walk(data)
        return variables


# ═══════════════════════════════════════════════════════════
#  EnvironmentManager — CRUD + apply
# ═══════════════════════════════════════════════════════════


@dataclass
class EnvironmentSummary:
    """Lightweight summary of an environment for listing."""

    id: str
    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    model: str = ""
    stage_count: int = 0
    tool_count: int = 0
    created_at: str = ""
    updated_at: str = ""


class EnvironmentManager:
    """Manages environment storage, loading, and application."""

    def __init__(self, storage_path: str = "./environments") -> None:
        self._storage = Path(storage_path)
        self._storage.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, EnvironmentManifest] = {}

    # ── CRUD ───────────────────────────────────────────────

    def save(
        self,
        snapshot: PipelineSnapshot,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        tools: Optional[ToolsSnapshot] = None,
    ) -> str:
        """Save a pipeline snapshot as an environment. Returns env_id."""
        manifest = EnvironmentManifest.from_snapshot(snapshot, name, description, tags, tools)
        env_id = manifest.metadata.id

        path = self._storage / f"{env_id}.json"
        path.write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._cache[env_id] = manifest
        return env_id

    def load(self, env_id: str) -> EnvironmentManifest:
        """Load an environment by ID."""
        if env_id in self._cache:
            return self._cache[env_id]

        path = self._storage / f"{env_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Environment not found: {env_id}")

        data = json.loads(path.read_text(encoding="utf-8"))
        manifest = EnvironmentManifest.from_dict(data)
        self._cache[env_id] = manifest
        return manifest

    def list_all(self) -> List[EnvironmentSummary]:
        """List all stored environments."""
        envs: List[EnvironmentSummary] = []
        for path in sorted(self._storage.glob("env_*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                meta = data.get("metadata", {})
                tools = data.get("tools", {})
                envs.append(
                    EnvironmentSummary(
                        id=meta.get("id", path.stem),
                        name=meta.get("name", "Unnamed"),
                        description=meta.get("description", ""),
                        tags=meta.get("tags", []),
                        model=data.get("model", {}).get("model", ""),
                        stage_count=len(data.get("stages", [])),
                        tool_count=(len(tools.get("built_in", [])) + len(tools.get("adhoc", []))),
                        created_at=meta.get("created_at", ""),
                        updated_at=meta.get("updated_at", ""),
                    )
                )
            except (json.JSONDecodeError, KeyError):
                continue
        return envs

    def delete(self, env_id: str) -> bool:
        """Delete an environment. Returns True if deleted."""
        path = self._storage / f"{env_id}.json"
        if path.exists():
            path.unlink()
            self._cache.pop(env_id, None)
            return True
        return False

    def update(self, env_id: str, changes: Dict[str, Any]) -> EnvironmentManifest:
        """Partially update an environment."""
        manifest = self.load(env_id)
        manifest.update(changes)

        path = self._storage / f"{env_id}.json"
        path.write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._cache[env_id] = manifest
        return manifest

    # ── Import / Export ────────────────────────────────────

    def export_json(self, env_id: str) -> str:
        """Export an environment as a JSON string (variables unresolved)."""
        manifest = self.load(env_id)
        return json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2)

    def import_json(self, json_str: str, override_name: Optional[str] = None) -> str:
        """Import an environment from JSON. Returns new env_id."""
        data = json.loads(json_str)

        new_id = f"env_{uuid4().hex[:8]}"
        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["id"] = new_id
        if override_name:
            data["metadata"]["name"] = override_name
        data["metadata"]["updated_at"] = datetime.now(timezone.utc).isoformat()

        manifest = EnvironmentManifest.from_dict(data)
        manifest.metadata.id = new_id  # ensure consistency

        path = self._storage / f"{new_id}.json"
        path.write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._cache[new_id] = manifest
        return new_id

    # ── Diff ───────────────────────────────────────────────

    def diff(self, env_id_a: str, env_id_b: str) -> EnvironmentDiff:
        """Compare two environments."""
        a = self.load(env_id_a).to_dict()
        b = self.load(env_id_b).to_dict()
        return EnvironmentDiff.compute(a, b)

    # ── Apply ──────────────────────────────────────────────

    def resolve_and_load(
        self,
        env_id: str,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> EnvironmentManifest:
        """Load an environment with variable references resolved."""
        manifest = self.load(env_id)
        resolved_data = EnvironmentResolver.resolve(manifest.to_dict(), env_vars)
        return EnvironmentManifest.from_dict(resolved_data)

    def get_required_variables(self, env_id: str) -> Set[str]:
        """Get the set of ${VAR} references used in an environment."""
        manifest = self.load(env_id)
        return EnvironmentResolver.extract_variables(manifest.to_dict())


# ═══════════════════════════════════════════════════════════
#  Sanitizer — remove sensitive data for sharing
# ═══════════════════════════════════════════════════════════


class EnvironmentSanitizer:
    """Removes or masks sensitive values from environment data for sharing."""

    SENSITIVE_KEYS = {"api_key", "token", "secret", "password", "credential"}

    @classmethod
    def sanitize(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Return a deep copy with sensitive values replaced by ${PLACEHOLDER}."""
        sanitized = copy.deepcopy(data)
        cls._walk(sanitized)
        return sanitized

    @classmethod
    def _walk(cls, obj: Any) -> None:
        if isinstance(obj, dict):
            for key in list(obj.keys()):
                lower = key.lower()
                if any(s in lower for s in cls.SENSITIVE_KEYS):
                    obj[key] = "${" + key.upper() + "}"
                else:
                    cls._walk(obj[key])
        elif isinstance(obj, list):
            for item in obj:
                cls._walk(item)
