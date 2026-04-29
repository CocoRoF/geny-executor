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
    """Snapshot of the tool configuration.

    ``external`` is a whitelist of names supplied by host-side
    :class:`~geny_executor.tools.providers.AdhocToolProvider`
    implementations. Unlike ``built_in`` / ``adhoc`` / ``mcp_servers``,
    these tools are not serializable into the manifest body — the
    manifest only records *which provider-backed names are active* for
    this environment. The pipeline resolves each name against the
    ``adhoc_providers`` passed to :meth:`Pipeline.from_manifest`.
    """

    built_in: List[str] = field(default_factory=list)
    adhoc: List[Dict[str, Any]] = field(default_factory=list)
    mcp_servers: List[Dict[str, Any]] = field(default_factory=list)
    external: List[str] = field(default_factory=list)
    scope: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "built_in": list(self.built_in),
            "adhoc": list(self.adhoc),
            "mcp_servers": list(self.mcp_servers),
            "external": list(self.external),
            "scope": dict(self.scope),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ToolsSnapshot:
        return cls(
            built_in=data.get("built_in", []),
            adhoc=data.get("adhoc", []),
            mcp_servers=data.get("mcp_servers", []),
            external=data.get("external", []),
            scope=data.get("scope", {}),
        )


MANIFEST_VERSION = "3.0"
# Older versions auto-migrated by ``EnvironmentManifest.from_dict``.
# v1 → v2 added the v2 stage fields (artifact / tool_binding /
# model_override / chain_order). v2 → v3 (Sub-phase 9a / S9a.4)
# pads the stages list out to the new 21-slot layout — any of the
# five new orders missing from the payload are inserted as the
# default pass-through entry with active=False.
_LEGACY_VERSIONS = {"1.0", "2.0"}


@dataclass
class HostSelections:
    """Per-environment subset selection of host-registered resources.

    Hooks, skills, and permission rules live host-level (one set of
    files shared by every environment on this machine). Each manifest
    records which subset of those host registrations is *active for
    this environment*. The runtime intersects the host registry with
    the env selection at session boot.

    Sentinel ``["*"]`` means "use everything the host has registered,
    including future additions" — distinct from selecting every
    currently-known name individually. An empty list means "use none"
    and is a deliberate opt-out (rare but supported, e.g. a sandbox
    env that must not fire any hook).

    The default for a fresh blank env is ``["*"]`` for all three —
    the friendliest possible default ("if you registered it host-side,
    it's on by default"). Users narrow on a per-env basis when they
    need to.

    .. note:: ``permissions`` is reserved but the runtime does not yet
       intersect it. The frontend exposes a placeholder picker so the
       data shape is forward-compatible; expect real enforcement in a
       future minor release.
    """

    hooks: List[str] = field(default_factory=lambda: ["*"])
    skills: List[str] = field(default_factory=lambda: ["*"])
    permissions: List[str] = field(default_factory=lambda: ["*"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hooks": list(self.hooks),
            "skills": list(self.skills),
            "permissions": list(self.permissions),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "HostSelections":
        # Missing payload → all-on. Pre-1.3.3 manifests don't record
        # this section at all, and the runtime should treat them as
        # "use whatever the host has" (the implicit pre-1.3.3
        # behaviour). An explicit empty list, by contrast, means "none".
        if not data:
            return cls()
        return cls(
            hooks=list(data.get("hooks", ["*"])),
            skills=list(data.get("skills", ["*"])),
            permissions=list(data.get("permissions", ["*"])),
        )

    @staticmethod
    def resolve(selection: List[str], available: List[str]) -> List[str]:
        """Apply a selection list to the host's registered names.

        ``["*"]`` → every available name (future-proof).
        ``[]``    → empty (explicit opt-out).
        Otherwise → intersection of the selection and what the host has.

        Names listed in the selection but not registered host-side are
        dropped silently — the manifest may outlive a host registration
        and the runtime should keep working.
        """
        if selection == ["*"]:
            return list(available)
        if not selection:
            return []
        avail = set(available)
        return [name for name in selection if name in avail]


@dataclass
class StageManifestEntry:
    """Structured stage entry in a v2 environment manifest.

    Mirrors :class:`StageSnapshot` but uses manifest-native field names
    (e.g. ``active`` / ``config``) for backward compat with v1 consumers.
    """

    order: int
    name: str
    active: bool = True
    artifact: str = "default"
    strategies: Dict[str, str] = field(default_factory=dict)
    strategy_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    tool_binding: Optional[Dict[str, Any]] = None
    model_override: Optional[Dict[str, Any]] = None
    chain_order: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order": self.order,
            "name": self.name,
            "active": self.active,
            "artifact": self.artifact,
            "strategies": dict(self.strategies),
            "strategy_configs": {k: dict(v) for k, v in self.strategy_configs.items()},
            "config": dict(self.config),
            "tool_binding": self.tool_binding,
            "model_override": self.model_override,
            "chain_order": {k: list(v) for k, v in self.chain_order.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StageManifestEntry:
        return cls(
            order=int(data.get("order", 0)),
            name=str(data.get("name", "")),
            active=bool(data.get("active", True)),
            artifact=str(data.get("artifact", "default")),
            strategies=dict(data.get("strategies", {})),
            strategy_configs={k: dict(v) for k, v in data.get("strategy_configs", {}).items()},
            config=dict(data.get("config", {})),
            tool_binding=data.get("tool_binding"),
            model_override=data.get("model_override"),
            chain_order={k: list(v) for k, v in data.get("chain_order", {}).items()},
        )


def _migrate_v1_to_v2(data: Dict[str, Any]) -> Dict[str, Any]:
    """Upgrade a v1 manifest dict to v2 shape in place.

    v1 manifests lack the ``artifact``/``tool_binding``/``model_override``/
    ``chain_order`` fields on each stage; default them conservatively. No
    behavioural defaults are injected — the v1 payload's existing strategies
    and configs are preserved byte-for-byte.
    """
    data = copy.deepcopy(data)
    stages = data.get("stages", [])
    migrated: List[Dict[str, Any]] = []
    for entry in stages:
        migrated.append(
            {
                "order": entry.get("order", 0),
                "name": entry.get("name", ""),
                "active": entry.get("active", True),
                "artifact": entry.get("artifact", "default"),
                "strategies": entry.get("strategies", {}),
                "strategy_configs": entry.get("strategy_configs", {}),
                "config": entry.get("config", {}),
                "tool_binding": entry.get("tool_binding"),
                "model_override": entry.get("model_override"),
                "chain_order": entry.get("chain_order", {}),
            }
        )
    data["stages"] = migrated
    data["version"] = "2.0"
    return data


# v2 → v3 (Sub-phase 9a / S9a.4): the canonical layout grew from 16
# slots to 21. v2 payloads have stage entries for whichever orders
# the host serialised — typically the original 16. The migration
# pads the array out to 21 by inserting default pass-through entries
# for any of the five new orders (11/13/15/19/20) that aren't
# already present. Entries the v2 payload supplied are preserved
# byte-for-byte; only the missing orders are filled. ``active`` is
# left at its v3 default (False) so consumers must explicitly opt
# the new stages in.
_V3_NEW_ORDERS: Dict[int, str] = {
    11: "tool_review",
    13: "task_registry",
    15: "hitl",
    19: "summarize",
    20: "persist",
}


def _migrate_v2_to_v3(data: Dict[str, Any]) -> Dict[str, Any]:
    """Pad a v2 manifest's stages list out to the 21-slot v3 layout."""
    data = copy.deepcopy(data)
    stages = list(data.get("stages", []))
    seen_orders = {int(s.get("order", 0)) for s in stages}
    for order, name in _V3_NEW_ORDERS.items():
        if order in seen_orders:
            continue
        stages.append(
            {
                "order": order,
                "name": name,
                "active": False,
                "artifact": "default",
                "strategies": {},
                "strategy_configs": {},
                "config": {},
                "tool_binding": None,
                "model_override": None,
                "chain_order": {},
            }
        )
    # Keep the array sorted by order so consumers iterating in
    # declaration order see a stable layout.
    stages.sort(key=lambda s: int(s.get("order", 0)))
    data["stages"] = stages
    data["version"] = MANIFEST_VERSION
    return data


@dataclass
class EnvironmentManifest:
    """Complete environment definition — the .geny-env.json format.

    **v2 (geny-executor v0.13.0)** adds first-class template fields to each
    stage entry: ``artifact``, ``tool_binding``, ``model_override``,
    ``chain_order``. v1 payloads are silently migrated on
    :meth:`from_dict` — callers that simply load + save a legacy file will
    upgrade it on next write.
    """

    version: str = MANIFEST_VERSION
    metadata: EnvironmentMetadata = field(default_factory=EnvironmentMetadata)
    model: Dict[str, Any] = field(default_factory=dict)
    pipeline: Dict[str, Any] = field(default_factory=dict)
    stages: List[Dict[str, Any]] = field(default_factory=list)
    tools: ToolsSnapshot = field(default_factory=ToolsSnapshot)
    host_selections: HostSelections = field(default_factory=HostSelections)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "metadata": self.metadata.to_dict(),
            "model": dict(self.model),
            "pipeline": dict(self.pipeline),
            "stages": list(self.stages),
            "tools": self.tools.to_dict(),
            "host_selections": self.host_selections.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EnvironmentManifest:
        """Load + auto-migrate to the current manifest version.

        v1 → v2: adds the v2 stage fields. v2 → v3: pads the stages
        list out to the 21-slot layout. Migrations are chained so
        a v1 payload upgrades all the way to the current version in
        one call.

        ``host_selections`` is read-or-default: pre-1.3.3 manifests
        omit the field and load with the all-on default, matching the
        implicit "host hooks/skills always apply" behaviour of those
        older versions. No version bump is needed because the change
        is a pure additive default.
        """
        version = str(data.get("version", "1.0"))
        if version == "1.0":
            data = _migrate_v1_to_v2(data)
            version = "2.0"
        if version == "2.0":
            data = _migrate_v2_to_v3(data)
            version = MANIFEST_VERSION
        return cls(
            version=version,
            metadata=EnvironmentMetadata.from_dict(data.get("metadata", {})),
            model=data.get("model", {}),
            pipeline=data.get("pipeline", {}),
            stages=data.get("stages", []),
            tools=ToolsSnapshot.from_dict(data.get("tools", {})),
            host_selections=HostSelections.from_dict(data.get("host_selections")),
        )

    # ── Structured stage access ─────────────────────────────

    def stage_entries(self) -> List[StageManifestEntry]:
        """Return stages as typed :class:`StageManifestEntry` objects."""
        return [StageManifestEntry.from_dict(s) for s in self.stages]

    def set_stage_entries(self, entries: List[StageManifestEntry]) -> None:
        """Replace the stages list from typed entries (back to dict form)."""
        self.stages = [e.to_dict() for e in entries]

    @classmethod
    def from_snapshot(
        cls,
        snapshot: PipelineSnapshot,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        tools: Optional[ToolsSnapshot] = None,
    ) -> EnvironmentManifest:
        """Create a v2 manifest from a PipelineSnapshot."""
        env_id = f"env_{uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()

        stages = []
        for s in snapshot.stages:
            entry = StageManifestEntry(
                order=s.order,
                name=s.name,
                active=s.is_active,
                artifact=s.artifact,
                strategies=dict(s.strategies),
                strategy_configs={k: dict(v) for k, v in s.strategy_configs.items()},
                config=dict(s.stage_config),
                tool_binding=s.tool_binding,
                model_override=s.model_override,
                chain_order={k: list(v) for k, v in s.chain_order.items()},
            )
            stages.append(entry.to_dict())

        return cls(
            version=MANIFEST_VERSION,
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

    @classmethod
    def blank_manifest(
        cls,
        name: str,
        *,
        description: str = "",
        tags: Optional[List[str]] = None,
        model: Optional[Dict[str, Any]] = None,
        pipeline: Optional[Dict[str, Any]] = None,
    ) -> EnvironmentManifest:
        """Build a 21-stage template with the structurally required stages on.

        Every stage is populated with its default artifact plus the artifact's
        default strategy implementations and config, so a UI can render all
        21 rows immediately and the user only has to edit fields — no
        "missing required field" errors the moment a stage is flipped active.

        Four stages — ``s01_input``, ``s06_api``, ``s09_parse``, ``s21_yield``
        — are load-bearing for every pipeline (see
        :data:`~geny_executor.core.introspection._STAGE_REQUIRED`) and default
        to ``active=True``; every other stage defaults to ``active=False`` so
        the user explicitly opts in. Requiring the UI to flip the required
        four on for every new blank env was the source of confusion that
        motivated this default — the runtime can't function without them, so
        the template shouldn't pretend they're optional.

        Unlike :meth:`from_snapshot`, ``blank_manifest`` never sets
        ``metadata.base_preset`` — a blank environment has no origin preset.

        ``tools.built_in`` defaults to ``["*"]`` (wildcard) — every built-in
        tool, including future additions, is exposed to the LLM at stage 10.
        The user can still narrow the whitelist by replacing the wildcard
        with explicit names. An empty list means the agent has no built-in
        tools, which is rarely what a fresh template wants.

        Session-less: construction goes through
        :func:`~geny_executor.core.introspection.introspect_all`, so no live
        :class:`Pipeline` is required.

        Raises:
            Any import-time error surfaced by :func:`introspect_all` — the
            library itself must be importable before the UI can call this.
        """
        from geny_executor.core.introspection import introspect_all

        env_id = f"env_{uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()

        stages: List[Dict[str, Any]] = []
        for insp in introspect_all():
            entry = StageManifestEntry(
                order=insp.order,
                name=insp.name,
                active=insp.required,
                artifact=insp.artifact,
                strategies={
                    slot: slot_info.current_impl
                    for slot, slot_info in insp.strategy_slots.items()
                    if slot_info.current_impl
                },
                strategy_configs={},
                config=dict(insp.config),
            )
            stages.append(entry.to_dict())

        return cls(
            version=MANIFEST_VERSION,
            metadata=EnvironmentMetadata(
                id=env_id,
                name=name,
                description=description,
                tags=list(tags or []),
                created_at=now,
                updated_at=now,
                base_preset="",
            ),
            model=dict(model) if model else {},
            pipeline=dict(pipeline) if pipeline else {},
            stages=stages,
            tools=ToolsSnapshot(built_in=["*"]),
            host_selections=HostSelections(),  # all wildcards by default
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
                    artifact=s.get("artifact", "default"),
                    tool_binding=s.get("tool_binding"),
                    model_override=s.get("model_override"),
                    chain_order=s.get("chain_order", {}),
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
