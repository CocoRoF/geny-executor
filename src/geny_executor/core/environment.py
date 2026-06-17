"""Environment system — serialize, manage, and apply pipeline environments.

An *environment* is a complete, portable description of a pipeline configuration:
model settings, stage strategies, tool setup, and pipeline parameters. It wraps
a PipelineSnapshot with rich metadata, variable references, and tool definitions.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from geny_executor.core.diff import EnvironmentDiff
from geny_executor.core.snapshot import PipelineSnapshot, StageSnapshot


logger = logging.getLogger(__name__)


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
    this environment*.

    Sentinel ``["*"]`` means "use everything the host has registered,
    including future additions" — distinct from selecting every
    currently-known name individually. An empty list means "use none"
    and is a deliberate opt-out (rare but supported, e.g. a sandbox
    env that must not fire any hook).

    The default for a fresh blank env is ``["*"]`` for all three —
    the friendliest possible default ("if you registered it host-side,
    it's on by default"). Users narrow on a per-env basis when they
    need to.

    .. note:: **The library does not apply these selections itself.**
       An earlier revision of this docstring claimed "the runtime
       intersects the host registry with the env selection at session
       boot" — that was aspirational, not true (2.2.0, audit
       2026-06-09 §3.5: zero ``HostSelections.resolve`` call sites in
       the library). The honest contract: hook runners, skill sets,
       and permission rules reach the pipeline as *already-built
       runtime objects* via :meth:`Pipeline.attach_runtime`, and only
       the host knows the name/id scheme its registries use (e.g.
       Geny's permission ids are ``"{tool}::{pattern}::{behavior}"``
       strings minted host-side). Hosts therefore apply the selection
       *before* attaching: filter the registry with
       :meth:`HostSelections.resolve` (or equivalent — see Geny's
       ``service/permission/install.py``) and pass only the surviving
       objects to ``attach_runtime(hook_runner=... / permission_rules=
       ...)``. :meth:`resolve` is the supported helper for that
       filtering and is contract-tested in this repo.
    """

    hooks: List[str] = field(default_factory=lambda: ["*"])
    skills: List[str] = field(default_factory=lambda: ["*"])
    permissions: List[str] = field(default_factory=lambda: ["*"])
    #: Host-specific per-env bindings the LIBRARY DOES NOT INTERPRET. A generic
    #: extension point so a host can attach its own per-environment selections
    #: (e.g. Geny stores ``{"trigger_preset_id": "..."}`` for its VTuber
    #: thinking-trigger preset) without the manifest dropping the value on
    #: round-trip. Preserved verbatim through ``to_dict``/``from_dict``; the
    #: runtime never reads it. Keys/values are entirely the host's contract.
    #: Added in 2.6.0 — pre-2.6.0 manifests load it as ``{}``.
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "hooks": list(self.hooks),
            "skills": list(self.skills),
            "permissions": list(self.permissions),
        }
        # Only emit ``extras`` when non-empty so existing manifests aren't
        # churned with an empty key.
        if self.extras:
            out["extras"] = dict(self.extras)
        return out

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "HostSelections":
        # Missing payload → all-on. Pre-1.3.3 manifests don't record
        # this section at all, and the runtime should treat them as
        # "use whatever the host has" (the implicit pre-1.3.3
        # behaviour). An explicit empty list, by contrast, means "none".
        if not data:
            return cls()
        raw_extras = data.get("extras")
        return cls(
            hooks=list(data.get("hooks", ["*"])),
            skills=list(data.get("skills", ["*"])),
            permissions=list(data.get("permissions", ["*"])),
            extras=dict(raw_extras) if isinstance(raw_extras, dict) else {},
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


# ── from_dict hygiene (2.2.0, audit §1-1 / host_ergonomics) ─────
#
# ``from_dict`` historically accepted any payload shape and silently
# dropped keys it didn't know. That tolerance is load-bearing for
# back-compat (a newer host writing a richer manifest must still load
# on an older library), but the *silence* caused real damage: GAPT
# dual-wrote its model settings to two locations because a misspelled
# key vanished without a trace ("両쪽 다 써야 안전" — audit §1-3).
# Unknown keys are still accepted, but now warned about — once per
# load, listing every offender — so a typo'd manifest is visible at
# load time instead of three debugging sessions later.

_KNOWN_TOP_LEVEL_KEYS: Set[str] = {
    "version",
    "metadata",
    "model",
    "pipeline",
    "stages",
    "tools",
    "host_selections",
    "subagents",
    "memory",
}

_KNOWN_STAGE_ENTRY_KEYS: Set[str] = {
    "order",
    "name",
    "active",
    "artifact",
    "strategies",
    "strategy_configs",
    "config",
    "tool_binding",
    "model_override",
    "chain_order",
}


def _warn_unknown_keys(data: Dict[str, Any]) -> None:
    """Log one warning per load listing unknown top-level / stage-entry keys."""
    unknown_top = sorted(set(data.keys()) - _KNOWN_TOP_LEVEL_KEYS)
    unknown_stage: Set[str] = set()
    for entry in data.get("stages", []) or []:
        if isinstance(entry, dict):
            unknown_stage.update(set(entry.keys()) - _KNOWN_STAGE_ENTRY_KEYS)
    if not unknown_top and not unknown_stage:
        return
    parts: List[str] = []
    if unknown_top:
        parts.append(f"top-level keys {unknown_top}")
    if unknown_stage:
        parts.append(f"stage-entry keys {sorted(unknown_stage)}")
    logger.warning(
        "EnvironmentManifest.from_dict: unknown %s — accepted for forward "
        "compatibility but the library will not consume them. Check for "
        "typos / misplaced declarations (each setting has exactly one "
        "manifest home).",
        " and ".join(parts),
    )


def _migrate_legacy_mock_provider(stages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rewrite pre-0.13.5 ``s06_api.strategies.provider == 'mock'`` entries.

    Absorbed from Geny's load layer (``service/environment/service.py::
    _migrate_legacy_mock_provider``, 2.2.0): older blank manifests
    recorded ``provider: mock`` on the s06 entry because session-less
    introspection instantiated APIStage with MockProvider. At runtime
    that meant ``PipelineMutator.restore()`` swapped the real provider
    for MockProvider and every "agent reply" was the literal string
    ``"Mock response"`` — a prod incident debugged in Geny before the
    introspection fix (see ``_STAGE_INTROSPECTION_KWARGS`` in
    ``core/introspection.py`` for the forward fix). The library now
    performs the same on-load rewrite so *every* host gets the healing,
    not just the one that already paid for the incident.

    Returns a new list when a rewrite happened; the input payload is
    never mutated (``from_dict`` must not edit the caller's dict).
    """

    def _order(entry: Dict[str, Any]) -> int:
        # from_dict tolerates malformed payloads everywhere else; a
        # garbage ``order`` must not turn this healing pass into a crash.
        try:
            return int(entry.get("order", 0) or 0)
        except (TypeError, ValueError):
            return 0

    rewritten: List[Dict[str, Any]] = []
    changed = False
    for entry in stages:
        if (
            isinstance(entry, dict)
            and _order(entry) == 6
            and str(entry.get("artifact", "default")) == "default"
            and isinstance(entry.get("strategies"), dict)
            and entry["strategies"].get("provider") == "mock"
        ):
            entry = copy.deepcopy(entry)
            entry["strategies"]["provider"] = "anthropic"
            changed = True
        rewritten.append(entry)
    if changed:
        logger.warning(
            "EnvironmentManifest.from_dict: migrated legacy s06 "
            "strategies['provider']='mock' → 'anthropic' (pre-0.13.5 blank-"
            "manifest artifact; re-save this environment to persist the fix). "
            "Note strategies['provider'] itself is a legacy location — the "
            "single home is stages[6].config['provider']."
        )
    return rewritten if changed else stages


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

    **2.2.0 (Wave 3, audit 2026-06-09 §1-1)** adds two optional sections —
    both default-empty, so every existing manifest loads and round-trips
    unchanged (no version bump; pure additive defaults):

    - ``subagents``: list of sub-agent type declarations. Each entry is a
      plain dict::

          {"agent_type": str,                  # registry key (required)
           "description": str,                  # LLM-visible summary
           "provider": Optional[str],           # None ⇒ inherit parent
           "model_override": Optional[str],
           "allowed_tools": List[str],
           "env_id": Optional[str],             # stored env (host-resolved)
           "manifest": Optional[dict]}          # OR inline sub-manifest

      ``Pipeline.from_manifest`` compiles these into
      :class:`~geny_executor.stages.s12_agent.subagent_type.
      SubagentTypeDescriptor` registrations — sub-agent environments were
      previously host-code-only ("not first-class", audit §1-1).
    - ``memory``: declarative memory-provider block mirroring
      :class:`~geny_executor.memory.factory.MemoryProviderFactory`'s
      config-dict schema::

          {"provider": "file" | "sql" | "ephemeral" | "composite",
           "config": {...}}    # the factory's per-provider keys

      ``Pipeline.from_manifest`` builds and wires the provider when the
      block is non-empty; runtime objects a host attaches later via
      ``attach_runtime(memory_*=...)`` win over this declaration.
    """

    version: str = MANIFEST_VERSION
    metadata: EnvironmentMetadata = field(default_factory=EnvironmentMetadata)
    model: Dict[str, Any] = field(default_factory=dict)
    pipeline: Dict[str, Any] = field(default_factory=dict)
    stages: List[Dict[str, Any]] = field(default_factory=list)
    tools: ToolsSnapshot = field(default_factory=ToolsSnapshot)
    host_selections: HostSelections = field(default_factory=HostSelections)
    subagents: List[Dict[str, Any]] = field(default_factory=list)
    memory: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "metadata": self.metadata.to_dict(),
            "model": dict(self.model),
            "pipeline": dict(self.pipeline),
            "stages": list(self.stages),
            "tools": self.tools.to_dict(),
            "host_selections": self.host_selections.to_dict(),
            "subagents": list(self.subagents),
            "memory": dict(self.memory),
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

        Hygiene (2.2.0): unknown top-level / per-stage-entry keys are
        still *accepted* (forward compat — newer hosts may write richer
        payloads) but logged once per load so typos and misplaced
        declarations stop vanishing silently. Legacy
        ``s06.strategies['provider'] == 'mock'`` entries are migrated
        to ``'anthropic'`` on load — see
        :func:`_migrate_legacy_mock_provider` for the incident history.
        """
        version = str(data.get("version", "1.0"))
        if version == "1.0":
            data = _migrate_v1_to_v2(data)
            version = "2.0"
        if version == "2.0":
            data = _migrate_v2_to_v3(data)
            version = MANIFEST_VERSION
        _warn_unknown_keys(data)
        stages = _migrate_legacy_mock_provider(data.get("stages", []) or [])
        return cls(
            version=version,
            metadata=EnvironmentMetadata.from_dict(data.get("metadata", {})),
            model=data.get("model", {}),
            pipeline=data.get("pipeline", {}),
            stages=stages,
            tools=ToolsSnapshot.from_dict(data.get("tools", {})),
            host_selections=HostSelections.from_dict(data.get("host_selections")),
            # Absent → empty (2.2.0 Wave 3 additive sections; pre-Wave-3
            # payloads simply don't carry them).
            subagents=list(data.get("subagents", []) or []),
            memory=dict(data.get("memory", {}) or {}),
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

    def drift_against(self, introspection_catalog: Optional[List[Any]] = None) -> EnvironmentDiff:
        """Diff this manifest's stage layout against the library's canonical layout.

        Why (2.2.0, audit §3.1 / stage_model): hosts accumulate stored
        manifests across library versions — a 16-slot v2-era environment
        sitting next to today's 21-slot canon. Before order-keyed stage
        diffing, comparing the two collapsed into one opaque "stages
        changed" blob; now each stage's drift is reported individually
        (``stages[order=N].…`` paths), with stages missing from this
        manifest showing as ``removed`` and foreign ones as ``added``.

        The canonical baseline is built the same way
        :meth:`blank_manifest` builds its template: session-less
        introspection over every registered stage, default artifact,
        the artifact's default strategy picks, and the stage's default
        config. ``introspection_catalog`` accepts a pre-computed
        ``introspect_all()`` result (list of
        :class:`~geny_executor.core.introspection.StageIntrospection`)
        so batch callers (env list views) pay the introspection cost
        once.

        Returns:
            An :class:`EnvironmentDiff` where side *a* is the canonical
            layout and side *b* is this manifest — so ``added`` means
            "this manifest declares a stage/field the canon doesn't"
            and ``removed`` means "the canon has it, this manifest
            lost it".
        """
        from geny_executor.core.introspection import introspect_all

        catalog = introspection_catalog if introspection_catalog is not None else introspect_all()
        canonical: List[Dict[str, Any]] = []
        for insp in catalog:
            canonical.append(
                StageManifestEntry(
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
                ).to_dict()
            )
        return EnvironmentDiff.compute({"stages": canonical}, {"stages": list(self.stages)})

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
#  Manifest validation — write-time contract checking
# ═══════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ManifestIssue:
    """One finding from :func:`validate_manifest`.

    Severity semantics (owner's write-time-only rule — validation runs
    where manifests are *written or loaded*, never on the per-run hot
    path):

    - ``"error"`` — the declaration cannot take effect / the pipeline
      cannot honour the manifest's promise. ``Pipeline.from_manifest
      (strict=True)`` refuses to build on these.
    - ``"warning"`` — the manifest is suspicious (typo'd key, dual-home
      declaration, unappliable ordering) but a pipeline can still be
      built faithfully. Logged, never fatal.

    Fields:
      - ``severity``: ``"error"`` | ``"warning"``.
      - ``code``: stable machine-readable identifier (``"stage.…"`` /
        ``"strategy.…"`` / ``"chain.…"`` / ``"config.…"`` /
        ``"provider.…"`` / ``"model.…"`` / ``"version.…"`` /
        ``"subagent.…"`` / ``"memory.…"``). Hosts may
        key i18n / suppression lists on it; codes are append-only
        within a major version.
      - ``stage_order`` / ``stage_name``: the offending stage entry,
        when the issue is stage-scoped (``None`` for manifest-level
        issues like ``version.unknown``).
      - ``field``: dotted locator inside the entry (e.g.
        ``"strategies.controller"``), when one field is to blame.
      - ``message``: human-readable explanation with the fix spelled
        out.
    """

    severity: str
    code: str
    message: str
    stage_order: Optional[int] = None
    stage_name: Optional[str] = None
    field: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready representation (env-editor diagnostics)."""
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "stage_order": self.stage_order,
            "stage_name": self.stage_name,
            "field": self.field,
        }


# Stage config keys consumed by the engine itself rather than the
# stage's own ConfigSchema. ``provider_override`` is read uniformly by
# ``Stage.resolve_local_client`` on every stage, so it is legitimate in
# any stage's config dict even though no per-stage schema declares it.
_ENGINE_CONFIG_KEYS: Set[str] = {"provider_override"}


def validate_manifest(
    manifest: EnvironmentManifest,
    *,
    registry_introspection: Optional[Callable[[str, str], Any]] = None,
) -> List[ManifestIssue]:
    """Validate a manifest against the library's stage/strategy catalogue.

    Why (2.2.0, audit §1-1 / §2.1): the manifest claims to be the
    single source of truth, but a whole class of declarations used to
    be "accepted, stored, schema-green, and inert" — strategy configs
    aimed at strategies that drop them, impl names no registry knows,
    required stages quietly missing. Geny prod ran with its worker
    evaluator config dropped on the floor exactly this way. This
    function makes those failures *visible at write time*: env editors
    call it on save, ``Pipeline.from_manifest`` calls it at build
    (strict → errors raise; lenient → everything logs).

    All checks are pure and offline: the stage catalogue is consulted
    via the artifact loader and session-less introspection helpers —
    no network, no filesystem writes, no live pipeline.

    Checks (severity in brackets; ``active=False`` entries downgrade
    entry-scoped errors to warnings because an inactive declaration is
    a parked intent, not a live promise):

    - unknown stage name [error when active] /
      duplicate ``order`` values [error when any duplicate is active] /
      unknown artifact for a stage [error when active] /
      stage that fails to construct from the catalogue [error when active]
    - declared ``order`` differing from the stage's canonical order
      [warning] — ``from_manifest`` registers stages at their
      class-level order, so the manifest's number is cosmetic but
      misleading.
    - unknown strategy slot / unknown impl name vs the stage's slot
      registries [error when active]
    - ``strategy_configs`` targeting a strategy whose ``configure`` is
      the base no-op (``type(strategy).configure is
      Strategy.configure``) [error] — the audit §2.1 masked-degradation
      class: the config parses, stores, round-trips, and does nothing.
    - ``strategy_configs`` whose values the selected impl's own
      ``configure()`` rejects with ``ValueError`` [error when active]
      — probed on a throwaway instance (2.2.0 review B4); restore
      records the same rejection and keeps the strategy's defaults, so
      the bad value must surface at write time.
    - ``strategy_configs`` for a slot with no matching ``strategies``
      selection [warning] — ``PipelineMutator.restore`` only applies a
      slot's config alongside its strategy selection, so the config
      would never land.
    - ``chain_order`` naming an unknown chain or unknown chain impls
      [error when active]; chain orderings that the default chain
      contents cannot satisfy [warning] — ``restore`` can only
      *reorder* existing items, not populate (hosts that populate
      chains at runtime, e.g. Geny's ``populate_guard_chain``, can
      ignore this one).
    - stage ``config`` keys not in the stage's ConfigSchema, where a
      schema exists [warning]
    - required stages (``introspection._STAGE_REQUIRED``) missing or
      inactive [error]
    - provider missing on an active s06 [error] / provider in the
      legacy ``strategies`` location [error] — mirrors the existing
      build-time checks so standalone validators see them too.
    - ``model`` declared in BOTH the top-level ``model`` block and the
      s06 stage config [warning] — the top-level block is the single
      home and wins (``_pipeline_config_from_manifest`` reunites it
      into ``PipelineConfig.model``; the stage-config copy is inert).
    - ``version`` unknown / newer than this library supports [warning]
    - ``subagents`` entries (2.2.0 Wave 3): non-dict entry / missing
      ``agent_type`` [error] — the entry cannot be registered;
      duplicate ``agent_type`` [error] — the registry raises on
      duplicates, so the build would fail; both ``env_id`` AND an
      inline ``manifest`` set [warning] — the inline manifest wins;
      ``provider`` not in :class:`~geny_executor.llm_client.registry.
      ClientRegistry` [warning] — hosts register custom providers
      late, so an unknown name here may resolve at build time;
      ``provider``/``model_override``/``allowed_tools`` alongside an
      inline ``manifest``/``env_id`` [warning] — the factory builds
      wholly from that source and ignores them; a non-Claude
      ``provider`` with no ``model_override`` and no sub-manifest
      [warning] — the compiled sub-environment would carry the default
      claude-* model id, which that backend will 404 on.
    - ``tools.adhoc`` / ``tools.scope`` carrying data [warning] — both
      fields are serialized but engine-unconsumed (2.2.0 review B6); a
      silent green check over them would institutionalize the decoy.
    - ``memory`` block (2.2.0 Wave 3): missing/unknown ``provider``
      name vs the built-in :class:`~geny_executor.memory.factory.
      MemoryProviderFactory` builders [error]; ``config`` keys the
      named builder does not accept [warning]; stray keys outside
      ``provider``/``config`` [warning] — per-provider settings
      belong nested under ``config``.

    Args:
        manifest: The manifest to validate. Not mutated.
        registry_introspection: Optional ``(stage_module, artifact) →
            Stage`` factory used to build catalogue instances. Defaults
            to ``create_stage`` with the session-less introspection
            kwargs (dummy credentials where a ctor demands them).
            Injection point for tests and for hosts with out-of-tree
            stage registries.

    Returns:
        Every finding, in stage order then manifest order. Empty list
        means the manifest is clean.
    """
    from geny_executor.core.artifact import (
        _MODULE_TO_ORDER,
        _resolve_stage_module,
        create_stage,
        list_artifacts,
    )
    from geny_executor.core.introspection import (
        _STAGE_REQUIRED,
        _introspection_kwargs,
    )
    from geny_executor.core.stage import Strategy
    from geny_executor.llm_client.registry import ClientRegistry

    issues: List[ManifestIssue] = []

    def add(
        severity: str,
        code: str,
        message: str,
        *,
        order: Optional[int] = None,
        name: Optional[str] = None,
        field_: Optional[str] = None,
    ) -> None:
        issues.append(
            ManifestIssue(
                severity=severity,
                code=code,
                message=message,
                stage_order=order,
                stage_name=name,
                field=field_,
            )
        )

    # ── Manifest-level checks ───────────────────────────────
    supported_versions = _LEGACY_VERSIONS | {MANIFEST_VERSION}
    if str(manifest.version) not in supported_versions:
        add(
            "warning",
            "version.unknown",
            f"manifest version {manifest.version!r} is unknown to this library "
            f"(supported: {sorted(supported_versions)}). It may have been "
            "written by a newer geny-executor; declarations this library "
            "does not understand will be ignored.",
            field_="version",
        )

    # ── Subagents section (2.2.0 Wave 3, audit §1-1) ────────
    seen_agent_types: Set[str] = set()
    for idx, raw_sub in enumerate(manifest.subagents or []):
        locator = f"subagents[{idx}]"
        if not isinstance(raw_sub, dict):
            add(
                "error",
                "subagent.malformed_entry",
                f"{locator} must be a dict, got {type(raw_sub).__name__}; "
                "the entry cannot be compiled into a descriptor.",
                field_=locator,
            )
            continue
        agent_type = str(raw_sub.get("agent_type") or "").strip()
        if not agent_type:
            add(
                "error",
                "subagent.missing_type",
                f"{locator} declares no 'agent_type' — it is the registry "
                "key and the value the LLM uses in delegate requests, so "
                "the entry cannot be registered without one.",
                field_=f"{locator}.agent_type",
            )
            continue
        if agent_type in seen_agent_types:
            add(
                "error",
                "subagent.duplicate_type",
                f"{locator} re-declares agent_type {agent_type!r}; "
                "SubagentTypeRegistry.register raises on duplicates, so "
                "the build would fail. Keep one entry per agent_type.",
                field_=f"{locator}.agent_type",
            )
        seen_agent_types.add(agent_type)
        if raw_sub.get("env_id") and raw_sub.get("manifest"):
            add(
                "warning",
                "subagent.dual_source",
                f"{locator} ({agent_type!r}) sets BOTH 'env_id' and an "
                "inline 'manifest'; the inline manifest wins and env_id is "
                "ignored — delete one so the intent is unambiguous.",
                field_=f"{locator}.env_id",
            )
        sub_provider = raw_sub.get("provider")
        if sub_provider and str(sub_provider) not in ClientRegistry.available():
            add(
                "warning",
                "subagent.unknown_provider",
                f"{locator} ({agent_type!r}) requests provider "
                f"{sub_provider!r}, which is not currently registered "
                f"(known: {sorted(ClientRegistry.available())}). Hosts "
                "register custom providers late, so this may resolve at "
                "build time — verify the name is not a typo.",
                field_=f"{locator}.provider",
            )
        has_sub_source = bool(raw_sub.get("manifest")) or bool(raw_sub.get("env_id"))
        ignored_overrides = sorted(
            key for key in ("provider", "model_override", "allowed_tools") if raw_sub.get(key)
        )
        if has_sub_source and ignored_overrides:
            # ManifestSubagentPipelineFactory builds the sub-pipeline
            # WHOLLY from the inline manifest / resolved environment on
            # those paths — of the entry's fields only agent_type
            # (registry key) and description (delegation metadata) are
            # honoured there; provider/model_override/allowed_tools
            # never reach the build.
            add(
                "warning",
                "subagent.overrides_ignored",
                f"{locator} ({agent_type!r}) sets {ignored_overrides} "
                "alongside an inline 'manifest'/'env_id'; the factory "
                "builds the sub-pipeline entirely from that source, so "
                "these fields are ignored (only agent_type and "
                "description are honoured on this path). Declare them "
                "inside the sub-manifest instead.",
                field_=locator,
            )
        if (
            sub_provider
            and str(sub_provider) not in ("anthropic", "claude_code_cli")
            and not raw_sub.get("model_override")
            and not has_sub_source
        ):
            # The no-manifest path materializes build_manifest(...,
            # model=descriptor.model_override) — with no override that
            # is the default ModelConfig model, a claude-* id an
            # OpenAI-compatible backend will 404 on.
            add(
                "warning",
                "subagent.model_default_mismatch",
                f"{locator} ({agent_type!r}) requests provider "
                f"{sub_provider!r} with no 'model_override' and no inline "
                "manifest — the compiled sub-environment falls back to "
                "the default ModelConfig model (a claude-* id), which "
                f"{sub_provider!r} will reject. Declare a model_override "
                "the provider actually serves.",
                field_=f"{locator}.model_override",
            )

    # ── Tools section — engine-unconsumed fields (2.2.0 review B6) ──
    # ``built_in`` / ``mcp_servers`` / ``external`` are all consumed at
    # build time; ``adhoc`` and ``scope`` are serialized, round-tripped,
    # and read by NOTHING in the engine. A green validate_manifest over
    # data in them would institutionalize the decoy — warn instead.
    tools_snapshot = manifest.tools
    if getattr(tools_snapshot, "adhoc", None):
        add(
            "warning",
            "tools.adhoc_unconsumed",
            f"manifest.tools.adhoc carries {len(tools_snapshot.adhoc)} "
            "definition(s), but the engine does not consume tools.adhoc — "
            "they are stored and round-tripped only. Register adhoc tools "
            "at runtime (ToolRegistry / adhoc providers via "
            "tools.external) instead.",
            field_="tools.adhoc",
        )
    if getattr(tools_snapshot, "scope", None):
        add(
            "warning",
            "tools.scope_unconsumed",
            "manifest.tools.scope carries data, but the engine does not "
            "consume tools.scope — it is stored and round-tripped only. "
            "Use stage tool_binding / permission rules for tool scoping.",
            field_="tools.scope",
        )

    # ── Memory block (2.2.0 Wave 3, audit §1-1) ─────────────
    memory_block = manifest.memory or {}
    if memory_block:
        from geny_executor.memory.factory import (
            MEMORY_PROVIDER_CONFIG_KEYS,
            MemoryProviderFactory,
        )

        registered_memory = MemoryProviderFactory().names()
        memory_provider = memory_block.get("provider")
        if not isinstance(memory_provider, str) or not memory_provider:
            add(
                "error",
                "memory.missing_provider",
                "manifest.memory is non-empty but declares no 'provider'; "
                f"set one of the registered names: {registered_memory}.",
                field_="memory.provider",
            )
        elif memory_provider not in registered_memory:
            add(
                "error",
                "memory.unknown_provider",
                f"manifest.memory requests unknown provider "
                f"{memory_provider!r}; registered: {registered_memory}. "
                "MemoryProviderFactory.build would refuse, so the "
                "declaration cannot take effect.",
                field_="memory.provider",
            )
        else:
            accepted_keys = MEMORY_PROVIDER_CONFIG_KEYS.get(memory_provider)
            memory_config = memory_block.get("config") or {}
            if accepted_keys is not None and isinstance(memory_config, dict):
                unknown_cfg = sorted(set(memory_config.keys()) - set(accepted_keys))
                if unknown_cfg:
                    add(
                        "warning",
                        "memory.unknown_config_key",
                        f"manifest.memory.config declares keys {unknown_cfg} "
                        f"that the {memory_provider!r} builder does not "
                        f"accept (known: {sorted(k for k in accepted_keys if k != 'provider')}); "
                        "they would be stored but not consumed.",
                        field_="memory.config",
                    )
        stray_keys = sorted(set(memory_block.keys()) - {"provider", "config"})
        if stray_keys:
            add(
                "warning",
                "memory.unknown_key",
                f"manifest.memory declares keys {stray_keys} outside "
                "'provider'/'config'; per-provider settings belong nested "
                "under memory.config — top-level strays are ignored.",
                field_="memory",
            )

    entries = manifest.stage_entries()

    seen_orders: Dict[int, List[StageManifestEntry]] = {}
    for entry in entries:
        seen_orders.setdefault(entry.order, []).append(entry)
    for order, group in sorted(seen_orders.items()):
        if len(group) > 1:
            severity = "error" if any(e.active for e in group) else "warning"
            add(
                severity,
                "stage.duplicate_order",
                f"order {order} is declared by {len(group)} stage entries "
                f"({[e.name for e in group]}); the pipeline keys stages by "
                "order, so later entries silently displace earlier ones.",
                order=order,
                name=group[0].name,
            )

    top_level_model = bool((manifest.model or {}).get("model"))

    # Catalogue instances are cached per (module, artifact) within one
    # validation pass — duplicate orders / repeated artifacts shouldn't
    # pay construction twice.
    catalogue: Dict[tuple, Any] = {}
    artifact_lists: Dict[str, List[str]] = {}

    def stage_instance(module: str, artifact: str) -> Any:
        key = (module, artifact)
        if key not in catalogue:
            if registry_introspection is not None:
                catalogue[key] = registry_introspection(module, artifact)
            else:
                kwargs = _introspection_kwargs(module, artifact)
                catalogue[key] = create_stage(module, artifact, **kwargs)
        return catalogue[key]

    active_required: Set[str] = set()

    # ── Per-entry checks ────────────────────────────────────
    for entry in entries:
        # Inactive entries are parked intent: their problems are worth
        # surfacing but must never block a build that won't run them.
        entry_severity = "error" if entry.active else "warning"

        try:
            module = _resolve_stage_module(entry.name)
        except (ValueError, AttributeError):
            add(
                entry_severity,
                "stage.unknown_name",
                f"stage entry order={entry.order} names unknown stage "
                f"{entry.name!r}; use a module name (s06_api), short name "
                "(api), or order string.",
                order=entry.order,
                name=entry.name,
                field_="name",
            )
            continue

        if entry.active and module in _STAGE_REQUIRED:
            active_required.add(module)

        canonical_order = _MODULE_TO_ORDER.get(module)
        if canonical_order is not None and entry.order != canonical_order:
            add(
                "warning",
                "stage.order_mismatch",
                f"stage {entry.name!r} declares order={entry.order} but its "
                f"canonical order is {canonical_order}; from_manifest "
                "registers stages at their class-level order, so the "
                "declared number is ignored.",
                order=entry.order,
                name=entry.name,
                field_="order",
            )

        if module not in artifact_lists:
            artifact_lists[module] = list_artifacts(module)
        if entry.artifact not in artifact_lists[module]:
            add(
                entry_severity,
                "stage.unknown_artifact",
                f"stage {entry.name!r} (order {entry.order}) requests unknown "
                f"artifact {entry.artifact!r}; available: "
                f"{artifact_lists[module]}.",
                order=entry.order,
                name=entry.name,
                field_="artifact",
            )
            continue

        strategies = entry.strategies or {}
        if "provider" in strategies:
            # Mirrors _validate_manifest_provider_locations — error even on
            # inactive entries, because reactivating one must not resurrect
            # the silent-divergence bug class the single home killed.
            add(
                "error",
                "provider.legacy_location",
                f"stage {entry.name!r} (order {entry.order}) stores provider "
                f"in the legacy strategies['provider']="
                f"{strategies['provider']!r} slot; move it to "
                "config['provider'] — the single source of truth.",
                order=entry.order,
                name=entry.name,
                field_="strategies.provider",
            )

        if module == "s06_api":
            cfg = entry.config or {}
            if entry.active and not cfg.get("provider"):
                add(
                    "error",
                    "provider.missing",
                    "Stage 6 ('api') is active but no provider is configured. "
                    "Set stages[6].config['provider'] (e.g. 'anthropic').",
                    order=entry.order,
                    name=entry.name,
                    field_="config.provider",
                )
            if top_level_model and cfg.get("model"):
                add(
                    "warning",
                    "model.dual_home",
                    "model is declared in BOTH the top-level manifest 'model' "
                    "block and stages[6].config['model']. The top-level block "
                    "is the single home and wins (it becomes "
                    "PipelineConfig.model); the stage-config copy is inert — "
                    "delete it.",
                    order=entry.order,
                    name=entry.name,
                    field_="config.model",
                )

        try:
            instance = stage_instance(module, entry.artifact)
        except Exception as exc:  # construction failed — catalogue unusable
            add(
                entry_severity,
                "stage.unbuildable",
                f"stage {entry.name!r} (order {entry.order}, artifact "
                f"{entry.artifact!r}) failed to construct for validation: "
                f"{type(exc).__name__}: {exc}",
                order=entry.order,
                name=entry.name,
            )
            continue

        slots = instance.get_strategy_slots() or {}
        chains = instance.get_strategy_chains() or {}

        for slot_name, impl_name in strategies.items():
            if slot_name == "provider":
                continue  # already flagged as provider.legacy_location
            slot = slots.get(slot_name)
            if slot is None:
                add(
                    entry_severity,
                    "strategy.unknown_slot",
                    f"stage {entry.name!r} (order {entry.order}) selects "
                    f"strategy for unknown slot {slot_name!r}; available "
                    f"slots: {sorted(slots.keys())}, chains: "
                    f"{sorted(chains.keys())}.",
                    order=entry.order,
                    name=entry.name,
                    field_=f"strategies.{slot_name}",
                )
                continue
            if impl_name not in slot.registry:
                add(
                    entry_severity,
                    "strategy.unknown_impl",
                    f"stage {entry.name!r} (order {entry.order}) slot "
                    f"{slot_name!r} selects unknown impl {impl_name!r}; "
                    f"available: {sorted(slot.registry.keys())}. "
                    "PipelineMutator.restore would skip this selection.",
                    order=entry.order,
                    name=entry.name,
                    field_=f"strategies.{slot_name}",
                )

        for slot_name, slot_config in (entry.strategy_configs or {}).items():
            if not slot_config:
                continue  # empty dict — nothing to lose
            field_path = f"strategy_configs.{slot_name}"
            slot = slots.get(slot_name)
            if slot is None:
                if slot_name in chains:
                    add(
                        "warning",
                        "strategy.config_unpaired",
                        f"stage {entry.name!r} (order {entry.order}) declares "
                        f"strategy_configs[{slot_name!r}] but {slot_name!r} "
                        "is a chain — restore only applies strategy_configs "
                        "to slots; configure chain items host-side or via "
                        "chain mutation APIs.",
                        order=entry.order,
                        name=entry.name,
                        field_=field_path,
                    )
                else:
                    add(
                        entry_severity,
                        "strategy.unknown_slot",
                        f"stage {entry.name!r} (order {entry.order}) declares "
                        f"strategy_configs for unknown slot {slot_name!r}; "
                        f"available slots: {sorted(slots.keys())}.",
                        order=entry.order,
                        name=entry.name,
                        field_=field_path,
                    )
                continue
            declared_impl = strategies.get(slot_name)
            if declared_impl is None:
                add(
                    "warning",
                    "strategy.config_unpaired",
                    f"stage {entry.name!r} (order {entry.order}) declares "
                    f"strategy_configs[{slot_name!r}] without selecting a "
                    f"strategy in strategies[{slot_name!r}] — "
                    "PipelineMutator.restore only applies a slot's config "
                    "alongside its strategy selection, so this config will "
                    "never land. Declare the impl name too.",
                    order=entry.order,
                    name=entry.name,
                    field_=field_path,
                )
                continue
            impl_cls = slot.registry.get(declared_impl)
            if impl_cls is None:
                continue  # unknown impl already flagged above
            if getattr(impl_cls, "configure", None) is Strategy.configure:
                add(
                    "error",
                    "strategy.config_dropped",
                    f"stage {entry.name!r} (order {entry.order}) declares "
                    f"strategy_configs[{slot_name!r}] for impl "
                    f"{declared_impl!r}, but that strategy does not override "
                    "Strategy.configure — the config would be accepted, "
                    "stored, and silently dropped (the audit §2.1 Geny prod "
                    "bug class). Implement configure() on the strategy or "
                    "remove the config block.",
                    order=entry.order,
                    name=entry.name,
                    field_=field_path,
                )
                continue
            # Probe the config against the strategy's own configure()
            # (2.2.0 review B4): wave 1 made configure fail loudly on
            # bad values, so the same bad value must be caught HERE, at
            # write time — not discovered as a hard strict-build failure
            # or a silently-defaulted lenient build. Offline by
            # construction: a throwaway instance takes the config; any
            # non-ValueError (constructor needs runtime deps, configure
            # needs live context) means this pass cannot judge and stays
            # quiet.
            try:
                probe = impl_cls()
            except Exception:
                continue  # not constructible offline — cannot probe
            try:
                probe.configure(dict(slot_config))
            except ValueError as exc:
                add(
                    entry_severity,
                    "strategy.config_invalid",
                    f"stage {entry.name!r} (order {entry.order}) declares "
                    f"strategy_configs[{slot_name!r}] that impl "
                    f"{declared_impl!r} rejects: {exc}. "
                    "PipelineMutator.restore records this as an error and "
                    "the strategy keeps its defaults — fix the value.",
                    order=entry.order,
                    name=entry.name,
                    field_=field_path,
                )
            except Exception:  # noqa: BLE001 — probe must never crash validation
                pass

        for chain_name, chain_order in (entry.chain_order or {}).items():
            if not chain_order:
                continue  # restore skips empty orderings
            field_path = f"chain_order.{chain_name}"
            chain = chains.get(chain_name)
            if chain is None:
                add(
                    entry_severity,
                    "chain.unknown",
                    f"stage {entry.name!r} (order {entry.order}) declares "
                    f"chain_order for unknown chain {chain_name!r}; "
                    f"available chains: {sorted(chains.keys())}.",
                    order=entry.order,
                    name=entry.name,
                    field_=field_path,
                )
                continue
            unknown_impls = [n for n in chain_order if n not in chain.registry]
            if unknown_impls:
                add(
                    entry_severity,
                    "chain.unknown_impl",
                    f"stage {entry.name!r} (order {entry.order}) chain "
                    f"{chain_name!r} ordering names unknown impls "
                    f"{unknown_impls}; available: "
                    f"{sorted(chain.registry.keys())}.",
                    order=entry.order,
                    name=entry.name,
                    field_=field_path,
                )
                continue
            current = sorted(item.name for item in chain.items)
            if sorted(chain_order) != current:
                add(
                    "warning",
                    "chain.order_unappliable",
                    f"stage {entry.name!r} (order {entry.order}) chain "
                    f"{chain_name!r} ordering {list(chain_order)} is not a "
                    f"permutation of the default chain contents {current} — "
                    "PipelineMutator.restore can only reorder existing items, "
                    "so this ordering will be skipped. Populate the chain at "
                    "runtime (e.g. Stage.add_to_chain) if that is the intent.",
                    order=entry.order,
                    name=entry.name,
                    field_=field_path,
                )

        schema = instance.get_config_schema() if hasattr(instance, "get_config_schema") else None
        if schema is not None and getattr(schema, "fields", None) is not None:
            known_keys = {f.name for f in schema.fields} | _ENGINE_CONFIG_KEYS
            unknown_keys = sorted(set((entry.config or {}).keys()) - known_keys)
            if unknown_keys:
                add(
                    "warning",
                    "config.unknown_key",
                    f"stage {entry.name!r} (order {entry.order}) config "
                    f"declares keys {unknown_keys} that its ConfigSchema does "
                    f"not define (known: {sorted(known_keys)}); they will be "
                    "stored but not consumed.",
                    order=entry.order,
                    name=entry.name,
                    field_="config",
                )

    # ── Required stages ─────────────────────────────────────
    for module in sorted(_STAGE_REQUIRED):
        if module not in active_required:
            add(
                "error",
                "stage.required_inactive",
                f"required stage {module!r} is missing or inactive — every "
                "pipeline is an LLM agent loop and cannot function without "
                "input/api/parse/yield (see introspection._STAGE_REQUIRED). "
                "Add the entry with active=true.",
                order=_MODULE_TO_ORDER.get(module),
                name=module,
            )

    return issues


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
