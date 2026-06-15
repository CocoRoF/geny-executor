"""Preset → :class:`EnvironmentManifest` factory — library-owned.

Why (2.2.0, audit §1-3 / Tier 1-2): "hosts are thin consumers" was
violated most blatantly by manifest construction. Geny carried a
728-line ``default_manifest.py`` (plus ``stage_manifest.py``) whose
entire job was to hand-mirror the canonical 21-stage layouts this
library already defines — every stage rename, every new slot, every
default-strategy change had to be re-mirrored by hand, and §2.1 showed
what happens when the mirror drifts (the prod worker loop ran with its
evaluator config dropped). :func:`build_manifest` is that factory,
owned by the library: hosts ask for a preset by name and get a
validated, strict-buildable :class:`EnvironmentManifest` back.

The canonical layouts encoded here are the ones Geny ships today
(``Geny/backend/service/executor/default_manifest.py`` — read as the
reference consumer this module absorbs, reproduced faithfully including
the s14 ``evaluation_chain`` and s16 ``multi_dim_budget``
``strategy_configs`` that Wave 1 made real). The vendored layout
snapshot in ``tests/_fixtures/geny_manifest_layout.json`` pins
byte-level compatibility with that file.

Layout (geny-executor 1.0+, Phase 9a/9b):

    1  input          | 12  agent           | 17  emit
    2  context        | 13  task_registry   | 18  memory
    3  system         | 14  evaluate        | 19  summarize
    4  guard          | 15  hitl            | 20  persist
    5  cache          | 16  loop            | 21  yield
    6  api
    7  token
    8  think
    9  parse
    10 tool
    11 tool_review

The returned manifest carries **only declarative shape** — stage list,
artifact names, slot strategy choices, static configs. Runtime-scoped
objects (memory retrievers/strategies, composable prompt builder
blocks, HITL requesters, file persisters) stay out of the manifest and
are wired by :meth:`Pipeline.attach_runtime` at session start. Slots
that a host is expected to swap at runtime carry a safe default here
and are marked ``# swapped by attach_runtime`` below.

Known limitation (inherited, documented): ``chain_order`` declarations
can only be *applied* by ``PipelineMutator.restore`` when the default
chain already contains the named items. The worker preset's s04 guard
ordering (the default guard chain ships empty) and the vtuber preset's
narrowed s11 reviewer chain therefore require host-side population
(see Geny's ``populate_guard_chain``); ``validate_manifest`` reports
these as ``chain.order_unappliable`` warnings, not errors, for exactly
this reason. The declarations are kept because they are the documented
intent the host helpers read.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from geny_executor.core.environment import (
    EnvironmentManifest,
    EnvironmentMetadata,
    StageManifestEntry,
    ToolsSnapshot,
)


# ── Preset names ─────────────────────────────────────────────────────

_VTUBER = "vtuber"
_WORKER_ADAPTIVE = "worker_adaptive"
_DEFAULT_ALIAS = "default"  # maps to worker_adaptive (AgentSession convention)

MANIFEST_PRESETS = frozenset({_VTUBER, _WORKER_ADAPTIVE, _DEFAULT_ALIAS})


# Defaults the adaptive evaluator consumes. Mirror
# ``geny_executor.memory.presets.GenyPresets.worker_adaptive`` directly.
_WORKER_ADAPTIVE_EASY_MAX_TURNS = 1
_WORKER_ADAPTIVE_NOT_EASY_MAX_TURNS = 30

# Loop max_turns defaults per preset. Mirror GenyPresets.* directly.
_WORKER_ADAPTIVE_MAX_TURNS = 30
_VTUBER_MAX_TURNS = 10


# ── Sub-phase 9a scaffold entries ────────────────────────────────────
#
# The five orders added by the 16→21 layout growth. Each defaults to
# active=False with the executor's safe no-op strategies so the entry
# is runnable the moment a preset (or a user edit) flips it on.

_SCAFFOLD_ENTRIES_SPEC: List[Dict[str, Any]] = [
    {
        "order": 11,
        "name": "tool_review",
        "strategies": {},  # chain stage — strategies live on chain_order
        "chain_order": {
            "reviewers": [
                "schema",
                "sensitive",
                "destructive",
                "network",
                "size",
            ],
        },
    },
    {
        "order": 13,
        "name": "task_registry",
        "strategies": {
            "registry": "in_memory",
            "policy": "fire_and_forget",
        },
    },
    {
        "order": 15,
        "name": "hitl",
        "strategies": {
            "requester": "null",  # safe default — always-approve
            "timeout": "indefinite",
        },
    },
    {
        "order": 19,
        "name": "summarize",
        "strategies": {
            "summarizer": "no_summary",  # default off
            "importance": "fixed",
        },
    },
    {
        "order": 20,
        "name": "persist",
        "strategies": {
            "persister": "no_persist",  # default off
            "frequency": "every_turn",
        },
    },
]


# Per-preset opt-in for scaffold stages — a partial override merged
# onto the matching spec entry. The rationale comments are inherited
# from the Geny integration sprints (G2.x) that proved each choice in
# prod; they live here now because the layout does.
_PRESET_SCAFFOLD_OVERRIDES: Dict[str, Dict[str, Dict[str, Any]]] = {
    _WORKER_ADAPTIVE: {
        # Tool Review chain on. Default reviewer order (schema →
        # sensitive → destructive → network → size) comes from the
        # scaffold spec; flags land at state.shared['tool_review_flags'].
        "tool_review": {
            "active": True,
        },
        # HITL gate on with the safe ``null`` requester placeholder.
        # The real resume-capable requester needs a Pipeline ref the
        # manifest cannot serialise — hosts swap it at runtime.
        # ``should_bypass`` returns True when nothing wrote to
        # state.shared['hitl_request'] this turn, so the active flag is
        # a free no-op until something opts in.
        "hitl": {
            "active": True,
            "strategies": {
                "requester": "null",  # swapped at runtime
                "timeout": "indefinite",  # rely on UI to resolve
            },
        },
        # Turn-summary writer + heuristic importance grader. Forwards
        # to the session's memory provider when one is attached.
        "summarize": {
            "active": True,
            "strategies": {
                "summarizer": "rule_based",
                "importance": "heuristic",
            },
        },
        # Declarative persist on. The persister slot stays ``no_persist``
        # in the manifest; a real persister (e.g. FilePersister) is
        # session-scoped and wired at runtime. ``on_significant``
        # keeps IO bounded — checkpoints land only on noteworthy events.
        "persist": {
            "active": True,
            "strategies": {
                "persister": "no_persist",  # swapped at runtime
                "frequency": "on_significant",
            },
        },
        # Task Registry on: in-memory backend + fire_and_forget policy
        # so sub-worker delegations acquire a per-pipeline lifecycle
        # handle without blocking the agent loop.
        "task_registry": {
            "active": True,
            "strategies": {
                "registry": "in_memory",
                "policy": "fire_and_forget",
            },
        },
    },
    _VTUBER: {
        # Light tool-review chain: the conversational persona's tool
        # surface is small, so keep schema (arg validation) + sensitive
        # (PII / secret leak detection) and drop the rest as noise.
        # NOTE: a narrowed ordering is host-population territory — see
        # the module docstring's chain_order limitation.
        "tool_review": {
            "active": True,
            "chain_order": {"reviewers": ["schema", "sensitive"]},
        },
        # Turn-summary writer + heuristic importance — keeps
        # long-conversation context coherent without the full
        # binary_classify evaluator.
        "summarize": {
            "active": True,
            "strategies": {
                "summarizer": "rule_based",
                "importance": "heuristic",
            },
        },
        # on_significant checkpointing with the no_persist placeholder;
        # real persister swapped at runtime, preset-agnostic.
        "persist": {
            "active": True,
            "strategies": {
                "persister": "no_persist",  # swapped at runtime
                "frequency": "on_significant",
            },
        },
        # task_registry / hitl stay off — the VTuber is a single-agent
        # autonomous persona: no delegation registry to track, no
        # human-approval surface to gate.
    },
}


def _make_scaffold_entries(
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[StageManifestEntry]:
    """Build the 5 scaffold entries with optional per-name overrides.

    ``overrides`` maps a scaffold name to a dict carrying any of:
    ``active`` (bool), ``strategies`` (dict — merged onto the spec
    defaults), ``strategy_configs`` (dict), ``chain_order`` (dict —
    replaces the spec default). Names not present keep the canonical
    scaffold defaults (active=False, no-op strategies).
    """
    overrides = overrides or {}
    out: List[StageManifestEntry] = []
    for spec in _SCAFFOLD_ENTRIES_SPEC:
        name = spec["name"]
        ov = overrides.get(name) or {}
        strategies = dict(spec.get("strategies") or {})
        strategies.update(ov.get("strategies") or {})
        out.append(
            StageManifestEntry(
                order=spec["order"],
                name=name,
                active=bool(ov.get("active", False)),
                strategies=strategies,
                strategy_configs=dict(ov.get("strategy_configs") or {}),
                chain_order=dict(
                    ov.get("chain_order") if "chain_order" in ov else spec.get("chain_order") or {}
                ),
            )
        )
    return out


def _merge_sorted(*entry_lists: List[StageManifestEntry]) -> List[StageManifestEntry]:
    """Concatenate and sort by ``order`` so the output is canonically ordered."""
    merged: List[StageManifestEntry] = []
    for el in entry_lists:
        merged.extend(el)
    merged.sort(key=lambda e: int(e.order))
    return merged


def _worker_adaptive_stage_entries(*, provider: str) -> List[StageManifestEntry]:
    """The adaptive worker stage chain (mirrors ``GenyPresets.worker_adaptive``)."""
    return [
        StageManifestEntry(
            order=1,
            name="input",
            strategies={"validator": "default", "normalizer": "default"},
        ),
        StageManifestEntry(
            order=2,
            name="context",
            strategies={
                "strategy": "simple_load",
                "compactor": "truncate",
                "retriever": "null",  # swapped by attach_runtime
            },
        ),
        StageManifestEntry(
            order=3,
            name="system",
            # Composable builder — the block list (persona + datetime +
            # memory context) is host-composed and attaches at runtime.
            strategies={"builder": "composable"},
        ),
        StageManifestEntry(
            order=4,
            name="guard",
            # Declare the guard chain explicitly so PermissionGuard joins
            # the pre-flight check. Cheapest checks first — token / cost /
            # iteration are O(1) state lookups, permission is a regex
            # match against the rule list. NOTE: the default guard chain
            # ships empty, so this ordering needs host-side population
            # (module docstring, chain_order limitation).
            chain_order={
                "guards": ["token_budget", "cost_budget", "iteration", "permission"],
            },
        ),
        StageManifestEntry(
            order=5,
            name="cache",
            strategies={"strategy": "aggressive_cache"},
        ),
        StageManifestEntry(
            order=6,
            name="api",
            # Provider lives at config['provider'] (single source of
            # truth) — never in strategies (legacy location, rejected at
            # strict load).
            config={"provider": provider},
            strategies={
                "retry": "exponential_backoff",
                # Capability-aware adaptive router. Strict superset of
                # passthrough — without strategy_configs the router falls
                # back to the session's bound model.
                "router": "adaptive",
            },
        ),
        StageManifestEntry(
            order=7,
            name="token",
            strategies={
                "tracker": "default",
                "calculator": "anthropic_pricing",
            },
        ),
        StageManifestEntry(
            order=8,
            name="think",
            strategies={
                "processor": "extract_and_store",
                # Adaptive budget planner — first turn gets a high
                # thinking budget (planning), subsequent turns step down.
                "budget_planner": "adaptive",
            },
        ),
        StageManifestEntry(
            order=9,
            name="parse",
            strategies={"parser": "default", "signal_detector": "regex"},
        ),
        StageManifestEntry(
            order=10,
            name="tool",
            # Capability-aware partition: concurrency_safe tool calls run
            # as a parallel batch (capped at 8), the rest serialize.
            strategies={"executor": "partition", "router": "registry"},
            config={"max_concurrency": 8},
        ),
        StageManifestEntry(
            order=12,
            name="agent",
            # Multi-agent on by default; Pipeline.from_manifest rewires
            # the slot with SubagentTypeOrchestrator(registry) when a
            # subagent_registry= is passed along.
            strategies={"orchestrator": "subagent_type"},
            config={"max_delegations": 4},
        ),
        StageManifestEntry(
            order=14,
            name="evaluate",
            # evaluation_chain wraps binary_classify + signal_based; the
            # chain returns the first non-null verdict. This is the
            # strategy_configs block that the audit §2.1 prod bug dropped
            # on the floor — Wave 1 made EvaluationChain.configure real,
            # and validate_manifest now refuses configs aimed at no-op
            # configure implementations.
            strategies={"strategy": "evaluation_chain", "scorer": "no_scorer"},
            strategy_configs={
                "strategy": {
                    "evaluators": ["binary_classify", "signal_based"],
                    "easy_max_turns": _WORKER_ADAPTIVE_EASY_MAX_TURNS,
                    "not_easy_max_turns": _WORKER_ADAPTIVE_NOT_EASY_MAX_TURNS,
                },
            },
        ),
        StageManifestEntry(
            order=16,
            name="loop",
            # multi_dim_budget controller, single iteration dimension
            # (= standard behaviour). Adding cost_usd / walltime_seconds
            # dimensions later is a strategy_configs edit — and since
            # Wave 1 that edit actually lands (configure is real).
            strategies={"controller": "multi_dim_budget"},
            config={"max_turns": _WORKER_ADAPTIVE_MAX_TURNS},
            strategy_configs={
                "controller": {
                    "dimensions": ["iterations"],
                },
            },
        ),
        StageManifestEntry(
            order=17,
            name="emit",
            strategies={},
            chain_order={"emitters": []},
        ),
        StageManifestEntry(
            order=18,
            name="memory",
            # structured_reflective degrades to append-only without
            # strategy_configs; hosts with a memory manager swap in their
            # real strategy at attach_runtime.
            strategies={
                "strategy": "structured_reflective",
                "persistence": "null",  # swapped by attach_runtime
            },
        ),
        StageManifestEntry(
            order=21,
            name="yield",
            strategies={"formatter": "default"},
        ),
    ]


def _vtuber_stage_entries(*, provider: str) -> List[StageManifestEntry]:
    """The conversational persona stage chain (mirrors ``GenyPresets.vtuber``).

    Diff vs worker_adaptive: Stage 8 (think) ships ``active=False`` (the
    persona's turns are conversational, not deep-planning), cache is
    ``system_cache`` (not ``aggressive_cache``), evaluator is
    ``signal_based`` (not the evaluation chain), router is
    ``passthrough`` (the session's bound model is honoured verbatim),
    tool executor is ``sequential``, and loop ``max_turns`` is 10.

    Stage 8 is declared inactive rather than omitted so environment
    editors render the order-8 slot like every other inactive stage —
    omitting it entirely made the slot a "missing" error in the canvas
    (incident inherited from the Geny builder this module absorbs).
    """
    return [
        StageManifestEntry(
            order=1,
            name="input",
            strategies={"validator": "default", "normalizer": "default"},
        ),
        StageManifestEntry(
            order=2,
            name="context",
            strategies={
                "strategy": "simple_load",
                "compactor": "truncate",
                "retriever": "null",  # swapped by attach_runtime
            },
        ),
        StageManifestEntry(
            order=3,
            name="system",
            strategies={"builder": "composable"},
        ),
        StageManifestEntry(
            order=4,
            name="guard",
        ),
        StageManifestEntry(
            order=5,
            name="cache",
            strategies={"strategy": "system_cache"},
        ),
        StageManifestEntry(
            order=6,
            name="api",
            # Provider lives at config['provider'] (single source).
            config={"provider": provider},
            strategies={
                "retry": "exponential_backoff",
                "router": "passthrough",
            },
        ),
        StageManifestEntry(
            order=7,
            name="token",
            strategies={
                "tracker": "default",
                "calculator": "anthropic_pricing",
            },
        ),
        StageManifestEntry(
            order=8,
            name="think",
            active=False,
            strategies={
                "processor": "extract_and_store",
                "budget_planner": "adaptive",
            },
        ),
        StageManifestEntry(
            order=9,
            name="parse",
            strategies={"parser": "default", "signal_detector": "regex"},
        ),
        StageManifestEntry(
            order=10,
            name="tool",
            strategies={"executor": "sequential", "router": "registry"},
        ),
        StageManifestEntry(
            order=12,
            name="agent",
            strategies={"orchestrator": "subagent_type"},
            config={"max_delegations": 4},
        ),
        StageManifestEntry(
            order=14,
            name="evaluate",
            strategies={"strategy": "signal_based", "scorer": "no_scorer"},
        ),
        StageManifestEntry(
            order=16,
            name="loop",
            strategies={"controller": "standard"},
            config={"max_turns": _VTUBER_MAX_TURNS},
        ),
        StageManifestEntry(
            order=17,
            name="emit",
            strategies={},
            chain_order={"emitters": []},
        ),
        StageManifestEntry(
            order=18,
            name="memory",
            strategies={
                "strategy": "append_only",  # swapped by attach_runtime
                "persistence": "null",  # swapped by attach_runtime
            },
        ),
        StageManifestEntry(
            order=21,
            name="yield",
            strategies={"formatter": "default"},
        ),
    ]


def _build_stage_entries(preset: str, *, provider: str) -> List[StageManifestEntry]:
    """Emit the full 21-entry :class:`StageManifestEntry` list for *preset*."""
    if preset == _VTUBER:
        base = _vtuber_stage_entries(provider=provider)
    else:
        base = _worker_adaptive_stage_entries(provider=provider)
    overrides = _PRESET_SCAFFOLD_OVERRIDES.get(preset, {})
    return _merge_sorted(base, _make_scaffold_entries(overrides=overrides))


def known_manifest_presets() -> List[str]:
    """Supported preset names for :func:`build_manifest` — UI/validation hook."""
    return sorted(MANIFEST_PRESETS)


def build_manifest(
    preset: str,
    *,
    provider: str,
    model: Optional[str] = None,
    built_in_tools: Optional[List[str]] = None,
    external_tools: Optional[List[Any]] = None,
    mcp_servers: Optional[List[dict]] = None,
    name: str = "",
    description: str = "",
) -> EnvironmentManifest:
    """Materialize a ready-to-build :class:`EnvironmentManifest` for a preset.

    The library-owned replacement for per-host manifest builders (audit
    Tier 1-2): the returned manifest carries the canonical 21-stage
    layout for *preset*, builds under
    ``Pipeline.from_manifest(strict=True)``, and round-trips through
    ``to_dict``/``from_dict`` unchanged.

    Args:
        preset: ``"worker_adaptive"`` (autonomous tool-using worker),
            ``"vtuber"`` (conversational persona), or ``"default"``
            (alias of ``worker_adaptive`` — the AgentSession
            convention). Anything else raises ``ValueError`` — no
            silent fallback, a typo'd preset must fail loudly.
        provider: The Stage-6 LLM backend. Lands at
            ``stages[6].config["provider"]`` — the single source of
            truth ``Pipeline._resolve_llm_client`` reads. Must be a
            provider registered in
            :class:`~geny_executor.llm_client.registry.ClientRegistry`
            (``anthropic`` / ``openai`` / ``google`` / ``vllm`` /
            ``claude_code_cli``); unknown names raise ``ValueError``
            at *factory* time instead of failing at the first run.
        model: Optional LLM model id for the top-level ``model`` block
            (the single manifest home for model selection). When
            omitted the block stays empty and the host sets the model
            via ``PipelineConfig`` / per-run ``ModelOverrides``.
        built_in_tools: Names of framework-shipped built-in tools to
            register (``["*"]`` = every built-in, including future
            additions; ``None`` / ``[]`` = none — what a conversational
            persona wants).
        external_tools: ``manifest.tools.external`` entries — plain
            name strings, or ``{"name": ..., "required": True}``
            mappings for tools the environment is broken without
            (strict build fails when a required entry resolves to no
            provider).
        mcp_servers: ``manifest.tools.mcp_servers`` entries — dicts
            with at least ``name`` plus the transport fields
            ``MCPServerConfig`` understands (``command`` / ``args`` /
            ``url`` / ``transport`` / ``env`` / ``headers``). Entries
            without a ``name`` raise ``ValueError`` here; the runtime
            would skip them silently, which is exactly the masked
            degradation this factory exists to prevent.
        name: ``metadata.name``. Defaults to ``"preset:<effective>"``.
        description: ``metadata.description``. Defaults to a sentence
            naming the preset.

    Returns:
        A fresh :class:`EnvironmentManifest` (new ``env_…`` id,
        ``metadata.base_preset`` set to the effective preset name).

    Raises:
        ValueError: Unknown *preset*, unknown *provider*, or a
            malformed *mcp_servers* entry.
    """
    from geny_executor.llm_client.registry import ClientRegistry

    if preset not in MANIFEST_PRESETS:
        raise ValueError(f"unknown preset {preset!r}. Expected one of: {known_manifest_presets()}")
    if not provider or not isinstance(provider, str):
        raise ValueError(
            f"provider must be a non-empty string; got {provider!r}. "
            f"Registered providers: {ClientRegistry.available()}"
        )
    if provider not in ClientRegistry.available():
        raise ValueError(
            f"unknown provider {provider!r}. Registered providers: {ClientRegistry.available()}"
        )
    for raw in mcp_servers or []:
        if not isinstance(raw, dict) or not raw.get("name"):
            raise ValueError(
                f"mcp_servers entries must be dicts with a 'name'; got {raw!r}. "
                "Nameless servers cannot be routed and would be dropped "
                "silently at connect time."
            )

    # Alias: the agent-session layer uses "default" for the adaptive
    # worker flow. Collapse it so downstream code sees one canonical name.
    effective = _WORKER_ADAPTIVE if preset == _DEFAULT_ALIAS else preset

    now = datetime.now(timezone.utc).isoformat()
    metadata = EnvironmentMetadata(
        id=f"env_{uuid4().hex[:8]}",
        name=name or f"preset:{effective}",
        description=description or f"Manifest materialized from preset '{effective}'.",
        created_at=now,
        updated_at=now,
        base_preset=effective,
    )

    tools = ToolsSnapshot(
        built_in=list(built_in_tools or []),
        external=list(external_tools or []),
        mcp_servers=[dict(s) for s in (mcp_servers or [])],
    )

    entries = _build_stage_entries(effective, provider=provider)

    return EnvironmentManifest(
        metadata=metadata,
        model={"model": model} if model else {},
        pipeline={},
        stages=[e.to_dict() for e in entries],
        tools=tools,
    )


# ── Preset catalog (host-facing, generalised) ────────────────────────
#
# The manifest presets above are stage blueprints; the *catalog* layers
# host-facing selection metadata on top so any consumer (Geny or other)
# can list selectable presets — display name, description, recommended
# Stage-6 provider, tags — without re-deriving them. A host shows
# :func:`preset_catalog`, lets the user pick a key, and materialises it
# via :func:`build_manifest_for`. Geny then builds its *own* custom
# presets on top of these (its tool-bearing templates reference a
# ``base_preset`` / catalog key here).


@dataclass(frozen=True)
class PresetDescriptor:
    """One selectable entry in the built-in preset catalog.

    ``base_preset`` is the canonical 21-stage blueprint (a member of
    :data:`MANIFEST_PRESETS`). ``provider`` is the recommended/locked
    Stage-6 backend (``None`` → the host chooses at build time). ``key``
    is the stable catalog id a host stores; it may differ from
    ``base_preset`` (e.g. ``claude_code_worker`` → base ``worker_adaptive``
    + provider ``claude_code_cli``).
    """

    key: str
    name: str
    description: str
    base_preset: str
    provider: Optional[str] = None
    tags: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "base_preset": self.base_preset,
            "provider": self.provider,
            "tags": list(self.tags),
        }


_PRESET_CATALOG: List[PresetDescriptor] = [
    PresetDescriptor(
        key=_WORKER_ADAPTIVE,
        name="Worker (Adaptive)",
        description="Autonomous tool-using worker — the full 21-stage agentic loop with an adaptive turn budget.",
        base_preset=_WORKER_ADAPTIVE,
        provider=None,
        tags=("worker", "agent"),
    ),
    PresetDescriptor(
        key=_VTUBER,
        name="VTuber",
        description="Conversational persona — a lighter loop with a narrowed tool roster, tuned for TTS replies.",
        base_preset=_VTUBER,
        provider=None,
        tags=("vtuber", "chat"),
    ),
    PresetDescriptor(
        key="claude_code_worker",
        name="Claude Code · Worker",
        description="Worker agentic loop backed by the Claude Code CLI provider (subscription auth, native CLI tool loop).",
        base_preset=_WORKER_ADAPTIVE,
        provider="claude_code_cli",
        tags=("worker", "agent", "claude_code"),
    ),
    PresetDescriptor(
        key="claude_code_vtuber",
        name="Claude Code · VTuber",
        description="Conversational VTuber persona backed by the Claude Code CLI provider.",
        base_preset=_VTUBER,
        provider="claude_code_cli",
        tags=("vtuber", "chat", "claude_code"),
    ),
]


def preset_catalog() -> List[PresetDescriptor]:
    """The built-in, host-facing preset catalog (a fresh list copy)."""
    return list(_PRESET_CATALOG)


def get_preset_descriptor(key: str) -> Optional[PresetDescriptor]:
    """Look up a catalog entry by its ``key`` (``None`` if absent)."""
    for d in _PRESET_CATALOG:
        if d.key == key:
            return d
    return None


def build_manifest_for(
    key: str,
    *,
    provider: Optional[str] = None,
    **kwargs: Any,
) -> EnvironmentManifest:
    """Materialise an :class:`EnvironmentManifest` from a *catalog* key.

    Resolves the descriptor and calls :func:`build_manifest` with its
    ``base_preset`` + provider. The ``provider`` argument overrides the
    descriptor's recommended provider; if neither is set a ``ValueError``
    is raised (the caller must choose a backend). A bare
    :data:`MANIFEST_PRESETS` name is also accepted (then ``provider`` is
    required), so this is a strict superset of :func:`build_manifest`.
    Display ``name`` / ``description`` default to the descriptor's.
    Extra kwargs (``model`` / ``built_in_tools`` / ``external_tools`` /
    ``mcp_servers``) pass through to :func:`build_manifest`.
    """
    desc = get_preset_descriptor(key)
    if desc is None:
        if key in MANIFEST_PRESETS:
            if not provider:
                raise ValueError(f"preset {key!r} requires an explicit provider=")
            return build_manifest(key, provider=provider, **kwargs)
        raise ValueError(f"unknown preset key {key!r}. Catalog: {[d.key for d in _PRESET_CATALOG]}")
    eff_provider = provider or desc.provider
    if not eff_provider:
        raise ValueError(f"preset {key!r} has no recommended provider; pass provider= explicitly.")
    kwargs.setdefault("name", desc.name)
    kwargs.setdefault("description", desc.description)
    return build_manifest(desc.base_preset, provider=eff_provider, **kwargs)
