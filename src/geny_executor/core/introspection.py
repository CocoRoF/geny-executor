"""Session-less introspection over the 16-stage pipeline.

The Environment Builder UI needs to render a form for every stage *before* a
live pipeline exists. Doing this directly against ``create_stage`` works for
most artifacts, but some (e.g. ``s06_api/openai``) require ctor kwargs that a
naive caller can't know. This module encapsulates that knowledge so callers can
ask simple questions::

    from geny_executor.core.introspection import introspect_stage, introspect_all

    insp = introspect_stage("s06_api", "openai")
    for slot in insp.strategy_slots.values():
        print(slot.slot_name, slot.available_impls, slot.impl_schemas)

All introspection is pure: no network calls, no filesystem writes. Stages that
cannot be safely instantiated (e.g. ``Stage = None`` strategy-only artifacts)
raise :class:`IntrospectionUnsupported` with a clear message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from geny_executor.core.artifact import (
    ArtifactInfo,
    DEFAULT_ARTIFACT,
    STAGE_MODULES,
    _resolve_stage_module,
    create_stage,
    describe_artifact,
    load_artifact_module,
)
from geny_executor.core.schema import ConfigSchema
from geny_executor.core.stage import Stage


class IntrospectionUnsupported(RuntimeError):
    """Raised when an artifact cannot be introspected (e.g. strategy-only)."""


# ── Per-stage ctor kwargs for introspection ─────────────────────
#
# Stages whose default ctor requires credentials (e.g. APIStage demands
# ``api_key`` or an explicit provider) receive a dummy string. No network
# call is issued during introspection — AnthropicProvider lazy-instantiates
# its SDK client on the first ``create_message`` call — so the dummy key is
# never exercised. Using a real (even fake) ``api_key`` is load-bearing: it
# makes ``slot.current_impl`` on the provider slot resolve to ``"anthropic"``
# (the runtime default) instead of ``"mock"``, so ``blank_manifest`` does
# not accidentally seed new environments with the test-only MockProvider.


_STAGE_INTROSPECTION_KWARGS: Dict[tuple, Any] = {
    ("s06_api", "default"): {"api_key": "introspection-dummy-key"},
    ("s06_api", "openai"): {"api_key": "introspection-dummy-key"},
    ("s06_api", "google"): {"api_key": "introspection-dummy-key"},
}


def _introspection_kwargs(stage_module: str, artifact: str) -> Dict[str, Any]:
    """Return ctor kwargs for introspecting (stage_module, artifact)."""
    specific = _STAGE_INTROSPECTION_KWARGS.get((stage_module, artifact))
    if specific is None:
        return {}
    if callable(specific):
        return dict(specific())
    return dict(specific)


# ── Per-stage runtime capabilities ──────────────────────────────
#
# Not every stage actually consumes ``tool_binding`` or ``model_override`` at
# runtime — most are plumbing. A UI that offers the binding/override inputs on
# every stage misleads the user into editing fields that get silently ignored.
#
# The truth, verified by grepping ``self.tool_binding`` / ``self.model_override``
# reads across ``src/geny_executor``:
#
#   - ``s02_context`` consumes ``model_override`` via ``LLMSummaryCompactor``
#     (gates the LLM-backed summarization path).
#   - ``s06_api`` reads ``self.model_override`` in ``_build_request`` — the
#     primary LLM-calling stage (overrides model / max_tokens / sampling /
#     thinking).
#   - ``s10_tool`` reads ``self.tool_binding`` in ``execute`` (only tool-
#     calling stage — enforces the per-stage allow/block list).
#   - ``s18_memory`` consumes ``model_override`` via
#     ``GenyMemoryStrategy._reflect`` native path (gates reflective
#     insight extraction).
#   - Every other stage reads neither.
#
# Alternative artifacts for these stages (e.g. ``s06_api/openai``) inherit the
# same capability because they're still the LLM-call / tool-call stage.
# If a future stage starts consuming these, add an entry here.
#
# Keyed by stage *module* name — same granularity as
# ``_STAGE_INTROSPECTION_KWARGS`` above.

_StageCapabilities = Dict[str, bool]

_STAGE_CAPABILITIES: Dict[str, _StageCapabilities] = {
    "s02_context": {"tool_binding": False, "model_override": True},
    "s06_api": {"tool_binding": False, "model_override": True},
    "s10_tool": {"tool_binding": True, "model_override": False},
    "s18_memory": {"tool_binding": False, "model_override": True},
}


def _stage_capabilities(stage_module: str) -> _StageCapabilities:
    """Return the ``(tool_binding, model_override)`` support flags for a stage.

    Unknown stages default to both-False — the safe position, since claiming
    support the runtime doesn't actually exercise is worse than under-promising.
    """
    return _STAGE_CAPABILITIES.get(stage_module, {"tool_binding": False, "model_override": False})


# ── Structurally required stages ────────────────────────────────
#
# Every geny-executor pipeline is fundamentally an LLM agent loop. Four stages
# are load-bearing for that contract and cannot be deactivated without making
# the pipeline meaningless:
#
#   - ``s01_input``  — turns the user's prompt into the initial artifact; no
#     pipeline can start without it.
#   - ``s06_api``    — the LLM call itself; removing it leaves nothing to parse
#     or emit.
#   - ``s09_parse``  — converts raw API output into the typed events every
#     downstream stage (tool, loop, memory, yield) consumes.
#   - ``s21_yield``  — surfaces the final result to the caller; without it the
#     run produces no output.
#
# This set mirrors the ``minimal`` preset (Input → API → Parse → Yield), which
# is the smallest canonical PipelineBuilder config. Every other stage is
# optional — a UI should let users toggle them, but the runtime will happily
# accept a manifest that omits them.
#
# Environment Builder UIs read this via ``StageIntrospection.required`` and
# force the stage's Active toggle on.

_STAGE_REQUIRED: Set[str] = {"s01_input", "s06_api", "s09_parse", "s21_yield"}


def _stage_required(stage_module: str) -> bool:
    """Return True if the stage is structurally required for any pipeline."""
    return stage_module in _STAGE_REQUIRED


# ── Dataclasses ────────────────────────────────────────────────


@dataclass(frozen=True)
class SlotIntrospection:
    """Introspection view of a single :class:`StrategySlot`."""

    slot_name: str
    description: str
    required: bool
    current_impl: str
    available_impls: List[str]
    impl_schemas: Dict[str, Optional[ConfigSchema]]
    impl_descriptions: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready representation."""
        return {
            "slot_name": self.slot_name,
            "description": self.description,
            "required": self.required,
            "current_impl": self.current_impl,
            "available_impls": list(self.available_impls),
            "impl_schemas": {
                name: (schema.to_json_schema() if schema is not None else None)
                for name, schema in self.impl_schemas.items()
            },
            "impl_descriptions": dict(self.impl_descriptions),
        }


@dataclass(frozen=True)
class ChainIntrospection:
    """Introspection view of a single :class:`SlotChain`."""

    chain_name: str
    description: str
    current_impls: List[str]  # names in their current order
    available_impls: List[str]
    impl_schemas: Dict[str, Optional[ConfigSchema]]
    impl_descriptions: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready representation."""
        return {
            "chain_name": self.chain_name,
            "description": self.description,
            "current_impls": list(self.current_impls),
            "available_impls": list(self.available_impls),
            "impl_schemas": {
                name: (schema.to_json_schema() if schema is not None else None)
                for name, schema in self.impl_schemas.items()
            },
            "impl_descriptions": dict(self.impl_descriptions),
        }


@dataclass(frozen=True)
class StageIntrospection:
    """Complete introspection view of one stage+artifact combination."""

    stage: str  # canonical module name, e.g. "s06_api"
    artifact: str
    order: int
    name: str
    category: str
    artifact_info: ArtifactInfo
    config_schema: Optional[ConfigSchema]
    config: Dict[str, Any]
    strategy_slots: Dict[str, SlotIntrospection]
    strategy_chains: Dict[str, ChainIntrospection]
    tool_binding_supported: bool = False
    model_override_supported: bool = False
    required: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready representation."""
        return {
            "stage": self.stage,
            "artifact": self.artifact,
            "order": self.order,
            "name": self.name,
            "category": self.category,
            "artifact_info": self.artifact_info.to_dict(),
            "config_schema": (self.config_schema.to_json_schema() if self.config_schema else None),
            "config": dict(self.config),
            "strategy_slots": {name: slot.to_dict() for name, slot in self.strategy_slots.items()},
            "strategy_chains": {
                name: chain.to_dict() for name, chain in self.strategy_chains.items()
            },
            "tool_binding_supported": self.tool_binding_supported,
            "model_override_supported": self.model_override_supported,
            "required": self.required,
            "extra": dict(self.extra),
        }


# ── Core helpers ───────────────────────────────────────────────


def _strategy_description(cls: Any) -> str:
    """Extract a human-readable description from a Strategy class."""
    try:
        instance = cls()
    except Exception:  # pragma: no cover - defensive
        return ""
    try:
        return getattr(instance, "description", "") or ""
    except Exception:  # pragma: no cover
        return ""


def _strategy_schema(cls: Any) -> Optional[ConfigSchema]:
    """Return the ConfigSchema for a Strategy class, or None."""
    fn = getattr(cls, "config_schema", None)
    if fn is None:
        return None
    try:
        return fn()
    except Exception:  # pragma: no cover - defensive
        return None


def _introspect_slot(slot: Any) -> SlotIntrospection:
    impl_schemas: Dict[str, Optional[ConfigSchema]] = {}
    impl_descriptions: Dict[str, str] = {}
    for impl_name, impl_cls in slot.registry.items():
        impl_schemas[impl_name] = _strategy_schema(impl_cls)
        impl_descriptions[impl_name] = _strategy_description(impl_cls)
    return SlotIntrospection(
        slot_name=slot.name,
        description=slot.description,
        required=slot.required,
        current_impl=slot.current_impl,
        available_impls=slot.available_impls,
        impl_schemas=impl_schemas,
        impl_descriptions=impl_descriptions,
    )


def _introspect_chain(chain: Any) -> ChainIntrospection:
    impl_schemas: Dict[str, Optional[ConfigSchema]] = {}
    impl_descriptions: Dict[str, str] = {}
    for impl_name, impl_cls in chain.registry.items():
        impl_schemas[impl_name] = _strategy_schema(impl_cls)
        impl_descriptions[impl_name] = _strategy_description(impl_cls)
    return ChainIntrospection(
        chain_name=chain.name,
        description=chain.description,
        current_impls=[item.name for item in chain.items],
        available_impls=chain.available_impls,
        impl_schemas=impl_schemas,
        impl_descriptions=impl_descriptions,
    )


# ── Public API ─────────────────────────────────────────────────


def introspect_stage(stage: str, artifact: str = DEFAULT_ARTIFACT) -> StageIntrospection:
    """Introspect a stage+artifact combination without running a pipeline.

    Args:
        stage: Stage identifier (module name, alias, or order).
        artifact: Artifact name (default: ``"default"``).

    Returns:
        A :class:`StageIntrospection` snapshot of the stage's configurable
        surface: its own :class:`ConfigSchema`, its strategy slots/chains,
        and the per-impl schemas those slots support.

    Raises:
        IntrospectionUnsupported: If the artifact is strategy-only
            (``Stage = None``), since there is no Stage class to inspect.
        ImportError: If the artifact module cannot be loaded.
    """
    module_name = _resolve_stage_module(stage)
    mod = load_artifact_module(module_name, artifact)
    if getattr(mod, "Stage", None) is None:
        raise IntrospectionUnsupported(
            f"Artifact '{artifact}' for stage '{module_name}' is strategy-only "
            f"(Stage is None). Introspect '{DEFAULT_ARTIFACT}' and swap this "
            f"artifact's strategies into the default stage's slots instead."
        )

    kwargs = _introspection_kwargs(module_name, artifact)
    instance: Stage = create_stage(module_name, artifact, **kwargs)

    slots_raw = instance.get_strategy_slots()
    chains_raw = instance.get_strategy_chains()
    strategy_slots = {name: _introspect_slot(slot) for name, slot in slots_raw.items()}
    strategy_chains = {name: _introspect_chain(chain) for name, chain in chains_raw.items()}
    caps = _stage_capabilities(module_name)

    return StageIntrospection(
        stage=module_name,
        artifact=artifact,
        order=instance.order,
        name=instance.name,
        category=instance.category,
        artifact_info=describe_artifact(module_name, artifact),
        config_schema=instance.get_config_schema(),
        config=instance.get_config(),
        strategy_slots=strategy_slots,
        strategy_chains=strategy_chains,
        tool_binding_supported=caps["tool_binding"],
        model_override_supported=caps["model_override"],
        required=_stage_required(module_name),
    )


def introspect_all(
    artifacts: Optional[Dict[str, str]] = None,
) -> List[StageIntrospection]:
    """Introspect every registered stage in canonical order.

    Args:
        artifacts: Optional per-stage override mapping ``stage_identifier → artifact``.
            Unspecified stages use :data:`DEFAULT_ARTIFACT`.

    Returns:
        A list of :class:`StageIntrospection` objects, ordered by stage order
        (1–16). Strategy-only artifacts silently fall back to ``"default"`` so
        ``introspect_all`` never raises for a well-formed override map.
    """
    overrides: Dict[str, str] = {}
    if artifacts:
        for key, art in artifacts.items():
            overrides[_resolve_stage_module(key)] = art

    results: List[StageIntrospection] = []
    for order in sorted(STAGE_MODULES):
        module_name = STAGE_MODULES[order]
        artifact = overrides.get(module_name, DEFAULT_ARTIFACT)
        try:
            results.append(introspect_stage(module_name, artifact))
        except IntrospectionUnsupported:
            # Fall back to default so the caller still gets a full 16-element result.
            results.append(introspect_stage(module_name, DEFAULT_ARTIFACT))
    return results
