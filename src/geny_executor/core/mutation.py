"""PipelineMutator — safe runtime mutation of pipeline configuration.

Provides a controlled API for modifying pipeline stages, strategies,
and configuration at runtime while maintaining consistency guarantees.
"""

from __future__ import annotations

import itertools
import logging
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Set

from geny_executor.core.errors import MutationError, MutationLocked
from geny_executor.core.snapshot import PipelineSnapshot, StageSnapshot
from geny_executor.core.stage import Stage

if TYPE_CHECKING:
    from geny_executor.core.pipeline import Pipeline


logger = logging.getLogger(__name__)

_HOOK_EVENTS = {"on_enter", "on_exit", "on_error"}


async def _await_maybe(value: Any) -> Any:
    """Await *value* if it is awaitable; otherwise return as-is."""
    if hasattr(value, "__await__"):
        return await value
    return value


class MutationKind(str, Enum):
    """Type of mutation performed."""

    SWAP_STRATEGY = "swap_strategy"
    UPDATE_STAGE_CONFIG = "update_stage_config"
    UPDATE_STRATEGY_CONFIG = "update_strategy_config"
    UPDATE_MODEL_CONFIG = "update_model_config"
    UPDATE_PIPELINE_CONFIG = "update_pipeline_config"
    SET_STAGE_ACTIVE = "set_stage_active"
    REGISTER_STAGE = "register_stage"
    REMOVE_STAGE = "remove_stage"
    REPLACE_STAGE = "replace_stage"
    REORDER_CHAIN = "reorder_chain"
    ADD_TO_CHAIN = "add_to_chain"
    REMOVE_FROM_CHAIN = "remove_from_chain"
    REGISTER_HOOK = "register_hook"
    UNREGISTER_HOOK = "unregister_hook"
    BIND_TOOL = "bind_tool"
    UNBIND_TOOL = "unbind_tool"
    SET_TOOL_SCOPE = "set_tool_scope"
    SET_STAGE_MODEL = "set_stage_model"
    RESTORE_SNAPSHOT = "restore_snapshot"


@dataclass
class MutationRecord:
    """A single mutation in the change log."""

    kind: MutationKind
    target: str  # e.g. "stage:6.provider", "model.temperature"
    old_value: Any = None
    new_value: Any = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class MutationResult:
    """Result of a mutation operation."""

    success: bool
    message: str = ""
    record: Optional[MutationRecord] = None


@dataclass
class RestoreReport:
    """What :meth:`PipelineMutator.restore` actually applied vs skipped.

    Restore used to be warn-free AND report-free: every unknown slot,
    unknown impl, and incompatible chain order hit a silent ``pass``,
    so a manifest could "restore" with half its declarations dropped
    and the only symptom was wrong runtime behaviour weeks later
    (2.2.0, audit 2026-06-09 §2.1 — Geny prod's evaluator config was
    lost exactly this way). ``restore(snapshot, report=True)`` returns
    this instead of the legacy :class:`MutationResult` so callers can
    inspect — or refuse to ship — a degraded restore.

    Fields (all entries are human-readable target strings,
    ``"stage:{order}.{name}"`` shaped, matching the change-log
    ``target`` convention):
      - ``configured``: targets applied successfully (slot strategies
        swapped/configured, stage configs, chains, tool bindings,
        model overrides).
      - ``skipped_slots``: snapshot slots the live stage doesn't have.
      - ``skipped_impls``: slots that exist but whose snapshot impl
        could not be set (unknown implementation name).
      - ``skipped_chains``: chains that don't exist on the live stage
        or rejected the snapshot ordering.
      - ``errors``: anything else that prevented application — e.g. a
        snapshot entry marked active for a stage that is not
        registered, or a strategy config the target's ``configure()``
        rejected with ``ValueError`` (the strategy keeps its defaults;
        2.2.0 review B4). Stage-construction failures never appear here
        (restore does not build stages); see
        ``Pipeline.dropped_stages`` for those.
    """

    configured: List[str] = field(default_factory=list)
    skipped_slots: List[str] = field(default_factory=list)
    skipped_impls: List[str] = field(default_factory=list)
    skipped_chains: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def has_skips(self) -> bool:
        """True when any declaration in the snapshot failed to land."""
        return bool(self.skipped_slots or self.skipped_impls or self.skipped_chains or self.errors)


class PipelineMutator:
    """Controlled mutation layer over a :class:`Pipeline`.

    All mutations are recorded in a change log. Config/strategy/
    structure mutations are additionally blocked with
    :class:`MutationLocked` while the pipeline is executing a run
    (``Pipeline.run_in_progress`` — engine-wired since 2.2.0); mutate
    between turns.

    Usage::

        mutator = PipelineMutator(pipeline)
        mutator.swap_strategy(6, "provider", "openai")
        mutator.update_model_config({"temperature": 0.7})
        snapshot = mutator.snapshot()
    """

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline
        self._change_log: List[MutationRecord] = []
        self._lock = threading.Lock()
        self._locked_stages: Set[int] = set()
        # (stage_order, event) → list of (hook_id, callback)
        self._hooks: Dict[tuple, List[tuple]] = {}
        self._hook_counter = itertools.count()

    # ── Public API ──────────────────────────────────────────

    def swap_strategy(
        self,
        stage_order: int,
        slot_name: str,
        impl_name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> MutationResult:
        """Replace a strategy in a stage's slot.

        Args:
            stage_order: Stage number (1-16).
            slot_name: Strategy slot identifier.
            impl_name: New implementation name from the slot's registry.
            config: Optional configuration for the new strategy.
        """
        with self._lock:
            self._check_stage_lock(stage_order)
            stage = self._get_stage(stage_order)
            old_impls = {s.slot_name: s.current_impl for s in stage.list_strategies()}
            old_value = old_impls.get(slot_name, "")

            try:
                stage.set_strategy(slot_name, impl_name, config)
            except KeyError as exc:
                raise MutationError(str(exc), stage_order=stage_order, slot_name=slot_name) from exc

            record = MutationRecord(
                kind=MutationKind.SWAP_STRATEGY,
                target=f"stage:{stage_order}.{slot_name}",
                old_value=old_value,
                new_value=impl_name,
            )
            self._change_log.append(record)
            return MutationResult(success=True, record=record)

    def update_stage_config(self, stage_order: int, config: Dict[str, Any]) -> MutationResult:
        """Apply a partial config update to a stage."""
        with self._lock:
            self._check_stage_lock(stage_order)
            stage = self._get_stage(stage_order)
            old_config = stage.get_config()

            stage.update_config(config)

            record = MutationRecord(
                kind=MutationKind.UPDATE_STAGE_CONFIG,
                target=f"stage:{stage_order}",
                old_value=old_config,
                new_value=config,
            )
            self._change_log.append(record)
            return MutationResult(success=True, record=record)

    def update_model_config(self, changes: Dict[str, Any]) -> MutationResult:
        """Update model configuration fields (temperature, max_tokens, etc.)."""
        with self._lock:
            self._check_run_lock()
            cfg = self._pipeline._config.model
            old_values: Dict[str, Any] = {}
            for key, value in changes.items():
                if not hasattr(cfg, key):
                    raise MutationError(f"ModelConfig has no field '{key}'")
                old_values[key] = getattr(cfg, key)
                setattr(cfg, key, value)

            record = MutationRecord(
                kind=MutationKind.UPDATE_MODEL_CONFIG,
                target="model",
                old_value=old_values,
                new_value=changes,
            )
            self._change_log.append(record)
            return MutationResult(success=True, record=record)

    def update_pipeline_config(self, changes: Dict[str, Any]) -> MutationResult:
        """Update top-level pipeline configuration fields."""
        with self._lock:
            self._check_run_lock()
            cfg = self._pipeline._config
            old_values: Dict[str, Any] = {}
            for key, value in changes.items():
                if key == "model":
                    continue  # use update_model_config() instead
                if not hasattr(cfg, key):
                    raise MutationError(f"PipelineConfig has no field '{key}'")
                old_values[key] = getattr(cfg, key)
                setattr(cfg, key, value)

            record = MutationRecord(
                kind=MutationKind.UPDATE_PIPELINE_CONFIG,
                target="pipeline",
                old_value=old_values,
                new_value=changes,
            )
            self._change_log.append(record)
            return MutationResult(success=True, record=record)

    def set_stage_active(self, stage_order: int, active: bool) -> MutationResult:
        """Enable or disable a stage.

        Disabled stages are removed from the pipeline and will be bypassed.
        Re-enabling requires the stage to have been previously registered.
        """
        with self._lock:
            self._check_stage_lock(stage_order)
            stage = self._pipeline.get_stage(stage_order)

            if active and stage_order in getattr(self, "_removed_stages", {}):
                restored = self._removed_stages.pop(stage_order)
                self._pipeline.register_stage(restored)
            elif active and stage is None:
                raise MutationError(
                    f"Cannot activate stage {stage_order}: not registered. "
                    f"Register the stage first.",
                    stage_order=stage_order,
                )

            if not active and stage is not None:
                if not hasattr(self, "_removed_stages"):
                    self._removed_stages: Dict[int, Any] = {}
                self._removed_stages[stage_order] = stage
                self._pipeline.remove_stage(stage_order)

            record = MutationRecord(
                kind=MutationKind.SET_STAGE_ACTIVE,
                target=f"stage:{stage_order}",
                old_value=not active,
                new_value=active,
            )
            self._change_log.append(record)
            return MutationResult(success=True, record=record)

    def update_strategy_config(
        self, stage_order: int, slot_name: str, config: Dict[str, Any]
    ) -> MutationResult:
        """Patch the config of the currently-selected strategy in a slot.

        Does not re-instantiate — the current Strategy is updated in place
        via its ``configure()`` method.
        """
        with self._lock:
            self._check_stage_lock(stage_order)
            stage = self._get_stage(stage_order)
            slots = stage.get_strategy_slots()
            slot = slots.get(slot_name)
            if slot is None:
                raise MutationError(
                    f"Stage '{stage.name}' has no strategy slot '{slot_name}'",
                    stage_order=stage_order,
                    slot_name=slot_name,
                )
            old_config: Dict[str, Any] = {}
            if hasattr(slot.strategy, "get_config"):
                old_config = dict(slot.strategy.get_config())
            slot.strategy.configure(config)

            record = MutationRecord(
                kind=MutationKind.UPDATE_STRATEGY_CONFIG,
                target=f"stage:{stage_order}.{slot_name}",
                old_value=old_config,
                new_value=config,
            )
            self._change_log.append(record)
            return MutationResult(success=True, record=record)

    def replace_stage(self, stage_order: int, new_stage: Stage) -> MutationResult:
        """Fully replace the Stage instance at *stage_order*."""
        if new_stage.order != stage_order:
            raise MutationError(
                f"new_stage.order={new_stage.order} does not match "
                f"target stage_order={stage_order}",
                stage_order=stage_order,
            )
        with self._lock:
            self._check_stage_lock(stage_order)
            old_stage = self._pipeline.get_stage(stage_order)
            self._pipeline.register_stage(new_stage)

            record = MutationRecord(
                kind=MutationKind.REPLACE_STAGE,
                target=f"stage:{stage_order}",
                old_value=type(old_stage).__name__ if old_stage else None,
                new_value=type(new_stage).__name__,
            )
            self._change_log.append(record)
            return MutationResult(success=True, record=record)

    def reorder_chain(self, stage_order: int, chain_name: str, order: List[str]) -> MutationResult:
        """Reorder items in a stage's chain to the given permutation."""
        with self._lock:
            self._check_stage_lock(stage_order)
            stage = self._get_stage(stage_order)
            chains = stage.get_strategy_chains()
            chain = chains.get(chain_name)
            if chain is None:
                raise MutationError(
                    f"Stage '{stage.name}' has no chain '{chain_name}'",
                    stage_order=stage_order,
                )
            old_order = [item.name for item in chain.items]
            try:
                stage.reorder_chain(chain_name, order)
            except (KeyError, ValueError) as exc:
                raise MutationError(str(exc), stage_order=stage_order) from exc

            record = MutationRecord(
                kind=MutationKind.REORDER_CHAIN,
                target=f"stage:{stage_order}.{chain_name}",
                old_value=old_order,
                new_value=list(order),
            )
            self._change_log.append(record)
            return MutationResult(success=True, record=record)

    def add_to_chain(
        self,
        stage_order: int,
        chain_name: str,
        impl_name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> MutationResult:
        """Append a new strategy onto a stage's chain."""
        with self._lock:
            self._check_stage_lock(stage_order)
            stage = self._get_stage(stage_order)
            try:
                stage.add_to_chain(chain_name, impl_name, config)
            except KeyError as exc:
                raise MutationError(str(exc), stage_order=stage_order) from exc

            record = MutationRecord(
                kind=MutationKind.ADD_TO_CHAIN,
                target=f"stage:{stage_order}.{chain_name}",
                old_value=None,
                new_value={"impl": impl_name, "config": config or {}},
            )
            self._change_log.append(record)
            return MutationResult(success=True, record=record)

    def remove_from_chain(
        self, stage_order: int, chain_name: str, item_name: str
    ) -> MutationResult:
        """Remove an item from a stage's chain by its strategy name."""
        with self._lock:
            self._check_stage_lock(stage_order)
            stage = self._get_stage(stage_order)
            try:
                removed = stage.remove_from_chain(chain_name, item_name)
            except KeyError as exc:
                raise MutationError(str(exc), stage_order=stage_order) from exc

            record = MutationRecord(
                kind=MutationKind.REMOVE_FROM_CHAIN,
                target=f"stage:{stage_order}.{chain_name}",
                old_value=type(removed).__name__,
                new_value=None,
            )
            self._change_log.append(record)
            return MutationResult(success=True, record=record)

    # ── Hooks ───────────────────────────────────────────────

    def register_hook(
        self,
        stage_order: int,
        event: str,
        callback: Callable[..., Any],
    ) -> MutationResult:
        """Attach *callback* to a stage lifecycle event.

        Supported events: ``on_enter``, ``on_exit``, ``on_error``.
        The callback is invoked in registration order after the stage's own
        hook method runs. Its signature should match the corresponding
        :class:`Stage` hook (``(state,)``, ``(result, state)``, or
        ``(error, state)``).

        Returns a :class:`MutationResult` whose ``message`` contains the
        stable ``hook_id`` that :meth:`unregister_hook` accepts.
        """
        if event not in _HOOK_EVENTS:
            raise MutationError(f"Unknown hook event '{event}'. Allowed: {sorted(_HOOK_EVENTS)}")
        with self._lock:
            stage = self._get_stage(stage_order)
            hook_id = f"hook_{next(self._hook_counter)}_{uuid.uuid4().hex[:6]}"
            key = (stage_order, event)
            if key not in self._hooks:
                self._install_hook_bridge(stage, event, key)
            self._hooks.setdefault(key, []).append((hook_id, callback))

            record = MutationRecord(
                kind=MutationKind.REGISTER_HOOK,
                target=f"stage:{stage_order}.{event}",
                old_value=None,
                new_value=hook_id,
            )
            self._change_log.append(record)
            return MutationResult(success=True, message=hook_id, record=record)

    def unregister_hook(self, stage_order: int, event: str, hook_id: str) -> MutationResult:
        """Remove a previously registered hook by id."""
        with self._lock:
            key = (stage_order, event)
            hooks = self._hooks.get(key, [])
            for idx, (hid, _cb) in enumerate(hooks):
                if hid == hook_id:
                    hooks.pop(idx)
                    record = MutationRecord(
                        kind=MutationKind.UNREGISTER_HOOK,
                        target=f"stage:{stage_order}.{event}",
                        old_value=hook_id,
                        new_value=None,
                    )
                    self._change_log.append(record)
                    return MutationResult(success=True, record=record)
            raise MutationError(
                f"No hook '{hook_id}' registered at stage {stage_order}.{event}",
                stage_order=stage_order,
            )

    def _install_hook_bridge(self, stage: Stage, event: str, key: tuple) -> None:
        """Wrap the stage's lifecycle method so registered callbacks fire."""
        original = getattr(stage, event)

        if event == "on_enter":

            async def wrapper(state):  # type: ignore[no-redef]
                result = await original(state)
                for _hid, cb in list(self._hooks.get(key, [])):
                    await _await_maybe(cb(state))
                return result
        elif event == "on_exit":

            async def wrapper(result, state):  # type: ignore[no-redef]
                ret = await original(result, state)
                for _hid, cb in list(self._hooks.get(key, [])):
                    await _await_maybe(cb(result, state))
                return ret
        else:  # on_error

            async def wrapper(error, state):  # type: ignore[no-redef]
                ret = await original(error, state)
                for _hid, cb in list(self._hooks.get(key, [])):
                    await _await_maybe(cb(error, state))
                return ret

        setattr(stage, event, wrapper)

    # ── Tool binding ────────────────────────────────────────

    def bind_tool_to_stage(self, stage_order: int, tool_name: str) -> MutationResult:
        """Grant *stage_order* access to *tool_name*."""
        with self._lock:
            self._check_run_lock()
            stage = self._get_stage(stage_order)
            stage.tool_binding.allow(tool_name)
            record = MutationRecord(
                kind=MutationKind.BIND_TOOL,
                target=f"stage:{stage_order}.tools",
                new_value=tool_name,
            )
            self._change_log.append(record)
            return MutationResult(success=True, record=record)

    def unbind_tool_from_stage(self, stage_order: int, tool_name: str) -> MutationResult:
        """Revoke *stage_order* access to *tool_name*."""
        with self._lock:
            self._check_run_lock()
            stage = self._get_stage(stage_order)
            stage.tool_binding.block(tool_name)
            record = MutationRecord(
                kind=MutationKind.UNBIND_TOOL,
                target=f"stage:{stage_order}.tools",
                new_value=tool_name,
            )
            self._change_log.append(record)
            return MutationResult(success=True, record=record)

    def set_stage_tool_scope(
        self,
        stage_order: int,
        allowed: Optional[List[str]] = None,
        blocked: Optional[List[str]] = None,
    ) -> MutationResult:
        """Replace the whole tool scope for a stage.

        ``allowed=None`` means inherit-everything; ``blocked=None`` means
        no blocks. Passing an empty list intentionally restricts to
        nothing / blocks nothing.
        """
        with self._lock:
            self._check_run_lock()
            stage = self._get_stage(stage_order)
            binding = stage.tool_binding
            binding.allowed = set(allowed) if allowed is not None else None
            binding.blocked = set(blocked) if blocked is not None else None
            record = MutationRecord(
                kind=MutationKind.SET_TOOL_SCOPE,
                target=f"stage:{stage_order}.tools",
                new_value={
                    "allowed": list(binding.allowed) if binding.allowed else None,
                    "blocked": list(binding.blocked) if binding.blocked else None,
                },
            )
            self._change_log.append(record)
            return MutationResult(success=True, record=record)

    # ── Model override ──────────────────────────────────────

    def set_stage_model(self, stage_order: int, model: Optional[Any]) -> MutationResult:
        """Override the model used by a single stage.

        Pass ``None`` to revert the stage to the pipeline-wide model.
        """
        with self._lock:
            self._check_run_lock()
            stage = self._get_stage(stage_order)
            stage.model_override = model
            record = MutationRecord(
                kind=MutationKind.SET_STAGE_MODEL,
                target=f"stage:{stage_order}.model",
                new_value=getattr(model, "model", None) if model else None,
            )
            self._change_log.append(record)
            return MutationResult(success=True, record=record)

    # ── Batch ───────────────────────────────────────────────

    @contextmanager
    def batch(self) -> Iterator["PipelineMutator"]:
        """Atomic multi-mutation context.

        All mutations committed inside the ``with`` block are rolled back
        if any exception escapes. Uses :meth:`snapshot` / :meth:`restore`
        to restore configuration state.

        Usage::

            with mutator.batch() as b:
                b.swap_strategy(6, "provider", "openai")
                b.update_model_config({"temperature": 0.3})
        """
        checkpoint = self.snapshot(description="batch-checkpoint")
        log_before = len(self._change_log)
        try:
            yield self
        except Exception:
            self.restore(checkpoint)
            # Discard records written during the failed batch
            del self._change_log[log_before:]
            raise

    # ── Snapshot ────────────────────────────────────────────

    def snapshot(self, description: str = "") -> PipelineSnapshot:
        """Capture the current pipeline configuration state.

        Captures the full v2 surface: strategy selections, stage config,
        artifact provenance, per-stage tool_binding, per-stage model_override,
        and chain ordering. Fields that were never set (e.g. model_override
        on a stage without overrides) serialize as ``None`` / ``{}``.
        """
        stages: List[StageSnapshot] = []
        # Sub-phase 9a (S9a.3) widened the layout from 16 to 21 stages.
        # Walk the live STAGE_MODULES registry instead of hard-coding
        # the upper bound so future renumberings do not need a code
        # edit here.
        from geny_executor.core.artifact import STAGE_MODULES

        for order in sorted(STAGE_MODULES):
            stage = self._pipeline.get_stage(order)
            if stage:
                strategies: Dict[str, str] = {}
                strategy_configs: Dict[str, Dict[str, Any]] = {}
                # list_strategies includes both slots and chains; only slots
                # carry a single current_impl we can use directly. Chains are
                # captured separately via chain_order.
                for slot_name, slot in stage.get_strategy_slots().items():
                    info = slot.describe()
                    strategies[slot_name] = info.current_impl
                    if info.config:
                        strategy_configs[slot_name] = info.config

                chain_order: Dict[str, List[str]] = {}
                for chain_name, chain in stage.get_strategy_chains().items():
                    chain_order[chain_name] = [item.name for item in chain.items]

                tool_binding_payload: Optional[Dict[str, Any]] = None
                binding = getattr(stage, "_tool_binding", None)
                if binding is not None:
                    tool_binding_payload = binding.to_dict()

                model_override_payload: Optional[Dict[str, Any]] = None
                model_override = getattr(stage, "_model_override", None)
                if model_override is not None and hasattr(model_override, "to_dict"):
                    model_override_payload = model_override.to_dict()

                stages.append(
                    StageSnapshot(
                        order=order,
                        name=stage.name,
                        is_active=True,
                        strategies=strategies,
                        strategy_configs=strategy_configs,
                        stage_config=stage.get_config(),
                        artifact=stage.artifact_name,
                        tool_binding=tool_binding_payload,
                        model_override=model_override_payload,
                        chain_order=chain_order,
                    )
                )
            else:
                stages.append(StageSnapshot(order=order, name=f"stage_{order}", is_active=False))

        # Serialize PipelineConfig
        cfg = self._pipeline._config
        model_dict = cfg.model.to_dict()
        pipeline_dict = {
            "name": cfg.name,
            "max_iterations": cfg.max_iterations,
            "cost_budget_usd": cfg.cost_budget_usd,
            "context_window_budget": cfg.context_window_budget,
            "stream": cfg.stream,
            "single_turn": cfg.single_turn,
            "artifacts": cfg.artifacts,
            "metadata": cfg.metadata,
        }

        return PipelineSnapshot(
            pipeline_name=cfg.name,
            stages=stages,
            pipeline_config=pipeline_dict,
            model_config=model_dict,
            description=description,
        )

    def restore(self, snapshot: PipelineSnapshot, *, report: bool = False) -> Any:
        """Restore pipeline configuration from a snapshot.

        This restores strategy selections and configurations. It does NOT
        replace Stage instances themselves — stages must already be registered.
        v2 snapshots additionally restore chain ordering, tool_binding, and
        model_override; absent fields in v1 snapshots are no-ops.

        Args:
            snapshot: The configuration snapshot to replay.
            report: When True, return a :class:`RestoreReport` detailing
                every declaration that was applied vs skipped (2.2.0,
                audit §2.1/§3.5 — restore's silent ``pass`` branches hid
                real manifest drift in prod). When False (default), the
                historical :class:`MutationResult` return is preserved
                for back-compat. The skip/apply *behaviour* is identical
                either way — only the visibility differs.

        Run-lock exemption: unlike the other mutating operations this
        method does NOT consult the run-in-progress lock — it is the
        rollback primitive :meth:`batch` relies on and the wiring step
        ``Pipeline.from_manifest`` runs at build time; gating it would
        make a failed batch unrecoverable.
        """
        from geny_executor.core.config import ModelConfig
        from geny_executor.tools.stage_binding import StageToolBinding

        outcome = RestoreReport()

        with self._lock:
            for stage_snap in snapshot.stages:
                stage = self._pipeline.get_stage(stage_snap.order)
                if stage is None:
                    if stage_snap.is_active:
                        # The snapshot says this slot should be live but
                        # nothing is registered there — restore cannot
                        # build stages, so the whole entry is dropped.
                        outcome.errors.append(
                            f"stage:{stage_snap.order} ({stage_snap.name}) is "
                            "active in the snapshot but not registered on the "
                            "pipeline"
                        )
                    continue  # Cannot restore an unregistered stage

                # Restore strategy selections. When the live slot already
                # holds the same implementation (common case for APIStage
                # whose provider was constructed with an api_key at
                # :meth:`Pipeline.from_manifest` time), skip the swap and
                # only replay the captured config — rebuilding the instance
                # via ``cls()`` would discard constructor-bound state such
                # as credentials, leaving the runtime provider unusable.
                slots = stage.get_strategy_slots()
                for slot_name, impl_name in stage_snap.strategies.items():
                    slot_config = stage_snap.strategy_configs.get(slot_name)
                    target = f"stage:{stage_snap.order}.{slot_name}"
                    slot = slots.get(slot_name)
                    if slot is not None and slot.current_impl == impl_name:
                        if slot_config and hasattr(slot.strategy, "configure"):
                            try:
                                slot.strategy.configure(slot_config)
                            except ValueError as exc:
                                # Wave-1 made configure() fail loudly on
                                # bad values; restore must not turn that
                                # into a hard build failure on the
                                # lenient path (review B4) — the strategy
                                # keeps its defaults, the loss is
                                # recorded + logged.
                                self._record_bad_config(outcome, target, impl_name, exc)
                                continue
                        outcome.configured.append(target)
                        continue
                    try:
                        stage.set_strategy(slot_name, impl_name, slot_config)
                        outcome.configured.append(target)
                    except (KeyError, AttributeError):
                        # Same skip semantics as ever — but recorded.
                        # KeyError from a missing slot vs a missing impl
                        # is distinguished via the slots map we already
                        # fetched (set_strategy raises KeyError for both).
                        if slot is None:
                            outcome.skipped_slots.append(target)
                        else:
                            outcome.skipped_impls.append(f"{target}→{impl_name}")
                    except ValueError as exc:
                        # set_strategy → StrategySlot.swap → configure()
                        # rejected the captured config (the other
                        # configure call site). The swap never landed,
                        # so the slot keeps the previous (default)
                        # strategy — record the rejection, don't raise.
                        self._record_bad_config(outcome, target, impl_name, exc)

                # Restore stage config
                if stage_snap.stage_config:
                    stage.update_config(stage_snap.stage_config)
                    outcome.configured.append(f"stage:{stage_snap.order}.config")

                # Restore chain ordering (v2)
                for chain_name, order in (stage_snap.chain_order or {}).items():
                    if not order:
                        continue
                    target = f"stage:{stage_snap.order}.{chain_name}"
                    try:
                        stage.reorder_chain(chain_name, order)
                        outcome.configured.append(target)
                    except (KeyError, ValueError):
                        # Unknown chain or incompatible order — recorded skip.
                        outcome.skipped_chains.append(target)

                # Restore tool binding (v2) — only apply when captured.
                if stage_snap.tool_binding is not None:
                    stage.tool_binding = StageToolBinding.from_dict(stage_snap.tool_binding)
                    outcome.configured.append(f"stage:{stage_snap.order}.tool_binding")

                # Restore model override (v2) — only apply when captured. A None
                # payload from a v1 snapshot doesn't clear the live override; use
                # set_stage_model(order, None) for an explicit reset.
                if stage_snap.model_override is not None:
                    stage.model_override = ModelConfig.from_dict(stage_snap.model_override)
                    outcome.configured.append(f"stage:{stage_snap.order}.model_override")

            # Restore model config
            if snapshot.model_config:
                cfg = self._pipeline._config.model
                for key, value in snapshot.model_config.items():
                    if hasattr(cfg, key):
                        setattr(cfg, key, value)

            # Restore pipeline config
            if snapshot.pipeline_config:
                cfg = self._pipeline._config
                for key, value in snapshot.pipeline_config.items():
                    if key == "model":
                        continue
                    if hasattr(cfg, key):
                        setattr(cfg, key, value)

            record = MutationRecord(
                kind=MutationKind.RESTORE_SNAPSHOT,
                target="pipeline",
                new_value=snapshot.pipeline_name,
            )
            self._change_log.append(record)
            if report:
                return outcome
            return MutationResult(success=True, record=record)

    # ── Change log ──────────────────────────────────────────

    def get_change_log(self) -> List[MutationRecord]:
        """Return a copy of the mutation change log."""
        return list(self._change_log)

    def clear_change_log(self) -> None:
        """Clear the change log."""
        self._change_log.clear()

    # ── Execution lock ──────────────────────────────────────
    #
    # Two layers (2.2.0, audit §3.5 honesty pass):
    #
    # * ``Pipeline.run_in_progress`` — the REAL lock. Wired by the
    #   engine around run()/run_stream() (including the streaming
    #   background task) and consulted by every config/strategy/
    #   structure mutation via ``_check_run_lock``. This is what makes
    #   ``MutationLocked`` actually fire in production.
    # * ``lock_stage`` / ``unlock_stage`` — legacy MANUAL per-stage
    #   flags. The engine has never called them (the audit's "dead
    #   lock" finding); they remain functional for hosts that want
    #   finer-than-run-granularity freezes (e.g. pin stage 6 while an
    #   operator edits other stages), and for back-compat. They are
    #   NOT how the engine protects a running pipeline.

    def lock_stage(self, order: int) -> None:
        """Manually freeze a stage against mutations (host-side flag).

        Legacy/manual API: the engine does NOT call this during
        execution — run-time protection comes from the pipeline's
        ``run_in_progress`` lock, which every mutating operation
        checks automatically. Use this only to hold a stage frozen
        across multiple turns (e.g. while an operator UI edits the
        rest of the environment). Pair with :meth:`unlock_stage`.
        """
        self._locked_stages.add(order)

    def unlock_stage(self, order: int) -> None:
        """Release a manual :meth:`lock_stage` freeze."""
        self._locked_stages.discard(order)

    # ── Internal ────────────────────────────────────────────

    @staticmethod
    def _record_bad_config(
        outcome: RestoreReport, target: str, impl_name: str, exc: ValueError
    ) -> None:
        """Record a configure()-rejected strategy config during restore.

        Why (2.2.0 review B4): wave 1 made ``Strategy.configure`` raise
        ``ValueError`` on bad values, which silently re-broke
        ``from_manifest(strict=False)`` — manifests 2.1.x accepted now
        hard-failed at restore time. Both configure call sites in
        :meth:`restore` funnel here instead: the error lands in
        ``RestoreReport.errors`` (strict builds refuse via
        ``has_skips``; ``validate_manifest`` reports the same bad value
        as ``strategy.config_invalid`` at write time) and a WARNING is
        logged for the legacy ``report=False`` path, where the report
        object is discarded.
        """
        message = f"{target}: configure({impl_name!r}) rejected snapshot config: {exc}"
        outcome.errors.append(message)
        logger.warning(
            "restore: %s — the strategy keeps its defaults; fix the "
            "manifest (validate_manifest reports this as "
            "strategy.config_invalid at write time)",
            message,
        )

    def _get_stage(self, order: int) -> Any:
        """Get a registered stage or raise."""
        stage = self._pipeline.get_stage(order)
        if stage is None:
            raise MutationError(
                f"No stage registered at order {order}",
                stage_order=order,
            )
        return stage

    def _check_run_lock(self) -> None:
        """Raise :class:`MutationLocked` while the pipeline is executing a run.

        2.2.0 (audit §3.5): this is the lock that actually fires. The
        pipeline maintains ``run_in_progress`` around ``run()`` /
        ``run_stream()`` (including the streaming background task), and
        every config/strategy/structure mutation consults it here — a
        mid-run swap would hand later stages different strategies than
        earlier stages already executed with. :meth:`restore` is the
        one deliberate exemption (see its docstring).
        """
        if getattr(self._pipeline, "run_in_progress", False):
            raise MutationLocked(
                "Pipeline is currently executing a run — mutations are "
                "locked until the run finishes. Mutate between turns "
                "(after run() returns / the run_stream iterator drains)."
            )

    def _check_stage_lock(self, order: int) -> None:
        """Raise if mutation is locked — run in progress, or manual stage lock.

        The run-in-progress check (2.2.0) is the engine-wired lock;
        the per-stage set is the legacy manual API (see
        :meth:`lock_stage`).
        """
        self._check_run_lock()
        if order in self._locked_stages:
            raise MutationLocked(
                f"Stage {order} is currently executing and cannot be mutated",
                stage_order=order,
            )
