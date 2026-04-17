"""PipelineMutator — safe runtime mutation of pipeline configuration.

Provides a controlled API for modifying pipeline stages, strategies,
and configuration at runtime while maintaining consistency guarantees.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from geny_executor.core.errors import MutationError, MutationLocked
from geny_executor.core.snapshot import PipelineSnapshot, StageSnapshot

if TYPE_CHECKING:
    from geny_executor.core.pipeline import Pipeline


class MutationKind(str, Enum):
    """Type of mutation performed."""

    SWAP_STRATEGY = "swap_strategy"
    UPDATE_STAGE_CONFIG = "update_stage_config"
    UPDATE_MODEL_CONFIG = "update_model_config"
    UPDATE_PIPELINE_CONFIG = "update_pipeline_config"
    SET_STAGE_ACTIVE = "set_stage_active"
    REGISTER_STAGE = "register_stage"
    REMOVE_STAGE = "remove_stage"
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


class PipelineMutator:
    """Controlled mutation layer over a :class:`Pipeline`.

    All mutations are recorded in a change log and protected by a lock
    so that stages currently executing cannot be mutated concurrently.

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

    # ── Snapshot ────────────────────────────────────────────

    def snapshot(self, description: str = "") -> PipelineSnapshot:
        """Capture the current pipeline configuration state."""
        stages: List[StageSnapshot] = []
        for order in range(1, 17):
            stage = self._pipeline.get_stage(order)
            if stage:
                strategies: Dict[str, str] = {}
                strategy_configs: Dict[str, Dict[str, Any]] = {}
                for info in stage.list_strategies():
                    strategies[info.slot_name] = info.current_impl
                    if info.config:
                        strategy_configs[info.slot_name] = info.config

                stages.append(
                    StageSnapshot(
                        order=order,
                        name=stage.name,
                        is_active=True,
                        strategies=strategies,
                        strategy_configs=strategy_configs,
                        stage_config=stage.get_config(),
                    )
                )
            else:
                stages.append(StageSnapshot(order=order, name=f"stage_{order}", is_active=False))

        # Serialize PipelineConfig
        cfg = self._pipeline._config
        model_dict = {
            "model": cfg.model.model,
            "max_tokens": cfg.model.max_tokens,
            "temperature": cfg.model.temperature,
            "top_p": cfg.model.top_p,
            "top_k": cfg.model.top_k,
            "stop_sequences": cfg.model.stop_sequences,
            "thinking_enabled": cfg.model.thinking_enabled,
            "thinking_budget_tokens": cfg.model.thinking_budget_tokens,
            "thinking_type": cfg.model.thinking_type,
            "thinking_display": cfg.model.thinking_display,
        }
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

    def restore(self, snapshot: PipelineSnapshot) -> MutationResult:
        """Restore pipeline configuration from a snapshot.

        This restores strategy selections and configurations. It does NOT
        replace Stage instances themselves — stages must already be registered.
        """
        with self._lock:
            for stage_snap in snapshot.stages:
                stage = self._pipeline.get_stage(stage_snap.order)
                if stage is None:
                    continue  # Cannot restore an unregistered stage

                # Restore strategy selections
                for slot_name, impl_name in stage_snap.strategies.items():
                    slot_config = stage_snap.strategy_configs.get(slot_name)
                    try:
                        stage.set_strategy(slot_name, impl_name, slot_config)
                    except (KeyError, AttributeError):
                        pass  # Skip unknown slots/impls silently

                # Restore stage config
                if stage_snap.stage_config:
                    stage.update_config(stage_snap.stage_config)

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
            return MutationResult(success=True, record=record)

    # ── Change log ──────────────────────────────────────────

    def get_change_log(self) -> List[MutationRecord]:
        """Return a copy of the mutation change log."""
        return list(self._change_log)

    def clear_change_log(self) -> None:
        """Clear the change log."""
        self._change_log.clear()

    # ── Execution lock ──────────────────────────────────────

    def lock_stage(self, order: int) -> None:
        """Mark a stage as currently executing (blocks mutations)."""
        self._locked_stages.add(order)

    def unlock_stage(self, order: int) -> None:
        """Mark a stage as no longer executing."""
        self._locked_stages.discard(order)

    # ── Internal ────────────────────────────────────────────

    def _get_stage(self, order: int) -> Any:
        """Get a registered stage or raise."""
        stage = self._pipeline.get_stage(order)
        if stage is None:
            raise MutationError(
                f"No stage registered at order {order}",
                stage_order=order,
            )
        return stage

    def _check_stage_lock(self, order: int) -> None:
        """Raise if the stage is locked for execution."""
        if order in self._locked_stages:
            raise MutationLocked(
                f"Stage {order} is currently executing and cannot be mutated",
                stage_order=order,
            )
