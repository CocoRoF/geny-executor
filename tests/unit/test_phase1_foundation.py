"""Phase 1 Foundation tests — ConfigSchema, StrategySlot, PipelineMutator, Snapshot."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.core.slot import StrategySlot
from geny_executor.core.snapshot import PipelineSnapshot, StageSnapshot
from geny_executor.core.mutation import (
    PipelineMutator,
    MutationKind,
)
from geny_executor.core.errors import MutationError, MutationLocked
from geny_executor.core.stage import Stage, Strategy, StrategyInfo
from geny_executor import Pipeline, PipelineConfig


# ══════════════════════════════════════════════════════════
# Fixtures — mock strategies and stages
# ══════════════════════════════════════════════════════════


class AlphaStrategy(Strategy):
    @property
    def name(self) -> str:
        return "alpha"

    @property
    def description(self) -> str:
        return "Alpha strategy"

    def get_config(self):
        return {"mode": "fast"}


class BetaStrategy(Strategy):
    def __init__(self):
        self._threshold = 0.5

    @property
    def name(self) -> str:
        return "beta"

    def configure(self, config):
        if "threshold" in config:
            self._threshold = config["threshold"]

    def get_config(self):
        return {"threshold": self._threshold}

    @classmethod
    def config_schema(cls):
        return ConfigSchema(
            name="beta",
            fields=[
                ConfigField(
                    name="threshold",
                    type="number",
                    label="Threshold",
                    default=0.5,
                    min_value=0.0,
                    max_value=1.0,
                )
            ],
        )


class SlotStage(Stage):
    """A stage that uses StrategySlot for runtime mutation."""

    def __init__(self):
        self._slots = {
            "primary": StrategySlot(
                name="primary",
                strategy=AlphaStrategy(),
                registry={"alpha": AlphaStrategy, "beta": BetaStrategy},
                description="Primary strategy slot",
            )
        }

    @property
    def name(self) -> str:
        return "slot_stage"

    @property
    def order(self) -> int:
        return 2

    async def execute(self, input, state):
        return input

    def get_strategy_slots(self):
        return self._slots


class BareStage(Stage):
    """Legacy stage with no slots."""

    @property
    def name(self) -> str:
        return "bare_stage"

    @property
    def order(self) -> int:
        return 3

    async def execute(self, input, state):
        return input


class ConfigurableStage(Stage):
    """Stage with stage-level configuration."""

    def __init__(self):
        self._verbosity = 1

    @property
    def name(self) -> str:
        return "configurable"

    @property
    def order(self) -> int:
        return 4

    async def execute(self, input, state):
        return input

    def get_config(self):
        return {"verbosity": self._verbosity}

    def update_config(self, config):
        if "verbosity" in config:
            self._verbosity = config["verbosity"]

    def get_config_schema(self):
        return ConfigSchema(
            name="configurable",
            fields=[
                ConfigField(
                    name="verbosity",
                    type="integer",
                    label="Verbosity",
                    default=1,
                    min_value=0,
                    max_value=3,
                )
            ],
        )


# ══════════════════════════════════════════════════════════
# ConfigSchema
# ══════════════════════════════════════════════════════════


class TestConfigField:
    def test_to_json_schema_number(self):
        f = ConfigField(
            name="temp",
            type="number",
            label="Temperature",
            default=0.7,
            min_value=0.0,
            max_value=2.0,
        )
        js = f.to_json_schema()
        assert js["type"] == "number"
        assert js["title"] == "Temperature"
        assert js["default"] == 0.7
        assert js["minimum"] == 0.0
        assert js["maximum"] == 2.0

    def test_to_json_schema_select(self):
        f = ConfigField(
            name="mode",
            type="select",
            label="Mode",
            options=[
                {"value": "fast", "label": "Fast"},
                {"value": "slow", "label": "Slow"},
            ],
        )
        js = f.to_json_schema()
        assert js["type"] == "string"
        assert js["enum"] == ["fast", "slow"]

    def test_to_json_schema_string_constraints(self):
        f = ConfigField(
            name="pattern",
            type="string",
            label="Pattern",
            min_length=1,
            max_length=100,
            pattern="^[a-z]+$",
        )
        js = f.to_json_schema()
        assert js["minLength"] == 1
        assert js["maxLength"] == 100
        assert js["pattern"] == "^[a-z]+$"

    def test_to_json_schema_ui_extensions(self):
        f = ConfigField(
            name="prompt",
            type="string",
            label="Prompt",
            ui_widget="textarea",
            ui_group="advanced",
            ui_order=5,
        )
        js = f.to_json_schema()
        assert js["x-ui-widget"] == "textarea"
        assert js["x-ui-group"] == "advanced"
        assert js["x-ui-order"] == 5


class TestConfigSchema:
    def _make_schema(self):
        return ConfigSchema(
            name="test_schema",
            fields=[
                ConfigField(name="name", type="string", label="Name", required=True, min_length=1),
                ConfigField(
                    name="count",
                    type="integer",
                    label="Count",
                    default=10,
                    min_value=1,
                    max_value=100,
                ),
                ConfigField(name="enabled", type="boolean", label="Enabled", default=True),
                ConfigField(
                    name="mode",
                    type="select",
                    label="Mode",
                    options=[{"value": "a", "label": "A"}, {"value": "b", "label": "B"}],
                ),
            ],
        )

    def test_to_json_schema(self):
        schema = self._make_schema()
        js = schema.to_json_schema()
        assert js["type"] == "object"
        assert js["title"] == "test_schema"
        assert "name" in js["properties"]
        assert "count" in js["properties"]
        assert js["required"] == ["name"]

    def test_validate_ok(self):
        schema = self._make_schema()
        errors = schema.validate({"name": "hello", "count": 5, "enabled": True, "mode": "a"})
        assert errors == []

    def test_validate_missing_required(self):
        schema = self._make_schema()
        errors = schema.validate({"count": 5})
        assert any("Required" in e and "name" in e for e in errors)

    def test_validate_type_error(self):
        schema = self._make_schema()
        errors = schema.validate({"name": "ok", "count": "not_a_number"})
        assert any("expected integer" in e for e in errors)

    def test_validate_range_error(self):
        schema = self._make_schema()
        errors = schema.validate({"name": "ok", "count": 200})
        assert any("maximum" in e for e in errors)

    def test_validate_enum_error(self):
        schema = self._make_schema()
        errors = schema.validate({"name": "ok", "mode": "z"})
        assert any("not in" in e for e in errors)

    def test_validate_string_length_error(self):
        schema = self._make_schema()
        errors = schema.validate({"name": ""})
        assert any("minimum" in e for e in errors)

    def test_apply_defaults(self):
        schema = self._make_schema()
        result = schema.apply_defaults({"name": "test"})
        assert result["name"] == "test"
        assert result["count"] == 10
        assert result["enabled"] is True

    def test_apply_defaults_no_overwrite(self):
        schema = self._make_schema()
        result = schema.apply_defaults({"name": "test", "count": 42})
        assert result["count"] == 42


# ══════════════════════════════════════════════════════════
# StrategySlot
# ══════════════════════════════════════════════════════════


class TestStrategySlot:
    def _make_slot(self):
        return StrategySlot(
            name="primary",
            strategy=AlphaStrategy(),
            registry={"alpha": AlphaStrategy, "beta": BetaStrategy},
        )

    def test_current_impl(self):
        slot = self._make_slot()
        assert slot.current_impl == "alpha"

    def test_available_impls(self):
        slot = self._make_slot()
        assert slot.available_impls == ["alpha", "beta"]

    def test_swap(self):
        slot = self._make_slot()
        new = slot.swap("beta")
        assert new.name == "beta"
        assert slot.current_impl == "beta"

    def test_swap_with_config(self):
        slot = self._make_slot()
        new = slot.swap("beta", config={"threshold": 0.8})
        assert new.get_config()["threshold"] == 0.8

    def test_swap_unknown_raises(self):
        slot = self._make_slot()
        with pytest.raises(KeyError, match="not found"):
            slot.swap("gamma")

    def test_describe(self):
        slot = self._make_slot()
        info = slot.describe()
        assert isinstance(info, StrategyInfo)
        assert info.slot_name == "primary"
        assert info.current_impl == "alpha"
        assert "alpha" in info.available_impls
        assert "beta" in info.available_impls


# ══════════════════════════════════════════════════════════
# Stage/Strategy extensions
# ══════════════════════════════════════════════════════════


class TestStrategyExtensions:
    def test_config_schema_default_none(self):
        assert AlphaStrategy.config_schema() is None

    def test_config_schema_override(self):
        schema = BetaStrategy.config_schema()
        assert schema is not None
        assert schema.name == "beta"

    def test_from_config(self):
        instance = BetaStrategy.from_config({"threshold": 0.9})
        assert instance.get_config()["threshold"] == 0.9

    def test_get_config_default_empty(self):
        class Minimal(Strategy):
            @property
            def name(self):
                return "minimal"

        assert Minimal().get_config() == {}


class TestStageExtensions:
    def test_slot_stage_list_strategies_auto(self):
        """list_strategies() auto-generates from get_strategy_slots()."""
        stage = SlotStage()
        infos = stage.list_strategies()
        assert len(infos) == 1
        assert infos[0].slot_name == "primary"
        assert infos[0].current_impl == "alpha"

    def test_bare_stage_list_strategies_empty(self):
        stage = BareStage()
        assert stage.list_strategies() == []

    def test_set_strategy(self):
        stage = SlotStage()
        stage.set_strategy("primary", "beta", {"threshold": 0.3})
        infos = stage.list_strategies()
        assert infos[0].current_impl == "beta"

    def test_set_strategy_unknown_slot_raises(self):
        stage = SlotStage()
        with pytest.raises(KeyError, match="no strategy slot"):
            stage.set_strategy("nonexistent", "alpha")

    def test_get_config_schema_default_none(self):
        stage = BareStage()
        assert stage.get_config_schema() is None

    def test_configurable_stage_roundtrip(self):
        stage = ConfigurableStage()
        assert stage.get_config()["verbosity"] == 1
        stage.update_config({"verbosity": 3})
        assert stage.get_config()["verbosity"] == 3

    def test_describe_includes_slots(self):
        stage = SlotStage()
        desc = stage.describe()
        assert desc.name == "slot_stage"
        assert len(desc.strategies) == 1


# ══════════════════════════════════════════════════════════
# PipelineSnapshot
# ══════════════════════════════════════════════════════════


class TestPipelineSnapshot:
    def _make_snapshot(self):
        return PipelineSnapshot(
            pipeline_name="test-pipe",
            stages=[
                StageSnapshot(order=1, name="input", is_active=True),
                StageSnapshot(
                    order=2,
                    name="context",
                    is_active=True,
                    strategies={"primary": "alpha"},
                    strategy_configs={"primary": {"mode": "fast"}},
                    stage_config={"verbosity": 2},
                ),
                StageSnapshot(order=3, name="system", is_active=False),
            ],
            pipeline_config={"max_iterations": 25, "stream": True},
            model_config={"model": "claude-sonnet-4-20250514", "temperature": 0.5},
            description="Test snapshot",
        )

    def test_to_dict(self):
        snap = self._make_snapshot()
        d = snap.to_dict()
        assert d["pipeline_name"] == "test-pipe"
        assert len(d["stages"]) == 3
        assert d["stages"][1]["strategies"]["primary"] == "alpha"

    def test_json_roundtrip(self):
        snap = self._make_snapshot()
        json_str = snap.to_json()
        restored = PipelineSnapshot.from_json(json_str)
        assert restored.pipeline_name == snap.pipeline_name
        assert len(restored.stages) == len(snap.stages)
        assert restored.stages[1].strategies["primary"] == "alpha"
        assert restored.model_config["temperature"] == 0.5
        assert restored.description == "Test snapshot"

    def test_from_dict(self):
        snap = self._make_snapshot()
        d = snap.to_dict()
        restored = PipelineSnapshot.from_dict(d)
        assert restored.pipeline_name == "test-pipe"
        assert restored.stages[2].is_active is False


# ══════════════════════════════════════════════════════════
# PipelineMutator
# ══════════════════════════════════════════════════════════


def _make_pipeline_with_stages():
    """Create a pipeline with SlotStage, BareStage, ConfigurableStage."""
    config = PipelineConfig(name="mut-test")
    pipeline = Pipeline(config)
    pipeline.register_stage(SlotStage())  # order 2
    pipeline.register_stage(BareStage())  # order 3
    pipeline.register_stage(ConfigurableStage())  # order 4
    return pipeline


class TestPipelineMutator:
    def test_swap_strategy(self):
        pipeline = _make_pipeline_with_stages()
        mutator = PipelineMutator(pipeline)
        result = mutator.swap_strategy(2, "primary", "beta")
        assert result.success
        assert result.record.kind == MutationKind.SWAP_STRATEGY
        assert result.record.new_value == "beta"
        # Verify the stage actually changed
        stage = pipeline.get_stage(2)
        infos = stage.list_strategies()
        assert infos[0].current_impl == "beta"

    def test_swap_strategy_with_config(self):
        pipeline = _make_pipeline_with_stages()
        mutator = PipelineMutator(pipeline)
        mutator.swap_strategy(2, "primary", "beta", config={"threshold": 0.9})
        stage = pipeline.get_stage(2)
        infos = stage.list_strategies()
        assert infos[0].config.get("threshold") == 0.9

    def test_swap_strategy_unknown_impl(self):
        pipeline = _make_pipeline_with_stages()
        mutator = PipelineMutator(pipeline)
        with pytest.raises(MutationError):
            mutator.swap_strategy(2, "primary", "gamma")

    def test_swap_strategy_unknown_stage(self):
        pipeline = _make_pipeline_with_stages()
        mutator = PipelineMutator(pipeline)
        with pytest.raises(MutationError, match="No stage"):
            mutator.swap_strategy(99, "primary", "alpha")

    def test_update_stage_config(self):
        pipeline = _make_pipeline_with_stages()
        mutator = PipelineMutator(pipeline)
        result = mutator.update_stage_config(4, {"verbosity": 3})
        assert result.success
        stage = pipeline.get_stage(4)
        assert stage.get_config()["verbosity"] == 3

    def test_update_model_config(self):
        pipeline = _make_pipeline_with_stages()
        mutator = PipelineMutator(pipeline)
        result = mutator.update_model_config({"temperature": 0.9, "max_tokens": 4096})
        assert result.success
        assert pipeline._config.model.temperature == 0.9
        assert pipeline._config.model.max_tokens == 4096

    def test_update_model_config_bad_field(self):
        pipeline = _make_pipeline_with_stages()
        mutator = PipelineMutator(pipeline)
        with pytest.raises(MutationError, match="no field"):
            mutator.update_model_config({"nonexistent": 42})

    def test_update_pipeline_config(self):
        pipeline = _make_pipeline_with_stages()
        mutator = PipelineMutator(pipeline)
        result = mutator.update_pipeline_config({"max_iterations": 25, "stream": False})
        assert result.success
        assert pipeline._config.max_iterations == 25
        assert pipeline._config.stream is False

    def test_update_pipeline_config_ignores_model(self):
        pipeline = _make_pipeline_with_stages()
        mutator = PipelineMutator(pipeline)
        old_model = pipeline._config.model.model
        mutator.update_pipeline_config({"model": "should_be_ignored"})
        assert pipeline._config.model.model == old_model

    def test_set_stage_active_disable(self):
        pipeline = _make_pipeline_with_stages()
        mutator = PipelineMutator(pipeline)
        assert pipeline.get_stage(3) is not None
        mutator.set_stage_active(3, False)
        assert pipeline.get_stage(3) is None

    def test_set_stage_active_reenable(self):
        pipeline = _make_pipeline_with_stages()
        mutator = PipelineMutator(pipeline)
        mutator.set_stage_active(3, False)
        assert pipeline.get_stage(3) is None
        mutator.set_stage_active(3, True)
        assert pipeline.get_stage(3) is not None
        assert pipeline.get_stage(3).name == "bare_stage"

    def test_set_stage_active_unregistered_raises(self):
        pipeline = _make_pipeline_with_stages()
        mutator = PipelineMutator(pipeline)
        with pytest.raises(MutationError, match="not registered"):
            mutator.set_stage_active(99, True)

    def test_change_log(self):
        pipeline = _make_pipeline_with_stages()
        mutator = PipelineMutator(pipeline)
        mutator.swap_strategy(2, "primary", "beta")
        mutator.update_model_config({"temperature": 0.5})
        log = mutator.get_change_log()
        assert len(log) == 2
        assert log[0].kind == MutationKind.SWAP_STRATEGY
        assert log[1].kind == MutationKind.UPDATE_MODEL_CONFIG

    def test_clear_change_log(self):
        pipeline = _make_pipeline_with_stages()
        mutator = PipelineMutator(pipeline)
        mutator.update_model_config({"temperature": 0.5})
        assert len(mutator.get_change_log()) == 1
        mutator.clear_change_log()
        assert len(mutator.get_change_log()) == 0

    def test_lock_blocks_mutation(self):
        pipeline = _make_pipeline_with_stages()
        mutator = PipelineMutator(pipeline)
        mutator.lock_stage(2)
        with pytest.raises(MutationLocked, match="currently executing"):
            mutator.swap_strategy(2, "primary", "beta")

    def test_unlock_allows_mutation(self):
        pipeline = _make_pipeline_with_stages()
        mutator = PipelineMutator(pipeline)
        mutator.lock_stage(2)
        mutator.unlock_stage(2)
        result = mutator.swap_strategy(2, "primary", "beta")
        assert result.success

    def test_snapshot(self):
        pipeline = _make_pipeline_with_stages()
        mutator = PipelineMutator(pipeline)
        snap = mutator.snapshot("test snapshot")
        assert snap.pipeline_name == "mut-test"
        assert snap.description == "test snapshot"
        assert len(snap.stages) == 21  # full 1-21 range (S9a.3)
        # Stage 2 should have strategy info
        s2 = next(s for s in snap.stages if s.order == 2)
        assert s2.is_active is True
        assert s2.strategies.get("primary") == "alpha"

    def test_snapshot_restore_roundtrip(self):
        pipeline = _make_pipeline_with_stages()
        mutator = PipelineMutator(pipeline)

        # Take snapshot with alpha
        snap = mutator.snapshot("before")

        # Mutate to beta
        mutator.swap_strategy(2, "primary", "beta")
        stage2 = pipeline.get_stage(2)
        assert stage2.list_strategies()[0].current_impl == "beta"

        # Restore
        result = mutator.restore(snap)
        assert result.success
        assert stage2.list_strategies()[0].current_impl == "alpha"

    def test_snapshot_restore_model_config(self):
        pipeline = _make_pipeline_with_stages()
        mutator = PipelineMutator(pipeline)

        snap = mutator.snapshot()
        original_temp = pipeline._config.model.temperature

        mutator.update_model_config({"temperature": 1.5})
        assert pipeline._config.model.temperature == 1.5

        mutator.restore(snap)
        assert pipeline._config.model.temperature == original_temp

    def test_snapshot_json_roundtrip(self):
        pipeline = _make_pipeline_with_stages()
        mutator = PipelineMutator(pipeline)
        snap = mutator.snapshot("v1")
        json_str = snap.to_json()
        restored = PipelineSnapshot.from_json(json_str)
        assert restored.pipeline_name == snap.pipeline_name
        assert len(restored.stages) == 21


# ══════════════════════════════════════════════════════════
# Error hierarchy
# ══════════════════════════════════════════════════════════


class TestMutationErrors:
    def test_mutation_error_fields(self):
        err = MutationError("bad", stage_order=6, slot_name="provider")
        assert err.stage_order == 6
        assert err.slot_name == "provider"
        assert "bad" in str(err)

    def test_mutation_locked_fields(self):
        err = MutationLocked("locked", stage_order=3)
        assert err.stage_order == 3
        assert "locked" in str(err)

    def test_inheritance(self):
        from geny_executor.core.errors import GenyExecutorError

        assert issubclass(MutationError, GenyExecutorError)
        assert issubclass(MutationLocked, GenyExecutorError)
