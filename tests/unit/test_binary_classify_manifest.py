"""Manifest-level resolution of the adaptive `binary_classify` evaluation
strategy (v0.25.0).

Before this release, the default Stage 12 (`EvaluateStage`) only registered
`signal_based`, `criteria_based`, and `agent_evaluation` in its strategy
slot. The `binary_classify` strategy lived only in the `adaptive` artifact
module and could only be injected via the builder's `.with_evaluate(
strategy=BinaryClassifyEvaluation(...))` path. That meant a manifest with
`strategies={"strategy": "binary_classify"}` silently fell back to
`signal_based` on restore — useless for manifest-first hosts like Geny
that want to serialize the adaptive preset.

v0.25.0 adds `binary_classify` to the default stage's registry and gives
`BinaryClassifyEvaluation.configure(...)` a working body so
`strategy_configs` flows through on restore.
"""

from __future__ import annotations

from geny_executor.core.environment import (
    EnvironmentManifest,
    EnvironmentMetadata,
    StageManifestEntry,
    ToolsSnapshot,
)
from geny_executor.core.pipeline import Pipeline
from geny_executor.stages.s12_evaluate.artifact.adaptive.strategy import (
    BinaryClassifyConfig,
    BinaryClassifyEvaluation,
)


def _manifest_with_binary_classify(*, strategy_config: dict | None = None) -> EnvironmentManifest:
    entries = [
        StageManifestEntry(order=1, name="input"),
        StageManifestEntry(
            order=12,
            name="evaluate",
            strategies={"strategy": "binary_classify", "scorer": "no_scorer"},
            strategy_configs=({"strategy": strategy_config} if strategy_config is not None else {}),
        ),
    ]
    return EnvironmentManifest(
        metadata=EnvironmentMetadata(
            id="",
            name="binary-classify-test",
            description="",
            base_preset="worker_adaptive",
        ),
        model={},
        pipeline={},
        stages=[e.to_dict() for e in entries],
        tools=ToolsSnapshot(built_in=[], external=[]),
    )


def test_binary_classify_resolves_from_manifest():
    manifest = _manifest_with_binary_classify()
    pipeline = Pipeline.from_manifest(manifest, api_key="sk-test", strict=False)

    stage12 = pipeline.get_stage(12)
    strategy = stage12.get_strategy_slots()["strategy"].strategy

    assert isinstance(strategy, BinaryClassifyEvaluation)
    assert strategy.name == "binary_classify"


def test_binary_classify_configure_applies_strategy_config():
    manifest = _manifest_with_binary_classify(
        strategy_config={"easy_max_turns": 1, "not_easy_max_turns": 30},
    )
    pipeline = Pipeline.from_manifest(manifest, api_key="sk-test", strict=False)

    stage12 = pipeline.get_stage(12)
    strategy = stage12.get_strategy_slots()["strategy"].strategy

    assert strategy._config.easy_max_turns == 1
    assert strategy._config.not_easy_max_turns == 30


def test_binary_classify_defaults_when_no_config():
    manifest = _manifest_with_binary_classify()
    pipeline = Pipeline.from_manifest(manifest, api_key="sk-test", strict=False)

    stage12 = pipeline.get_stage(12)
    strategy = stage12.get_strategy_slots()["strategy"].strategy

    defaults = BinaryClassifyConfig()
    assert strategy._config.easy_max_turns == defaults.easy_max_turns
    assert strategy._config.not_easy_max_turns == defaults.not_easy_max_turns


def test_binary_classify_configure_ignores_unknown_keys():
    strategy = BinaryClassifyEvaluation()
    strategy.configure({"easy_max_turns": 5, "some_future_key": 999})

    assert strategy._config.easy_max_turns == 5
    assert strategy._config.not_easy_max_turns == 30  # unchanged default


def test_binary_classify_configure_empty_dict_is_noop():
    strategy = BinaryClassifyEvaluation(
        BinaryClassifyConfig(easy_max_turns=7, not_easy_max_turns=77)
    )
    strategy.configure({})

    assert strategy._config.easy_max_turns == 7
    assert strategy._config.not_easy_max_turns == 77


def test_default_evaluate_registry_still_has_other_strategies():
    """Adding binary_classify to the default registry must not displace the
    three pre-existing strategies. Manifest-first consumers rely on every
    strategy being spellable by name."""
    from geny_executor.stages.s12_evaluate.artifact.default.stage import EvaluateStage

    stage = EvaluateStage()
    available = stage.get_strategy_slots()["strategy"].available_impls
    assert set(available) == {
        "signal_based",
        "criteria_based",
        "agent_evaluation",
        "binary_classify",
    }
