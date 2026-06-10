"""2.2.0 review B4 — bad strategy_configs must not hard-fail lenient builds.

Wave 1 made ``Strategy.configure()`` raise ``ValueError`` on bad values
(the §2.1 fail-loudly fix), but ``PipelineMutator.restore`` called both
of its configure sites unguarded — so ``from_manifest(strict=False)``
hard-failed on manifests 2.1.x accepted, defeating the entire point of
the lenient path. Pinned here:

  * lenient build: degrades — the slot keeps its default strategy, the
    rejection is logged as a WARNING (the legacy ``report=False``
    surface) and recorded in ``RestoreReport.errors``;
  * strict build: refuses at write time — ``validate_manifest`` probes
    the config against the impl's own ``configure()`` and reports
    ``strategy.config_invalid`` as an error finding;
  * both configure call sites in restore are covered (same-impl
    reconfigure and ``set_strategy`` swap).
"""

from __future__ import annotations

import logging

import pytest

from geny_executor import ConfigError, build_manifest, validate_manifest
from geny_executor.core.environment import EnvironmentManifest
from geny_executor.core.mutation import PipelineMutator
from geny_executor.core.pipeline import Pipeline


BAD_EVALUATORS = {"strategy": {"evaluators": "not-a-list"}}


def _manifest_with_bad_evaluate_config(*, active: bool = True) -> EnvironmentManifest:
    data = build_manifest("worker_adaptive", provider="anthropic").to_dict()
    for entry in data["stages"]:
        if entry["name"] == "evaluate":
            entry["strategy_configs"] = dict(BAD_EVALUATORS)
            entry["active"] = active
    return EnvironmentManifest.from_dict(data)


def _evaluate_slot(pipeline: Pipeline):
    stage = next(s for s in pipeline.stages if s.name == "evaluate")
    return stage.get_strategy_slots()["strategy"]


# ── Lenient builds degrade instead of raising ────────────────


def test_lenient_build_survives_bad_strategy_config(caplog):
    manifest = _manifest_with_bad_evaluate_config()
    with caplog.at_level(logging.WARNING, logger="geny_executor.core.mutation"):
        pipeline = Pipeline.from_manifest(manifest, api_key="sk-test", strict=False)

    # The swap never landed — the slot keeps its default strategy.
    slot = _evaluate_slot(pipeline)
    assert slot.current_impl == "signal_based"
    assert "rejected snapshot config" in caplog.text
    assert "evaluators" in caplog.text


def test_lenient_reconfigure_of_current_impl_also_degrades(caplog):
    """The OTHER configure call site: the live slot already holds the
    declared impl, so restore re-applies only the config."""
    manifest = _manifest_with_bad_evaluate_config()
    pipeline = Pipeline.from_manifest(
        build_manifest("worker_adaptive", provider="anthropic"),
        api_key="sk-test",
        strict=True,
    )
    # The strict build above left the slot on evaluation_chain with a
    # good config; replaying the bad snapshot hits the same-impl branch.
    assert _evaluate_slot(pipeline).current_impl == "evaluation_chain"
    good_config = dict(_evaluate_slot(pipeline).strategy.get_config())

    with caplog.at_level(logging.WARNING, logger="geny_executor.core.mutation"):
        report = PipelineMutator(pipeline).restore(manifest.to_snapshot(), report=True)

    assert any("rejected snapshot config" in e for e in report.errors)
    # Bad value did not disturb the previously-good configuration.
    assert _evaluate_slot(pipeline).strategy.get_config() == good_config
    assert "rejected snapshot config" in caplog.text


def test_restore_report_records_configure_rejection():
    manifest = _manifest_with_bad_evaluate_config()
    pipeline = Pipeline.from_manifest(manifest, api_key="sk-test", strict=False)
    report = PipelineMutator(pipeline).restore(manifest.to_snapshot(), report=True)
    assert any(
        "stage:14.strategy" in e and "rejected snapshot config" in e
        for e in report.errors
    )
    assert report.has_skips


def test_legacy_report_false_path_returns_success_and_logs(caplog):
    manifest = _manifest_with_bad_evaluate_config()
    pipeline = Pipeline.from_manifest(manifest, api_key="sk-test", strict=False)
    with caplog.at_level(logging.WARNING, logger="geny_executor.core.mutation"):
        result = PipelineMutator(pipeline).restore(manifest.to_snapshot())
    assert result.success is True  # legacy surface unchanged
    assert "rejected snapshot config" in caplog.text


# ── Strict builds refuse at write time ───────────────────────


def test_validate_manifest_flags_bad_config_as_error():
    issues = validate_manifest(_manifest_with_bad_evaluate_config())
    found = [i for i in issues if i.code == "strategy.config_invalid"]
    assert len(found) == 1
    assert found[0].severity == "error"
    assert found[0].stage_order == 14
    assert found[0].field == "strategy_configs.strategy"
    assert "not-a-list" in found[0].message


def test_validate_manifest_inactive_entry_downgrades_to_warning():
    issues = validate_manifest(_manifest_with_bad_evaluate_config(active=False))
    found = [i for i in issues if i.code == "strategy.config_invalid"]
    assert len(found) == 1
    assert found[0].severity == "warning"


def test_strict_build_refuses_via_validate_manifest():
    with pytest.raises(ConfigError, match="strategy.config_invalid"):
        Pipeline.from_manifest(
            _manifest_with_bad_evaluate_config(), api_key="sk-test", strict=True
        )


def test_valid_strategy_config_probes_clean():
    """The probe must not false-positive on the stock preset's real
    evaluator config (the Geny prod shape)."""
    issues = validate_manifest(build_manifest("worker_adaptive", provider="anthropic"))
    assert not [i for i in issues if i.code == "strategy.config_invalid"]
