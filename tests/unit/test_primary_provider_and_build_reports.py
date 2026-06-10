"""2.2.0 small lifecycle items (audit §2.8 / §1-1 / §2.1):

* ``_init_state`` writes the resolved Stage 6 provider into
  ``state.shared[SharedKeys.PRIMARY_PROVIDER]`` — the sub-agent
  inheritance contract finally gets its producer;
* lenient ``from_manifest`` records construction-failed stages on
  ``pipeline.dropped_stages`` (and warns) instead of a bare continue;
* ``PipelineMutator.restore(snapshot, report=True)`` returns a
  ``RestoreReport`` instead of silently passing over skips.
"""

from __future__ import annotations

import logging

import pytest

from geny_executor import Pipeline, PipelineConfig, PipelineMutator, PipelineState
from geny_executor.core.environment import (
    EnvironmentManifest,
    EnvironmentMetadata,
    StageManifestEntry,
    ToolsSnapshot,
)
from geny_executor.core.mutation import RestoreReport
from geny_executor.core.shared_keys import SharedKeys
from geny_executor.core.snapshot import PipelineSnapshot, StageSnapshot
from geny_executor.llm_client.credentials import ConfigError, CredentialBundle
from geny_executor.stages.s01_input import InputStage
from geny_executor.stages.s06_api import APIStage, MockProvider
from geny_executor.stages.s09_parse import ParseStage
from geny_executor.stages.s21_yield import YieldStage


def _make_pipeline() -> Pipeline:
    pipeline = Pipeline(PipelineConfig(name="small-items"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(provider=MockProvider(default_text="ok")))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())
    return pipeline


def _manifest(entries) -> EnvironmentManifest:
    m = EnvironmentManifest(
        metadata=EnvironmentMetadata(id="env_small", name="small"),
        model={},
        pipeline={},
        stages=[],
        tools=ToolsSnapshot(),
    )
    m.set_stage_entries(entries)
    return m


# ── PRIMARY_PROVIDER producer ────────────────────────────────────────


def test_shared_keys_primary_provider_is_legacy_bare_string():
    """Host factories already read the bare key — the constant must
    match what shipped, not gain an ``executor.`` prefix."""
    assert SharedKeys.PRIMARY_PROVIDER == "primary_provider"


def test_init_state_writes_primary_provider_from_stage6():
    pipeline = _make_pipeline()
    state = pipeline._init_state(None)
    # MockProvider-backed APIStage reports its provider name ("mock").
    assert state.shared[SharedKeys.PRIMARY_PROVIDER] == "mock"


def test_init_state_primary_provider_prefers_live_client():
    """The live client's provider attribute is ground truth — it covers
    attached/override clients that contradict the stage declaration."""
    pipeline = _make_pipeline()

    class _Client:
        provider = "anthropic"

    state = PipelineState(session_id="s")
    state.llm_client = _Client()
    out = pipeline._init_state(state)
    assert out.shared[SharedKeys.PRIMARY_PROVIDER] == "anthropic"


def test_init_state_no_api_stage_writes_nothing():
    pipeline = Pipeline(PipelineConfig(name="no-api"))
    pipeline.register_stage(InputStage())
    state = pipeline._init_state(None)
    assert SharedKeys.PRIMARY_PROVIDER not in state.shared


@pytest.mark.asyncio
async def test_primary_provider_refreshed_every_run():
    """Written at every run start, so a between-turn provider change is
    visible to the next turn's sub-agent factories."""
    pipeline = _make_pipeline()
    state = PipelineState(session_id="s")
    await pipeline.run("turn", state)
    assert state.shared[SharedKeys.PRIMARY_PROVIDER] == "mock"


# ── dropped_stages (lenient from_manifest) ───────────────────────────


def test_lenient_from_manifest_records_dropped_stage(caplog):
    entries = [
        StageManifestEntry(order=1, name="input", active=True, artifact="default"),
        # Unknown artifact → create_stage raises → lenient drop.
        StageManifestEntry(
            order=6, name="api", active=True, artifact="does_not_exist",
            config={"provider": "anthropic"},
        ),
        StageManifestEntry(order=21, name="yield", active=True, artifact="default"),
    ]
    with caplog.at_level(logging.WARNING, logger="geny_executor.core.pipeline"):
        pipeline = Pipeline.from_manifest(
            _manifest(entries), credentials=CredentialBundle(), strict=False
        )

    assert pipeline.get_stage(6) is None
    assert len(pipeline.dropped_stages) == 1
    dropped = pipeline.dropped_stages[0]
    assert dropped.name == "api"
    assert dropped.order == 6
    assert dropped.error  # ExcType: message summary present
    assert any("DROPPED" in r.message for r in caplog.records)


def test_strict_from_manifest_still_raises_and_records_nothing():
    entries = [
        StageManifestEntry(
            order=6, name="api", active=True, artifact="does_not_exist",
            config={"provider": "anthropic"},
        ),
    ]
    with pytest.raises(Exception):
        Pipeline.from_manifest(
            _manifest(entries), credentials=CredentialBundle(), strict=True
        )


def test_clean_build_has_empty_dropped_stages():
    entries = [
        StageManifestEntry(order=1, name="input", active=True, artifact="default"),
        StageManifestEntry(
            order=6, name="api", active=True, artifact="default",
            config={"provider": "anthropic"},
        ),
    ]
    pipeline = Pipeline.from_manifest(
        _manifest(entries), credentials=CredentialBundle(), strict=False
    )
    assert pipeline.dropped_stages == []


# ── RestoreReport ────────────────────────────────────────────────────


def test_restore_default_return_stays_mutation_result():
    pipeline = _make_pipeline()
    mutator = PipelineMutator(pipeline)
    snapshot = mutator.snapshot()
    result = mutator.restore(snapshot)
    assert result.success is True  # MutationResult, unchanged contract


def test_restore_report_true_returns_restore_report():
    pipeline = _make_pipeline()
    mutator = PipelineMutator(pipeline)
    snapshot = mutator.snapshot()
    outcome = mutator.restore(snapshot, report=True)
    assert isinstance(outcome, RestoreReport)
    assert outcome.has_skips is False
    assert outcome.configured  # a healthy restore applied things


def test_restore_report_records_skipped_slot_and_impl():
    pipeline = _make_pipeline()
    mutator = PipelineMutator(pipeline)

    snapshot = PipelineSnapshot(
        pipeline_name="drifted",
        stages=[
            StageSnapshot(
                order=6,
                name="api",
                is_active=True,
                strategies={
                    "no_such_slot": "whatever",  # slot the stage lacks
                    "retry": "no_such_impl",  # slot exists, impl doesn't
                },
            ),
            # Active in snapshot, not registered on the pipeline.
            StageSnapshot(order=14, name="evaluate", is_active=True),
        ],
        pipeline_config={},
        model_config={},
    )

    outcome = mutator.restore(snapshot, report=True)

    assert isinstance(outcome, RestoreReport)
    assert outcome.has_skips is True
    assert "stage:6.no_such_slot" in outcome.skipped_slots
    assert "stage:6.retry→no_such_impl" in outcome.skipped_impls
    assert any("stage:14" in err for err in outcome.errors)


def test_restore_report_records_skipped_chain():
    pipeline = _make_pipeline()
    mutator = PipelineMutator(pipeline)
    snapshot = PipelineSnapshot(
        pipeline_name="chains",
        stages=[
            StageSnapshot(
                order=6,
                name="api",
                is_active=True,
                chain_order={"no_such_chain": ["a", "b"]},
            ),
        ],
        pipeline_config={},
        model_config={},
    )
    outcome = mutator.restore(snapshot, report=True)
    assert "stage:6.no_such_chain" in outcome.skipped_chains


def test_strict_from_manifest_warns_on_restore_skips(caplog):
    """A strict load whose restore dropped declarations must say so —
    that silent drop is how Geny prod lost its evaluator config (§2.1).

    Since the write-time ``validate_manifest`` gate landed (2.2.0 Wave
    2), unknown-slot declarations like the original ``ghost_slot``
    fixture are refused *before* restore ever runs. The restore-skip
    warning remains the backstop for declarations validation accepts
    but restore cannot apply — chain orderings over the default-empty
    guard chain are the canonical (warning-class) case: restore can
    only reorder existing items, never populate.
    """
    entries = [
        StageManifestEntry(order=1, name="input", active=True, artifact="default"),
        StageManifestEntry(
            order=4, name="guard", active=True, artifact="default",
            chain_order={"guards": ["token_budget", "cost_budget"]},
        ),
        StageManifestEntry(
            order=6, name="api", active=True, artifact="default",
            strategies={"retry": "exponential_backoff"},
            config={"provider": "anthropic"},
        ),
        StageManifestEntry(order=9, name="parse", active=True, artifact="default"),
        StageManifestEntry(order=21, name="yield", active=True, artifact="default"),
    ]
    with caplog.at_level(logging.WARNING, logger="geny_executor.core.pipeline"):
        Pipeline.from_manifest(
            _manifest(entries), credentials=CredentialBundle(), strict=True
        )
    assert any("restore skipped" in r.message for r in caplog.records)


def test_strict_from_manifest_rejects_unknown_slot(caplog):
    """The original §2.1 fixture (a ``ghost_slot`` strategy on an active
    stage) is now refused at build time by the write-time validation
    gate — promoted from the restore-skip WARNING to a hard error."""
    entries = [
        StageManifestEntry(order=1, name="input", active=True, artifact="default"),
        StageManifestEntry(
            order=6, name="api", active=True, artifact="default",
            strategies={"retry": "exponential_backoff", "ghost_slot": "x"},
            config={"provider": "anthropic"},
        ),
        StageManifestEntry(order=9, name="parse", active=True, artifact="default"),
        StageManifestEntry(order=21, name="yield", active=True, artifact="default"),
    ]
    with pytest.raises(ConfigError, match="strategy.unknown_slot"):
        Pipeline.from_manifest(
            _manifest(entries), credentials=CredentialBundle(), strict=True
        )
