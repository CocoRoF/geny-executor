"""attach_runtime(llm_client=) provider-mismatch guard (2.2.0, audit §2.7).

Incident #866: a host attached an Anthropic client onto a pipeline
whose manifest declared ``claude_code_cli`` — the attached client beats
the manifest unconditionally in ``_resolve_llm_client``, so every run
silently used the wrong backend. The only prior defence was a comment
inside Geny. These tests pin the structural guard:

* manifest provider declared + client reports a DIFFERENT provider →
  ``ConfigError`` naming both;
* same provider → allowed;
* hand-built / fixture pipelines (no manifest declaration) → allowed,
  whatever the client says;
* ``override_manifest=True`` → allowed + announced via a
  ``runtime.llm_client_override`` event at the next run start.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor import Pipeline, PipelineConfig
from geny_executor.core.environment import (
    EnvironmentManifest,
    EnvironmentMetadata,
    StageManifestEntry,
    ToolsSnapshot,
)
from geny_executor.llm_client.credentials import ConfigError, CredentialBundle
from geny_executor.stages.s01_input import InputStage
from geny_executor.stages.s06_api import APIStage, MockProvider
from geny_executor.stages.s09_parse import ParseStage
from geny_executor.stages.s21_yield import YieldStage


class _FakeClient:
    """Duck-typed BaseClient — only the ``provider`` attr matters here."""

    def __init__(self, provider: str = "anthropic") -> None:
        self.provider = provider


def _manifest(provider: str = "claude_code_cli") -> EnvironmentManifest:
    m = EnvironmentManifest(
        metadata=EnvironmentMetadata(id="env_866", name="guard"),
        model={},
        pipeline={},
        stages=[],
        tools=ToolsSnapshot(),
    )
    m.set_stage_entries(
        [
            StageManifestEntry(order=1, name="input", active=True, artifact="default"),
            StageManifestEntry(
                order=6,
                name="api",
                active=True,
                artifact="default",
                config={"provider": provider},
            ),
            StageManifestEntry(order=9, name="parse", active=True, artifact="default"),
            StageManifestEntry(order=21, name="yield", active=True, artifact="default"),
        ]
    )
    return m


def _manifest_pipeline(provider: str = "claude_code_cli") -> Pipeline:
    return Pipeline.from_manifest(
        _manifest(provider), credentials=CredentialBundle(), strict=True
    )


def _fixture_pipeline() -> Pipeline:
    """Hand-built pipeline — the builder/fixture class the guard must
    never touch (its APIStage derives _provider_name='mock' from the
    provider object, which is NOT a manifest declaration)."""
    pipeline = Pipeline(PipelineConfig(name="fixture"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(provider=MockProvider(default_text="hi")))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())
    return pipeline


# ── Mismatch raises ──────────────────────────────────────────────────


def test_mismatched_client_raises_naming_both_providers():
    pipeline = _manifest_pipeline("claude_code_cli")
    with pytest.raises(ConfigError) as excinfo:
        pipeline.attach_runtime(llm_client=_FakeClient("anthropic"))
    message = str(excinfo.value)
    assert "anthropic" in message
    assert "claude_code_cli" in message
    assert "#866" in message
    # And nothing got attached — the pipeline is unchanged.
    assert pipeline._attached_llm_client is None


def test_mismatch_raises_via_refresh_runtime_too():
    """refresh_runtime shares the wiring, so it shares the guard."""
    pipeline = _manifest_pipeline("claude_code_cli")
    with pytest.raises(ConfigError, match="#866"):
        pipeline.refresh_runtime(llm_client=_FakeClient("openai"))


# ── Allowed cases ────────────────────────────────────────────────────


def test_matching_provider_is_allowed():
    pipeline = _manifest_pipeline("anthropic")
    client = _FakeClient("anthropic")
    pipeline.attach_runtime(llm_client=client)
    assert pipeline._attached_llm_client is client


def test_fixture_pipeline_unaffected():
    """No manifest declaration → no guard, even though the APIStage's
    constructor-derived _provider_name ('mock') differs from the client."""
    pipeline = _fixture_pipeline()
    client = _FakeClient("anthropic")
    pipeline.attach_runtime(llm_client=client)
    assert pipeline._attached_llm_client is client


def test_client_without_provider_attr_is_allowed():
    """Clients that don't report a provider can't be checked — the
    guard refuses only provable mismatches."""
    pipeline = _manifest_pipeline("claude_code_cli")

    class _Opaque:
        pass

    client = _Opaque()
    pipeline.attach_runtime(llm_client=client)
    assert pipeline._attached_llm_client is client


# ── override_manifest escape hatch ───────────────────────────────────


def test_override_manifest_allows_mismatch():
    pipeline = _manifest_pipeline("claude_code_cli")
    client = _FakeClient("anthropic")
    pipeline.attach_runtime(llm_client=client, override_manifest=True)
    assert pipeline._attached_llm_client is client


@pytest.mark.asyncio
async def test_override_emits_event_at_next_run_start():
    """The acknowledged override must be visible in the event stream —
    a silent override is the original foot-gun with extra steps."""
    pipeline = _manifest_pipeline("claude_code_cli")
    # The fake client is not callable as a real backend; replace the API
    # stage with a Mock-backed one AFTER attaching so the run completes.
    pipeline.attach_runtime(
        llm_client=_FakeClient("anthropic"), override_manifest=True
    )
    pipeline.register_stage(APIStage(provider=MockProvider(default_text="ok")))
    pipeline._attached_llm_client = None  # let the stage's mock provider serve

    result = await pipeline.run("hello")

    override_events = [
        e for e in result.events if e["type"] == "runtime.llm_client_override"
    ]
    assert len(override_events) == 1
    assert override_events[0]["data"] == {
        "manifest_provider": "claude_code_cli",
        "client_provider": "anthropic",
    }

    # One-shot announcement: the next run does not repeat it.
    second = await pipeline.run("again", result.state)
    assert not [
        e for e in second.events if e["type"] == "runtime.llm_client_override"
    ]
