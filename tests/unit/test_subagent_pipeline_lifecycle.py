"""2.2.0 review B1 — dispatch-built sub-pipelines are closed per dispatch.

``SubagentTypeOrchestrator._run_descriptor`` used to build a
sub-pipeline and drop the handle: a manifest-declared sub-environment
(``ManifestSubagentPipelineFactory`` → ``Pipeline.from_manifest_async``)
may connect MCP servers / tool providers / memory providers, all of
which leaked once per dispatch — the same stdio-child leak class
``Pipeline.aclose`` exists to stop, one level down. Pinned here:

  (a) exactly one ``aclose()`` per dispatch for manifest-built
      descriptors — batch path and single-call path;
  (b) the close also happens when ``run()`` fails;
  (c) factory failure (nothing built) closes nothing;
  (d) foreign factory objects without ``aclose`` are skipped, and an
      ``aclose`` that raises is swallowed (a teardown failure must not
      poison a good sub-result).
"""

from __future__ import annotations

import pytest

from geny_executor.core.pipeline import Pipeline
from geny_executor.core.shared_keys import SharedKeys
from geny_executor.core.state import PipelineState
from geny_executor.llm_client.credentials import CredentialBundle, ProviderCredentials
from geny_executor.stages.s12_agent.subagent_type import (
    SubagentTypeDescriptor,
    SubagentTypeOrchestrator,
    SubagentTypeRegistry,
    compile_subagent_descriptors,
)


def _bundle(*providers: str) -> CredentialBundle:
    return CredentialBundle(
        by_provider={p: ProviderCredentials(api_key=f"sk-{p}") for p in providers}
    )


def _parent_state() -> PipelineState:
    state = PipelineState(session_id="parent")
    state.credentials = _bundle("anthropic")
    state.shared[SharedKeys.PRIMARY_PROVIDER] = "anthropic"
    return state


def _manifest_orchestrator() -> SubagentTypeOrchestrator:
    """Registry with one manifest-compiled descriptor (real Pipeline builds)."""
    registry = SubagentTypeRegistry()
    for d in compile_subagent_descriptors([{"agent_type": "worker"}]):
        registry.register(d)
    return SubagentTypeOrchestrator(registry)


def _spy_on_aclose(monkeypatch) -> list:
    """Count every Pipeline.aclose() call; real teardown still runs."""
    calls: list = []
    original = Pipeline.aclose

    async def spy(self):  # noqa: ANN001
        calls.append(self)
        await original(self)

    monkeypatch.setattr(Pipeline, "aclose", spy)
    return calls


@pytest.mark.asyncio
async def test_orchestrate_closes_manifest_built_pipeline_once(monkeypatch):
    calls = _spy_on_aclose(monkeypatch)
    orch = _manifest_orchestrator()
    state = _parent_state()
    state.delegate_requests = [{"agent_type": "worker", "task": "go"}]

    # MockProvider isn't registered for "anthropic" here, so the sub-run
    # itself may fail — the point under test is the lifecycle: exactly
    # one aclose per dispatch, regardless of run outcome.
    result = await orch.orchestrate(state)

    assert result.delegated is True
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_each_dispatch_closes_its_own_pipeline(monkeypatch):
    calls = _spy_on_aclose(monkeypatch)
    orch = _manifest_orchestrator()
    state = _parent_state()
    state.delegate_requests = [
        {"agent_type": "worker", "task": "one"},
        {"agent_type": "worker", "task": "two"},
    ]

    await orch.orchestrate(state)

    assert len(calls) == 2
    # Per-call factory: two dispatches build (and close) two pipelines.
    assert calls[0] is not calls[1]


@pytest.mark.asyncio
async def test_run_failure_still_closes_pipeline(monkeypatch):
    calls = _spy_on_aclose(monkeypatch)

    async def explode(self, input, state=None, **kwargs):  # noqa: ANN001
        raise RuntimeError("sub-run exploded")

    monkeypatch.setattr(Pipeline, "run", explode)
    orch = _manifest_orchestrator()
    state = _parent_state()
    state.delegate_requests = [{"agent_type": "worker", "task": "go"}]

    result = await orch.orchestrate(state)

    (record,) = result.sub_results
    assert record["success"] is False
    assert "run_error" in record["error"]
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_run_subagent_single_call_closes_pipeline(monkeypatch):
    calls = _spy_on_aclose(monkeypatch)

    class _Result:
        success = True
        text = "done"
        error = None

    async def fake_run(self, input, state=None, **kwargs):  # noqa: ANN001
        return _Result()

    monkeypatch.setattr(Pipeline, "run", fake_run)
    orch = _manifest_orchestrator()

    record = await orch.run_subagent("worker", "go", state=_parent_state())

    assert record["success"] is True
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_factory_failure_closes_nothing(monkeypatch):
    calls = _spy_on_aclose(monkeypatch)

    def factory(ctx):  # noqa: ANN001
        raise RuntimeError("factory exploded")

    registry = SubagentTypeRegistry().register(
        SubagentTypeDescriptor(agent_type="t", factory=factory)
    )
    state = _parent_state()
    state.delegate_requests = [{"agent_type": "t", "task": "go"}]

    result = await SubagentTypeOrchestrator(registry).orchestrate(state)

    (record,) = result.sub_results
    assert "factory_error" in record["error"]
    assert calls == []


@pytest.mark.asyncio
async def test_foreign_pipeline_without_aclose_is_skipped():
    class _Foreign:
        async def run(self, task, state):  # noqa: ANN001
            class _R:
                success = True
                text = "ok"
                error = None

            return _R()

    registry = SubagentTypeRegistry().register(
        SubagentTypeDescriptor(agent_type="t", factory=lambda ctx: _Foreign())
    )
    state = _parent_state()
    state.delegate_requests = [{"agent_type": "t", "task": "go"}]

    result = await SubagentTypeOrchestrator(registry).orchestrate(state)

    (record,) = result.sub_results
    assert record["success"] is True


@pytest.mark.asyncio
async def test_aclose_failure_is_swallowed():
    closed = []

    class _Fragile:
        async def run(self, task, state):  # noqa: ANN001
            class _R:
                success = True
                text = "ok"
                error = None

            return _R()

        async def aclose(self):
            closed.append(True)
            raise RuntimeError("teardown wedged")

    registry = SubagentTypeRegistry().register(
        SubagentTypeDescriptor(agent_type="t", factory=lambda ctx: _Fragile())
    )
    state = _parent_state()
    state.delegate_requests = [{"agent_type": "t", "task": "go"}]

    result = await SubagentTypeOrchestrator(registry).orchestrate(state)

    (record,) = result.sub_results
    assert record["success"] is True  # teardown failure did not poison it
    assert record["error"] is None
    assert closed == [True]
