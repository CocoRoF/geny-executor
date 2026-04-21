"""Tests for state.llm_client slot and Pipeline.attach_runtime wiring."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor import Pipeline, PipelineConfig, PipelineState
from geny_executor.llm_client import BaseClient, ClientCapabilities, ProviderBackedClient
from geny_executor.llm_client.types import APIResponse, ContentBlock
from geny_executor.stages.s01_input import InputStage
from geny_executor.stages.s06_api import APIStage, MockProvider
from geny_executor.stages.s09_parse import ParseStage
from geny_executor.stages.s16_yield import YieldStage


def test_fresh_state_has_null_llm_client():
    state = PipelineState()
    assert state.llm_client is None


class _FakeClient(BaseClient):
    provider = "fake"
    capabilities = ClientCapabilities()

    async def _send(self, request, *, purpose=""):
        return APIResponse(content=[ContentBlock(type="text", text="fake")], stop_reason="end_turn")


def _mock_pipeline() -> Pipeline:
    pipeline = Pipeline(PipelineConfig(name="test"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(provider=MockProvider()))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())
    return pipeline


@pytest.mark.asyncio
async def test_attach_runtime_accepts_explicit_client():
    pipeline = _mock_pipeline()
    client = _FakeClient()
    pipeline.attach_runtime(llm_client=client)
    result = await pipeline.run("hi")
    assert result is not None


@pytest.mark.asyncio
async def test_explicit_client_lands_on_state():
    pipeline = _mock_pipeline()
    client = _FakeClient()
    pipeline.attach_runtime(llm_client=client)
    captured: dict = {}

    class _Probe(InputStage):
        async def execute(self, input, state):
            captured["client"] = state.llm_client
            return await super().execute(input, state)

    pipeline2 = Pipeline(PipelineConfig(name="probe"))
    pipeline2.register_stage(_Probe())
    pipeline2.register_stage(APIStage(provider=MockProvider()))
    pipeline2.register_stage(ParseStage())
    pipeline2.register_stage(YieldStage())
    pipeline2.attach_runtime(llm_client=client)
    await pipeline2.run("hi")
    assert captured["client"] is client


@pytest.mark.asyncio
async def test_auto_bridge_from_s06_provider_when_no_explicit_client():
    captured: dict = {}

    class _Probe(InputStage):
        async def execute(self, input, state):
            captured["client"] = state.llm_client
            return await super().execute(input, state)

    pipeline2 = Pipeline(PipelineConfig(name="auto-bridge"))
    pipeline2.register_stage(_Probe())
    pipeline2.register_stage(APIStage(provider=MockProvider()))
    pipeline2.register_stage(ParseStage())
    pipeline2.register_stage(YieldStage())
    await pipeline2.run("hi")
    client = captured.get("client")
    assert isinstance(client, ProviderBackedClient)
    assert client.provider == "mock"


@pytest.mark.asyncio
async def test_no_api_stage_leaves_client_none():
    captured: dict = {}

    class _Probe(ParseStage):
        async def execute(self, input, state):
            captured["client"] = state.llm_client
            return await super().execute(input, state)

    pipeline2 = Pipeline(PipelineConfig(name="parse-only-probe"))
    pipeline2.register_stage(InputStage())
    pipeline2.register_stage(_Probe())
    pipeline2.register_stage(YieldStage())
    await pipeline2.run("hi")
    assert captured["client"] is None
