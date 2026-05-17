"""Tests for state.llm_client slot and Pipeline.attach_runtime wiring.

After Phase A3, state.llm_client is sourced from one of two places:
  1. ``Pipeline.attach_runtime(llm_client=...)`` (host-supplied client)
  2. ``Pipeline.from_manifest_async(credentials=...)`` resolving Stage 6's
     ``config["provider"]`` through ``ClientRegistry`` + ``CredentialBundle``

Manifest-driven resolution is exercised by the conformance harness; this
file focuses on the attach_runtime + fresh-state contract.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor import Pipeline, PipelineConfig, PipelineState
from geny_executor.llm_client import BaseClient, ClientCapabilities
from geny_executor.llm_client.types import APIResponse, ContentBlock
from geny_executor.stages.s01_input import InputStage
from geny_executor.stages.s06_api import APIStage
from geny_executor.stages.s09_parse import ParseStage
from geny_executor.stages.s21_yield import YieldStage


def test_fresh_state_has_null_llm_client():
    state = PipelineState()
    assert state.llm_client is None


def test_fresh_state_has_null_credentials():
    state = PipelineState()
    assert state.credentials is None


class _FakeClient(BaseClient):
    provider = "fake"
    capabilities = ClientCapabilities()

    async def _send(self, request, *, purpose=""):
        return APIResponse(
            content=[ContentBlock(type="text", text="fake")], stop_reason="end_turn"
        )


def _build_pipeline(stages):
    pipeline = Pipeline(PipelineConfig(name="test"))
    for s in stages:
        pipeline.register_stage(s)
    return pipeline


@pytest.mark.asyncio
async def test_attach_runtime_accepts_explicit_client():
    pipeline = _build_pipeline([
        InputStage(),
        APIStage(provider="anthropic"),
        ParseStage(),
        YieldStage(),
    ])
    client = _FakeClient()
    pipeline.attach_runtime(llm_client=client)
    result = await pipeline.run("hi")
    assert result is not None


@pytest.mark.asyncio
async def test_explicit_client_lands_on_state():
    client = _FakeClient()
    captured: dict = {}

    class _Probe(InputStage):
        async def execute(self, input, state):
            captured["client"] = state.llm_client
            return await super().execute(input, state)

    pipeline = _build_pipeline([
        _Probe(),
        APIStage(provider="anthropic"),
        ParseStage(),
        YieldStage(),
    ])
    pipeline.attach_runtime(llm_client=client)
    await pipeline.run("hi")
    assert captured["client"] is client


@pytest.mark.asyncio
async def test_no_attach_no_credentials_leaves_client_none():
    """Manual pipelines (no from_manifest, no attach_runtime) get None."""
    captured: dict = {}

    class _Probe(InputStage):
        async def execute(self, input, state):
            captured["client"] = state.llm_client
            return await super().execute(input, state)

    pipeline = _build_pipeline([
        _Probe(),
        APIStage(provider="anthropic"),
        ParseStage(),
        YieldStage(),
    ])
    # Stage 6 is registered but no credentials bundle is wired on the
    # pipeline → _resolve_llm_client cannot build a client and the
    # Stage 6 execute path will surface that. We probe the InputStage
    # which runs before the API stage so we observe the None state.
    # The pipeline will fail later at stage 6, which is the contract.
    try:
        await pipeline.run("hi")
    except Exception:
        # Stage 6 raises because no client is available — expected.
        pass
    assert captured["client"] is None


@pytest.mark.asyncio
async def test_no_api_stage_leaves_client_none():
    captured: dict = {}

    class _Probe(ParseStage):
        async def execute(self, input, state):
            captured["client"] = state.llm_client
            return await super().execute(input, state)

    pipeline = _build_pipeline([
        InputStage(),
        _Probe(),
        YieldStage(),
    ])
    await pipeline.run("hi")
    assert captured["client"] is None
