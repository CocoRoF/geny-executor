"""Tests for APIStage provider-string kwarg + state.llm_client routing."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor import (
    Pipeline,
    PipelineConfig,
    ClientRegistry,
)
from geny_executor.llm_client import BaseClient, ClientCapabilities
from geny_executor.llm_client.types import APIResponse, ContentBlock
from geny_executor.stages.s01_input import InputStage
from geny_executor.stages.s06_api import APIStage, MockProvider
from geny_executor.stages.s09_parse import ParseStage
from geny_executor.stages.s21_yield import YieldStage


class _CaptureClient(BaseClient):
    """Records create_message invocations."""

    provider = "capture"
    capabilities = ClientCapabilities(
        supports_thinking=True,
        supports_tools=True,
        supports_streaming=True,
        supports_tool_choice=True,
        supports_stop_sequences=True,
        supports_top_k=True,
        supports_system_prompt=True,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls: list = []

    async def _send(self, request, *, purpose=""):
        self.calls.append({"request": request, "purpose": purpose})
        return APIResponse(
            content=[ContentBlock(type="text", text="captured")],
            stop_reason="end_turn",
        )


def test_default_provider_is_anthropic():
    stage = APIStage(api_key="sk-test")
    assert stage.get_config()["provider"] == "anthropic"


def test_provider_string_anthropic_reported_in_config():
    stage = APIStage(provider="anthropic", api_key="sk-test")
    assert stage.get_config()["provider"] == "anthropic"


def test_provider_string_mock_reported_in_config():
    stage = APIStage(provider="mock")
    assert stage.get_config()["provider"] == "mock"


def test_provider_instance_infers_name():
    stage = APIStage(provider=MockProvider())
    assert stage.get_config()["provider"] == "mock"


def test_update_config_switches_provider():
    stage = APIStage(provider="mock")
    stage.update_config({"provider": "anthropic"})
    assert stage.get_config()["provider"] == "anthropic"


def test_schema_provider_field_lists_registered_providers():
    stage = APIStage(provider="mock")
    schema = stage.get_config_schema()
    provider_field = next(f for f in schema.fields if f.name == "provider")
    assert provider_field.type == "select"
    values = {opt["value"] for opt in provider_field.options}
    assert {"anthropic", "openai", "google", "vllm", "mock"} <= values


@pytest.mark.asyncio
async def test_execute_routes_through_state_llm_client():
    pipeline = Pipeline(PipelineConfig(name="test"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(provider=MockProvider()))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())

    client = _CaptureClient()
    pipeline.attach_runtime(llm_client=client)
    await pipeline.run("hi")
    assert len(client.calls) == 1
    assert client.calls[0]["purpose"] == "api"


@pytest.mark.asyncio
async def test_execute_uses_shared_client_not_local_provider():
    """When state.llm_client is attached, the stage's own provider is
    bypassed — the capture client's call_count increments, the underlying
    MockProvider's does not."""
    mock_provider = MockProvider()
    pipeline = Pipeline(PipelineConfig(name="test"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(provider=mock_provider))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())

    client = _CaptureClient()
    pipeline.attach_runtime(llm_client=client)
    await pipeline.run("hi")
    assert len(client.calls) == 1
    assert mock_provider.call_count == 0


@pytest.mark.asyncio
async def test_execute_falls_back_to_local_provider_without_attach():
    """No explicit attach + no api_key → auto-bridge from the provider slot."""
    mock_provider = MockProvider()
    pipeline = Pipeline(PipelineConfig(name="test"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(provider=mock_provider))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())
    await pipeline.run("hi")
    assert mock_provider.call_count == 1


def test_unknown_provider_in_registry_raises():
    with pytest.raises(ValueError):
        ClientRegistry.get("definitely_not_registered")


def test_vllm_provider_string_without_base_url_raises_on_resolve():
    """vLLM fallback client construction fails without base_url."""
    stage = APIStage(provider="vllm", api_key="EMPTY")
    from geny_executor.core.state import PipelineState

    state = PipelineState()
    with pytest.raises(ValueError) as ei:
        stage._resolve_client(state)
    assert "base_url" in str(ei.value).lower() or "vllm" in str(ei.value).lower()


def test_vllm_provider_string_with_base_url_constructs():
    stage = APIStage(provider="vllm", api_key="EMPTY", base_url="http://localhost:8000/v1")
    from geny_executor.core.state import PipelineState

    state = PipelineState()
    client = stage._resolve_client(state)
    assert client.provider == "vllm"
