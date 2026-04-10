"""Phase 1 tests — minimal pipeline: Input → API(mock) → Parse → Yield."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor import Pipeline, PipelineConfig, PipelineState
from geny_executor.core.state import TokenUsage
from geny_executor.stages.s01_input import InputStage
from geny_executor.stages.s06_api import APIStage, MockProvider, APIResponse
from geny_executor.stages.s06_api.types import ContentBlock
from geny_executor.stages.s09_parse import ParseStage
from geny_executor.stages.s16_yield import YieldStage


def _make_mock_pipeline(text: str = "Hello from mock!", **kwargs) -> Pipeline:
    """Create a minimal pipeline with MockProvider."""
    provider = MockProvider(default_text=text)
    config = PipelineConfig(name="test")

    pipeline = Pipeline(config)
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(provider=provider))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())
    return pipeline


# ── Pipeline basic execution ──


@pytest.mark.asyncio
async def test_minimal_pipeline():
    """Basic pipeline: Input → API → Parse → Yield produces text."""
    pipeline = _make_mock_pipeline("Test response")
    result = await pipeline.run("Hello")

    assert result.success is True
    assert result.text == "Test response"
    assert result.iterations == 0


@pytest.mark.asyncio
async def test_pipeline_events():
    """Pipeline emits stage events."""
    pipeline = _make_mock_pipeline()
    events = []
    pipeline.on("*", lambda e: events.append(e))

    await pipeline.run("Hi")

    event_types = [e.type for e in events]
    assert "pipeline.start" in event_types
    assert "stage.enter" in event_types
    assert "stage.exit" in event_types
    assert "pipeline.complete" in event_types


@pytest.mark.asyncio
async def test_pipeline_state_tracks_messages():
    """State accumulates messages through the pipeline."""
    pipeline = _make_mock_pipeline("Response text")
    state = PipelineState(session_id="test-session")
    await pipeline.run("User input", state)

    # Should have user message + assistant message
    assert len(state.messages) == 2
    assert state.messages[0]["role"] == "user"
    assert state.messages[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_pipeline_with_custom_state():
    """Custom state is preserved through execution."""
    pipeline = _make_mock_pipeline()
    state = PipelineState(
        session_id="my-session",
        metadata={"custom": "value"},
    )
    result = await pipeline.run("Hello", state)

    assert result.session_id == "my-session"
    assert state.metadata["custom"] == "value"


# ── InputStage ──


@pytest.mark.asyncio
async def test_input_stage_validates():
    """InputStage rejects empty input."""
    pipeline = _make_mock_pipeline()
    result = await pipeline.run("")

    # Empty input should fail validation
    assert result.success is False


@pytest.mark.asyncio
async def test_input_stage_normalizes():
    """InputStage normalizes text."""
    pipeline = _make_mock_pipeline()
    result = await pipeline.run("  Hello World  ")
    assert result.success is True


# ── MockProvider ──


@pytest.mark.asyncio
async def test_mock_provider_queued_responses():
    """MockProvider returns queued responses in order."""
    provider = MockProvider()
    provider.add_response(
        APIResponse(
            content=[ContentBlock(type="text", text="First")],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            model="test",
        )
    )
    provider.add_response(
        APIResponse(
            content=[ContentBlock(type="text", text="Second")],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            model="test",
        )
    )

    pipeline = Pipeline(PipelineConfig(name="test"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(provider=provider))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())

    r1 = await pipeline.run("Q1")
    assert r1.text == "First"

    r2 = await pipeline.run("Q2")
    assert r2.text == "Second"


# ── Completion signals ──


@pytest.mark.asyncio
async def test_completion_signal_detection():
    """ParseStage detects [TASK_COMPLETE] signal."""
    provider = MockProvider(default_text="Done! [TASK_COMPLETE]")
    pipeline = Pipeline(PipelineConfig(name="test"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(provider=provider))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())

    state = PipelineState()
    await pipeline.run("Do something", state)

    assert state.completion_signal == "complete"


@pytest.mark.asyncio
async def test_continue_signal():
    """ParseStage detects [CONTINUE: next step] signal."""
    provider = MockProvider(default_text="Working... [CONTINUE: analyze results]")
    pipeline = Pipeline(PipelineConfig(name="test"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(provider=provider))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())

    state = PipelineState()
    await pipeline.run("Start", state)

    assert state.completion_signal == "continue"
    assert state.completion_detail == "analyze results"


# ── Pipeline describe (UI metadata) ──


def test_pipeline_describe():
    """Pipeline.describe() returns stage metadata."""
    pipeline = _make_mock_pipeline()
    desc = pipeline.describe()

    assert len(desc) == 16  # all 16 slots
    active = [d for d in desc if d.is_active]
    assert len(active) == 4  # 4 registered stages
    names = {d.name for d in active}
    assert names == {"input", "api", "parse", "yield"}


# ── Streaming ──


@pytest.mark.asyncio
async def test_streaming_mode():
    """run_stream yields PipelineEvents including text.delta."""
    pipeline = _make_mock_pipeline("Streamed response")
    events = []
    async for event in pipeline.run_stream("Hi"):
        events.append(event)

    types = [e.type for e in events]
    assert "pipeline.start" in types
    assert "pipeline.complete" in types

    # MockProvider now streams text chunks → text.delta events must arrive
    deltas = [e for e in events if e.type == "text.delta"]
    assert len(deltas) > 0, f"Expected text.delta events, got: {types}"


# ── Error handling ──


@pytest.mark.asyncio
async def test_pipeline_error_handling():
    """Pipeline returns error result on exception."""
    from geny_executor.stages.s06_api.providers import APIProvider

    class FailProvider(APIProvider):
        @property
        def name(self):
            return "fail"

        async def create_message(self, request):
            raise RuntimeError("API exploded")

    pipeline = Pipeline(PipelineConfig(name="test"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(
        APIStage(
            provider=FailProvider(),
            retry=__import__("geny_executor.stages.s06_api.retry", fromlist=["NoRetry"]).NoRetry(),
        )
    )
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())

    result = await pipeline.run("Hello")
    assert result.success is False
    assert "API exploded" in (result.error or "")
