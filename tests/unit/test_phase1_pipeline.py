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
from geny_executor.stages.s21_yield import YieldStage


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

    assert len(desc) == 21  # all 21 slots (S9a.3 widened layout)
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


@pytest.mark.asyncio
async def test_streaming_pipeline_complete_carries_full_result():
    """Regression: pipeline.complete.result must not be truncated.

    Consumers (e.g. Geny) overwrite the streamed accumulation with
    `result` on completion. If `result` is preview-truncated, the user
    sees the full text during streaming and a chopped version at the
    end.
    """
    long_text = "x" * 1500  # exceeds the historical 500-char preview cap
    pipeline = _make_mock_pipeline(long_text)
    completes = [e async for e in pipeline.run_stream("Hi") if e.type == "pipeline.complete"]
    assert len(completes) == 1
    assert completes[0].data["result"] == long_text


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


# ── Model parameter propagation ──


@pytest.mark.asyncio
async def test_model_config_propagates_all_params():
    """ModelConfig fields are fully propagated: config → state → request."""
    from geny_executor.core.config import ModelConfig, PipelineConfig

    # 1. Config → State propagation
    model_config = ModelConfig(
        model="claude-opus-4-6",
        max_tokens=4096,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        stop_sequences=["STOP"],
        thinking_enabled=True,
        thinking_budget_tokens=2048,
        thinking_type="adaptive",
        thinking_display="omitted",
    )
    config = PipelineConfig(model=model_config, cost_budget_usd=1.0)
    state = PipelineState()
    config.apply_to_state(state)

    assert state.model == "claude-opus-4-6"
    assert state.max_tokens == 4096
    assert state.temperature == 0.7
    assert state.top_p == 0.9
    assert state.top_k == 40
    assert state.stop_sequences == ["STOP"]
    assert state.thinking_enabled is True
    assert state.thinking_budget_tokens == 2048
    assert state.thinking_type == "adaptive"
    assert state.thinking_display == "omitted"
    assert state.cost_budget_usd == 1.0

    # 2. State → Request propagation (via APIStage._build_request)
    provider = MockProvider()
    api_stage = APIStage(provider=provider)
    request = api_stage._build_request(state)

    assert request.model == "claude-opus-4-6"
    assert request.max_tokens == 4096
    assert request.temperature == 0.7
    assert request.top_p == 0.9
    assert request.top_k == 40
    assert request.stop_sequences == ["STOP"]
    assert request.thinking == {"type": "adaptive", "display": "omitted"}


@pytest.mark.asyncio
async def test_builder_routes_model_kwargs_correctly():
    """PipelineBuilder.with_model() routes kwargs to ModelConfig vs PipelineConfig."""
    from geny_executor.core.builder import PipelineBuilder

    builder = PipelineBuilder("test", api_key="test-key")
    builder.with_model(
        "claude-opus-4-6",
        max_tokens=4096,
        temperature=0.5,
        top_p=0.85,
        top_k=50,
        thinking_enabled=True,
        thinking_type="adaptive",
        # PipelineConfig kwargs
        max_iterations=10,
    )

    pipeline = builder.build()
    state = PipelineState()
    pipeline._config.apply_to_state(state)

    # ModelConfig params correctly routed
    assert state.model == "claude-opus-4-6"
    assert state.max_tokens == 4096
    assert state.temperature == 0.5
    assert state.top_p == 0.85
    assert state.top_k == 50
    assert state.thinking_enabled is True
    assert state.thinking_type == "adaptive"

    # PipelineConfig params correctly routed
    assert state.max_iterations == 10


@pytest.mark.asyncio
async def test_cost_budget_enforced_in_loop():
    """Pipeline stops when cost budget is exceeded."""
    from geny_executor.core.config import ModelConfig

    # Create pipeline that loops: tool_use → tool_result → continue
    tool_response = APIResponse(
        content=[
            ContentBlock(type="text", text="Working..."),
            ContentBlock(
                type="tool_use",
                tool_use_id="t1",
                tool_name="test",
                tool_input={},
            ),
        ],
        stop_reason="tool_use",
        usage=TokenUsage(input_tokens=1000, output_tokens=500),
        model="claude-sonnet-4-6",
    )
    final_response = APIResponse(
        content=[ContentBlock(type="text", text="Done [TASK_COMPLETE]")],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=100, output_tokens=50),
        model="claude-sonnet-4-6",
    )

    provider = MockProvider()
    # Queue: tool_use → tool_use → final (but budget should stop after first)
    provider.add_response(tool_response)
    provider.add_response(tool_response)
    provider.add_response(final_response)

    # Very low budget: $0.0001 (should be exceeded after first API call)
    config = PipelineConfig(
        name="test",
        model=ModelConfig(model="claude-sonnet-4-6"),
        cost_budget_usd=0.0001,
    )

    from geny_executor.stages.s07_token import TokenStage
    from geny_executor.stages.s10_tool import ToolStage
    from geny_executor.stages.s16_loop import LoopStage
    from geny_executor.tools.registry import ToolRegistry
    from geny_executor.tools.base import Tool, ToolResult

    class DummyTool(Tool):
        @property
        def name(self):
            return "test"

        @property
        def description(self):
            return "test tool"

        @property
        def input_schema(self):
            return {"type": "object", "properties": {}}

        async def execute(self, input, context=None):
            return ToolResult(content="ok")

    registry = ToolRegistry()
    registry.register(DummyTool())

    pipeline = Pipeline(config)
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(provider=provider))
    pipeline.register_stage(TokenStage())
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(ToolStage(registry=registry))
    pipeline.register_stage(LoopStage())
    pipeline.register_stage(YieldStage())

    result = await pipeline.run("Do something expensive")
    assert result.success is True
    # Budget should have caused early termination
    assert result.total_cost_usd > 0
    # If budget worked, the pipeline should have used fewer iterations than
    # the 3 responses we queued
    events = [e for e in result.events if e.get("type") == "loop.force_complete"]
    if events:
        assert events[0]["data"]["reason"] == "cost_budget"


# ── stream / single_turn config ──


@pytest.mark.asyncio
async def test_stream_config_propagates_to_api_stage():
    """PipelineConfig.stream controls APIStage streaming behavior."""
    from geny_executor.core.config import PipelineConfig

    # stream=False should be propagated to state
    config = PipelineConfig(name="test", stream=False)
    state = PipelineState()
    config.apply_to_state(state)
    assert state.stream is False

    # stream=True (default) should be propagated
    config2 = PipelineConfig(name="test")
    state2 = PipelineState()
    config2.apply_to_state(state2)
    assert state2.stream is True

    # APIStage._resolve_stream reads from state
    provider = MockProvider()
    api_stage = APIStage(provider=provider, stream=True)

    # state.stream=False overrides constructor default
    assert api_stage._resolve_stream(state) is False
    # state.stream=True matches constructor
    assert api_stage._resolve_stream(state2) is True


@pytest.mark.asyncio
async def test_single_turn_completes_after_one_pass():
    """PipelineConfig.single_turn=True stops loop after first pass."""

    # Create a tool_use response that would normally cause looping
    tool_response = APIResponse(
        content=[
            ContentBlock(type="text", text="Using tool"),
            ContentBlock(
                type="tool_use",
                tool_use_id="t1",
                tool_name="test",
                tool_input={},
            ),
        ],
        stop_reason="tool_use",
        usage=TokenUsage(input_tokens=100, output_tokens=50),
        model="test",
    )
    final_response = APIResponse(
        content=[ContentBlock(type="text", text="Done")],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=100, output_tokens=50),
        model="test",
    )

    provider = MockProvider()
    provider.add_response(tool_response)
    provider.add_response(final_response)

    from geny_executor.stages.s07_token import TokenStage
    from geny_executor.stages.s10_tool import ToolStage
    from geny_executor.stages.s16_loop import LoopStage
    from geny_executor.tools.registry import ToolRegistry
    from geny_executor.tools.base import Tool, ToolResult

    class DummyTool(Tool):
        @property
        def name(self):
            return "test"

        @property
        def description(self):
            return "test"

        @property
        def input_schema(self):
            return {"type": "object", "properties": {}}

        async def execute(self, input, context=None):
            return ToolResult(content="ok")

    registry = ToolRegistry()
    registry.register(DummyTool())

    # single_turn=True → should complete after first pass even with tool_use
    config = PipelineConfig(name="test", single_turn=True)
    pipeline = Pipeline(config)
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(provider=provider))
    pipeline.register_stage(TokenStage())
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(ToolStage(registry=registry))
    pipeline.register_stage(LoopStage())
    pipeline.register_stage(YieldStage())

    result = await pipeline.run("Do something")
    assert result.success is True
    # single_turn should prevent a second loop iteration
    assert result.iterations == 0


# ── Assistant message format consistency ──


@pytest.mark.asyncio
async def test_assistant_content_always_list():
    """_build_assistant_content always returns List[Dict], never str."""
    provider = MockProvider(default_text="Simple text response")
    api_stage = APIStage(provider=provider)

    # Single text block — previously returned str, now should return list
    response = APIResponse(
        content=[ContentBlock(type="text", text="Hello")],
        stop_reason="end_turn",
    )
    content = api_stage._build_assistant_content(response)
    assert isinstance(content, list)
    assert len(content) == 1
    assert content[0] == {"type": "text", "text": "Hello"}

    # Multiple blocks — should also return list
    response2 = APIResponse(
        content=[
            ContentBlock(type="text", text="Let me use a tool"),
            ContentBlock(
                type="tool_use",
                tool_use_id="t1",
                tool_name="read",
                tool_input={"path": "/tmp/test"},
            ),
        ],
        stop_reason="tool_use",
    )
    content2 = api_stage._build_assistant_content(response2)
    assert isinstance(content2, list)
    assert len(content2) == 2
