"""Phase 2 tests — agent loop with tool use."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor import Pipeline, PipelineConfig, PipelineState
from geny_executor.core.state import TokenUsage
from geny_executor.stages.s01_input import InputStage
from geny_executor.stages.s04_guard import GuardStage, CostBudgetGuard, IterationGuard
from geny_executor.stages.s06_api import APIStage, MockProvider, APIResponse
from geny_executor.stages.s06_api.types import ContentBlock
from geny_executor.stages.s06_api.retry import NoRetry
from geny_executor.stages.s07_token import TokenStage
from geny_executor.stages.s09_parse import ParseStage
from geny_executor.stages.s10_tool import ToolStage, SequentialExecutor
from geny_executor.stages.s13_loop import LoopStage, StandardLoopController, SingleTurnController
from geny_executor.stages.s16_yield import YieldStage
from geny_executor.tools import Tool, ToolResult, ToolContext, ToolRegistry


# ── Test Tools ──


class EchoTool(Tool):
    @property
    def name(self): return "echo"
    @property
    def description(self): return "Echoes input"
    @property
    def input_schema(self):
        return {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    async def execute(self, input, context):
        return ToolResult(content=f"Echo: {input['text']}")


class CalculatorTool(Tool):
    @property
    def name(self): return "calculator"
    @property
    def description(self): return "Evaluates math"
    @property
    def input_schema(self):
        return {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}

    async def execute(self, input, context):
        try:
            result = eval(input["expression"])  # noqa: S307
            return ToolResult(content=str(result))
        except Exception as e:
            return ToolResult(content=str(e), is_error=True)


# ── Helpers ──


def _tool_use_response(tool_name: str, tool_input: dict, tool_id: str = "tu_1") -> APIResponse:
    return APIResponse(
        content=[
            ContentBlock(
                type="tool_use",
                tool_use_id=tool_id,
                tool_name=tool_name,
                tool_input=tool_input,
                raw={"type": "tool_use", "id": tool_id, "name": tool_name, "input": tool_input},
            )
        ],
        stop_reason="tool_use",
        usage=TokenUsage(input_tokens=100, output_tokens=50),
        model="test",
    )


def _text_response(text: str) -> APIResponse:
    return APIResponse(
        content=[ContentBlock(type="text", text=text)],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=50, output_tokens=30),
        model="test",
    )


def _make_agent_pipeline(provider: MockProvider, registry: ToolRegistry) -> Pipeline:
    """Full agent pipeline with tool support."""
    pipeline = Pipeline(PipelineConfig(name="agent-test"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(GuardStage([IterationGuard(max_iterations=10)]))
    pipeline.register_stage(APIStage(provider=provider, retry=NoRetry()))
    pipeline.register_stage(TokenStage())
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(ToolStage(registry=registry))
    pipeline.register_stage(LoopStage(StandardLoopController()))
    pipeline.register_stage(YieldStage())
    return pipeline


# ── Tests ──


@pytest.mark.asyncio
async def test_tool_use_loop():
    """Pipeline calls tool, gets result, continues to final response."""
    provider = MockProvider()
    # Turn 1: Model requests tool use
    provider.add_response(_tool_use_response("echo", {"text": "hello"}))
    # Turn 2: Model responds with final text
    provider.add_response(_text_response("The echo said: hello"))

    registry = ToolRegistry()
    registry.register(EchoTool())

    pipeline = _make_agent_pipeline(provider, registry)
    result = await pipeline.run("Use echo tool")

    assert result.success is True
    assert result.text == "The echo said: hello"
    assert provider.call_count == 2


@pytest.mark.asyncio
async def test_multi_tool_calls():
    """Pipeline handles multiple tool calls in one response."""
    provider = MockProvider()
    # Turn 1: Two tool calls
    provider.add_response(APIResponse(
        content=[
            ContentBlock(type="tool_use", tool_use_id="tu_1", tool_name="echo", tool_input={"text": "a"}),
            ContentBlock(type="tool_use", tool_use_id="tu_2", tool_name="calculator", tool_input={"expression": "2+2"}),
        ],
        stop_reason="tool_use",
        usage=TokenUsage(input_tokens=100, output_tokens=60),
        model="test",
    ))
    # Turn 2: Final
    provider.add_response(_text_response("Results: Echo a, Calc 4"))

    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(CalculatorTool())

    pipeline = _make_agent_pipeline(provider, registry)
    result = await pipeline.run("Use both tools")

    assert result.success is True
    assert result.text == "Results: Echo a, Calc 4"
    assert provider.call_count == 2


@pytest.mark.asyncio
async def test_tool_error_handling():
    """Pipeline handles tool execution errors gracefully."""
    provider = MockProvider()
    provider.add_response(_tool_use_response("calculator", {"expression": "invalid///"}))
    provider.add_response(_text_response("Calculator had an error"))

    registry = ToolRegistry()
    registry.register(CalculatorTool())

    pipeline = _make_agent_pipeline(provider, registry)
    result = await pipeline.run("Calculate something bad")

    assert result.success is True  # Pipeline itself succeeds
    assert "error" in result.text.lower() or provider.call_count == 2


@pytest.mark.asyncio
async def test_unknown_tool():
    """Pipeline handles unknown tool gracefully."""
    provider = MockProvider()
    provider.add_response(_tool_use_response("nonexistent", {"x": 1}))
    provider.add_response(_text_response("Tool not found, proceeding"))

    registry = ToolRegistry()

    pipeline = _make_agent_pipeline(provider, registry)
    result = await pipeline.run("Use missing tool")
    assert result.success is True


@pytest.mark.asyncio
async def test_single_turn_controller():
    """SingleTurnController prevents looping."""
    provider = MockProvider(default_text="One shot")

    pipeline = Pipeline(PipelineConfig(name="single"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(provider=provider, retry=NoRetry()))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(LoopStage(SingleTurnController()))
    pipeline.register_stage(YieldStage())

    result = await pipeline.run("Hello")
    assert result.success is True
    assert result.text == "One shot"
    assert provider.call_count == 1


@pytest.mark.asyncio
async def test_iteration_guard():
    """IterationGuard prevents infinite loops."""
    provider = MockProvider()
    # Keep returning tool_use forever
    for _ in range(15):
        provider.add_response(_tool_use_response("echo", {"text": "loop"}))
    provider.add_response(_text_response("Done"))

    registry = ToolRegistry()
    registry.register(EchoTool())

    pipeline = Pipeline(PipelineConfig(name="guard-test", max_iterations=5))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(GuardStage([IterationGuard(max_iterations=5)]))
    pipeline.register_stage(APIStage(provider=provider, retry=NoRetry()))
    pipeline.register_stage(TokenStage())
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(ToolStage(registry=registry))
    pipeline.register_stage(LoopStage(StandardLoopController()))
    pipeline.register_stage(YieldStage())

    result = await pipeline.run("Loop forever")

    # Should have stopped at iteration limit (guard rejects at iteration 5)
    assert result.iterations <= 6


@pytest.mark.asyncio
async def test_cost_guard():
    """CostBudgetGuard stops when budget exceeded."""
    pipeline = Pipeline(PipelineConfig(name="cost-test"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(GuardStage([CostBudgetGuard(max_cost_usd=0.001)]))
    pipeline.register_stage(APIStage(provider=MockProvider(), retry=NoRetry()))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())

    state = PipelineState(total_cost_usd=0.002)  # Already over budget
    result = await pipeline.run("Hello", state)

    # Should fail due to cost guard
    assert result.success is False


@pytest.mark.asyncio
async def test_token_tracking():
    """TokenStage tracks usage and calculates cost."""
    provider = MockProvider()
    provider.add_response(APIResponse(
        content=[ContentBlock(type="text", text="Response")],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=1000, output_tokens=500),
        model="claude-sonnet-4-20250514",
    ))

    pipeline = Pipeline(PipelineConfig(name="token-test"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(provider=provider, retry=NoRetry()))
    pipeline.register_stage(TokenStage())
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(LoopStage(SingleTurnController()))
    pipeline.register_stage(YieldStage())

    state = PipelineState()
    result = await pipeline.run("Hello", state)

    assert state.token_usage.input_tokens == 1000
    assert state.token_usage.output_tokens == 500
    assert state.total_cost_usd > 0


@pytest.mark.asyncio
async def test_tool_registry():
    """ToolRegistry basic operations."""
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(CalculatorTool())

    assert len(registry) == 2
    assert "echo" in registry
    assert "calculator" in registry

    tool = registry.get("echo")
    assert tool is not None
    assert tool.name == "echo"

    api_format = registry.to_api_format()
    assert len(api_format) == 2
    assert all("name" in t and "description" in t and "input_schema" in t for t in api_format)

    filtered = registry.filter(include={"echo"})
    assert len(filtered) == 1

    registry.unregister("echo")
    assert "echo" not in registry
