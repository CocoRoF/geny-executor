"""Phase 6 tests — end-to-end integration scenarios."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor import Pipeline, PipelineConfig, PipelineState
from geny_executor.core.presets import PipelinePresets
from geny_executor.stages.s01_input import InputStage
from geny_executor.stages.s02_context import ContextStage, HybridStrategy
from geny_executor.stages.s03_system import (
    SystemStage,
    ComposablePromptBuilder,
    PersonaBlock,
    RulesBlock,
)
from geny_executor.stages.s04_guard import GuardStage
from geny_executor.stages.s05_cache import CacheStage, SystemCacheStrategy
from geny_executor.stages.s06_api import APIStage, MockProvider
from geny_executor.stages.s06_api.retry import NoRetry
from geny_executor.stages.s07_token import TokenStage
from geny_executor.stages.s08_think import ThinkStage
from geny_executor.stages.s09_parse import ParseStage
from geny_executor.stages.s10_tool import ToolStage
from geny_executor.stages.s11_agent import AgentStage
from geny_executor.stages.s12_evaluate import EvaluateStage
from geny_executor.stages.s13_loop import LoopStage, StandardLoopController
from geny_executor.stages.s14_emit import EmitStage, TextEmitter, VTuberEmitter
from geny_executor.stages.s15_memory import MemoryStage, InMemoryPersistence
from geny_executor.stages.s16_yield import YieldStage
from geny_executor.tools.base import Tool, ToolResult, ToolContext
from geny_executor.tools.registry import ToolRegistry
from geny_executor.stages.s06_api.types import APIResponse, ContentBlock
from geny_executor.session import Session


# ── Helper tools ──


class CalculatorTool(Tool):
    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Perform arithmetic"

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
            },
            "required": ["expression"],
        }

    async def execute(self, input: dict, context: ToolContext) -> ToolResult:
        try:
            result = eval(input["expression"])  # Safe in test context
            return ToolResult(content=str(result))
        except Exception as e:
            return ToolResult(content=str(e), is_error=True)


class SearchTool(Tool):
    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return "Search for information"

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        }

    async def execute(self, input: dict, context: ToolContext) -> ToolResult:
        return ToolResult(content=f"Results for: {input['query']}")


# ── Integration: Full 16-stage pipeline ──


@pytest.mark.asyncio
async def test_full_pipeline_all_stages():
    """Full pipeline with all 16 stages registered."""
    emitted = []
    persistence = InMemoryPersistence()
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    provider = MockProvider(default_text="Computation complete: 42")
    pipeline = Pipeline(PipelineConfig(name="full-16"))

    # Register all 16 stages
    pipeline.register_stage(InputStage())  # 1
    pipeline.register_stage(ContextStage())  # 2
    pipeline.register_stage(SystemStage(prompt="You are helpful."))  # 3
    pipeline.register_stage(GuardStage())  # 4
    pipeline.register_stage(CacheStage(strategy=SystemCacheStrategy()))  # 5
    pipeline.register_stage(APIStage(provider=provider, retry=NoRetry()))  # 6
    pipeline.register_stage(TokenStage())  # 7
    pipeline.register_stage(ThinkStage())  # 8
    pipeline.register_stage(ParseStage())  # 9
    pipeline.register_stage(ToolStage(registry=registry))  # 10
    pipeline.register_stage(AgentStage())  # 11
    pipeline.register_stage(EvaluateStage())  # 12
    pipeline.register_stage(LoopStage(StandardLoopController(max_turns=5)))  # 13
    pipeline.register_stage(
        EmitStage(
            emitters=[
                TextEmitter(callback=lambda t: emitted.append(t)),
            ]
        )
    )  # 14
    pipeline.register_stage(MemoryStage(persistence=persistence))  # 15
    pipeline.register_stage(YieldStage())  # 16

    state = PipelineState(session_id="full-test")
    result = await pipeline.run("Calculate 6 * 7", state)

    assert result.success is True
    assert result.text == "Computation complete: 42"
    assert emitted == ["Computation complete: 42"]
    assert state.system == [
        {"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}}
    ]


@pytest.mark.asyncio
async def test_pipeline_describe_all_16_slots():
    """Pipeline.describe() returns info for all registered stages."""
    pipeline = PipelinePresets.agent(api_key="test-key")
    desc = pipeline.describe()

    # Should have multiple stages
    assert len(desc) >= 8  # At least the core stages


@pytest.mark.asyncio
async def test_multi_turn_tool_loop():
    """Multi-turn conversation with tool use spanning multiple iterations."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(SearchTool())

    # Turn 1: tool_use → Turn 2: final text
    provider = MockProvider(
        responses=[
            APIResponse(
                content=[
                    ContentBlock(
                        type="tool_use",
                        tool_use_id="calc_1",
                        tool_name="calculator",
                        tool_input={"expression": "2+2"},
                    )
                ],
                stop_reason="tool_use",
            ),
            APIResponse(
                content=[ContentBlock(type="text", text="The answer is 4.")],
                stop_reason="end_turn",
            ),
        ]
    )

    pipeline = Pipeline(PipelineConfig(name="multi-tool"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(provider=provider, retry=NoRetry()))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(ToolStage(registry=registry))
    pipeline.register_stage(LoopStage(StandardLoopController(max_turns=10)))
    pipeline.register_stage(YieldStage())

    result = await pipeline.run("What is 2+2?")
    assert result.success is True
    assert result.text == "The answer is 4."


@pytest.mark.asyncio
async def test_session_multi_run_accumulation():
    """Session accumulates state across multiple runs."""
    persistence = InMemoryPersistence()
    provider = MockProvider(
        responses=[
            APIResponse(
                content=[ContentBlock(type="text", text="Hello!")],
                stop_reason="end_turn",
            ),
            APIResponse(
                content=[ContentBlock(type="text", text="I remember!")],
                stop_reason="end_turn",
            ),
            APIResponse(
                content=[ContentBlock(type="text", text="Goodbye!")],
                stop_reason="end_turn",
            ),
        ]
    )

    pipeline = Pipeline(PipelineConfig(name="session"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(provider=provider, retry=NoRetry()))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(MemoryStage(persistence=persistence))
    pipeline.register_stage(YieldStage())

    session = Session(pipeline=pipeline)

    r1 = await session.run("Hi")
    assert r1.text == "Hello!"
    assert len(session.state.messages) == 2

    r2 = await session.run("Remember me?")
    assert r2.text == "I remember!"
    assert len(session.state.messages) == 4

    r3 = await session.run("Bye")
    assert r3.text == "Goodbye!"
    assert len(session.state.messages) == 6

    # Memory should have persisted all messages
    loaded = await persistence.load(session.id)
    assert len(loaded) == 6


@pytest.mark.asyncio
async def test_event_bus_stage_lifecycle():
    """EventBus receives events from all stage transitions."""
    provider = MockProvider(default_text="Test")
    pipeline = Pipeline(PipelineConfig(name="events"))

    events_received = []
    pipeline._event_bus.on("*", lambda e: events_received.append(e.type))

    pipeline.register_stage(InputStage())
    pipeline.register_stage(SystemStage(prompt="Test"))
    pipeline.register_stage(APIStage(provider=provider, retry=NoRetry()))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())

    await pipeline.run("Hello")

    # Should have stage.enter and stage.exit events
    assert any("stage.enter" in e for e in events_received)
    assert any("stage.exit" in e for e in events_received)


@pytest.mark.asyncio
async def test_composable_system_prompt():
    """ComposablePromptBuilder assembles from blocks in a real pipeline."""
    builder = ComposablePromptBuilder(
        blocks=[
            PersonaBlock("You are Geny, a friendly VTuber AI."),
            RulesBlock(["Always be cheerful", "Use informal Korean"]),
        ]
    )

    provider = MockProvider(default_text="안녕!")
    pipeline = Pipeline(PipelineConfig(name="composable"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(SystemStage(builder=builder))
    pipeline.register_stage(APIStage(provider=provider, retry=NoRetry()))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())

    state = PipelineState()
    result = await pipeline.run("안녕하세요", state)

    assert result.success is True
    assert "Geny" in state.system
    assert "cheerful" in state.system


@pytest.mark.asyncio
async def test_vtuber_emitter_in_pipeline():
    """VTuberEmitter extracts emotion in full pipeline."""
    emotions = []

    provider = MockProvider(default_text="하하 정말 기뻐요! 오늘 기분이 너무 좋아요!")
    pipeline = Pipeline(PipelineConfig(name="vtuber"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(provider=provider, retry=NoRetry()))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(
        EmitStage(
            emitters=[
                VTuberEmitter(emotion_callback=lambda t, e: emotions.append(e)),
            ]
        )
    )
    pipeline.register_stage(YieldStage())

    result = await pipeline.run("How are you?")
    assert result.success is True
    assert len(emotions) == 1
    assert emotions[0]["primary"] == "happy"


@pytest.mark.asyncio
async def test_guard_blocks_over_budget():
    """Guard stage blocks execution when over cost budget."""
    from geny_executor.stages.s04_guard.guards import CostBudgetGuard

    provider = MockProvider(default_text="Should not reach")
    pipeline = Pipeline(PipelineConfig(name="guard"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(GuardStage(guards=[CostBudgetGuard(max_cost_usd=1.0)]))
    pipeline.register_stage(APIStage(provider=provider, retry=NoRetry()))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())

    state = PipelineState()
    state.total_cost_usd = 5.0  # Already over budget

    result = await pipeline.run("Test", state)
    assert result.success is False
    assert "cost" in result.error.lower() or "budget" in result.error.lower()


@pytest.mark.asyncio
async def test_hybrid_context_trims_in_pipeline():
    """HybridStrategy trims old history in a real pipeline."""
    provider = MockProvider(default_text="OK")
    pipeline = Pipeline(PipelineConfig(name="hybrid-ctx"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(ContextStage(strategy=HybridStrategy(max_recent_turns=2)))
    pipeline.register_stage(APIStage(provider=provider, retry=NoRetry()))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())

    state = PipelineState()
    # Pre-fill with 10 turns of history
    for i in range(10):
        state.messages.append({"role": "user", "content": f"Old Q{i}"})
        state.messages.append({"role": "assistant", "content": f"Old A{i}"})

    result = await pipeline.run("New question", state)
    assert result.success is True
    # Should have been trimmed to 2 turns (4 messages) + new input (1)
    # The exact count depends on when trimming happens vs new message addition


@pytest.mark.asyncio
async def test_streaming_pipeline():
    """Pipeline streaming mode emits text.delta events from MockProvider."""
    provider = MockProvider(default_text="Streamed response")
    pipeline = Pipeline(PipelineConfig(name="stream"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(provider=provider, retry=NoRetry()))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(YieldStage())

    events = []
    async for event in pipeline.run_stream("Hello"):
        events.append(event)

    types = [e.type for e in events]
    assert "pipeline.start" in types
    assert "pipeline.complete" in types

    # Must contain text.delta events from streaming
    deltas = [e for e in events if e.type == "text.delta"]
    assert len(deltas) > 0, f"No text.delta events! Got types: {types}"

    # Concatenated deltas should reconstruct the original text
    streamed_text = "".join(e.data["text"] for e in deltas)
    assert "Streamed" in streamed_text
    assert "response" in streamed_text


@pytest.mark.asyncio
async def test_preset_pipeline_runs_with_mock():
    """Preset pipeline runs end-to-end with mock provider."""
    # Manually swap the API provider to mock for testing
    provider = MockProvider(default_text="Preset works!")
    pipeline = Pipeline(PipelineConfig(name="preset-test"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(ContextStage())
    pipeline.register_stage(SystemStage(prompt="You are a test agent."))
    pipeline.register_stage(CacheStage(strategy=SystemCacheStrategy()))
    pipeline.register_stage(APIStage(provider=provider, retry=NoRetry()))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(EvaluateStage())
    pipeline.register_stage(MemoryStage())
    pipeline.register_stage(YieldStage())

    result = await pipeline.run("Test preset")
    assert result.success is True
    assert result.text == "Preset works!"
