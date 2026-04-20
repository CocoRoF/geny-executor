"""Phase 5 tests — Emit, Presets, MCP, Builder integration."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor import Pipeline, PipelineConfig, PipelineState
from geny_executor.stages.s01_input import InputStage
from geny_executor.stages.s06_api import APIStage, MockProvider
from geny_executor.stages.s06_api.retry import NoRetry
from geny_executor.stages.s09_parse import ParseStage
from geny_executor.stages.s16_yield import YieldStage

# Emit imports
from geny_executor.stages.s14_emit import (
    EmitStage,
    TextEmitter,
    CallbackEmitter,
    VTuberEmitter,
    EmitterChain,
)

# Presets
from geny_executor.core.presets import PipelinePresets

# Builder
from geny_executor.core.builder import PipelineBuilder

# MCP
from geny_executor.tools.mcp import MCPManager, MCPServerConfig, MCPToolAdapter


# ── Emit Stage ──


@pytest.mark.asyncio
async def test_emit_bypass_no_emitters():
    """EmitStage bypasses when no emitters registered."""
    stage = EmitStage()
    state = PipelineState()
    assert stage.should_bypass(state) is True


@pytest.mark.asyncio
async def test_emit_text_emitter():
    """TextEmitter delivers text via callback."""
    received = []
    emitter = TextEmitter(callback=lambda t: received.append(t))

    state = PipelineState()
    state.final_text = "Hello world"

    result = await emitter.emit(state)
    assert result.emitted is True
    assert received == ["Hello world"]


@pytest.mark.asyncio
async def test_emit_callback_emitter():
    """CallbackEmitter delivers full state."""
    received = []
    emitter = CallbackEmitter(callback=lambda s: received.append(s.final_text))

    state = PipelineState()
    state.final_text = "Full state access"

    await emitter.emit(state)
    assert received == ["Full state access"]


@pytest.mark.asyncio
async def test_emit_vtuber_emitter_emotion():
    """VTuberEmitter extracts emotion from text."""
    emotions = []
    emitter = VTuberEmitter(emotion_callback=lambda t, e: emotions.append(e))

    state = PipelineState()
    state.final_text = "하하 기뻐요! 정말 좋아요!"

    result = await emitter.emit(state)
    assert result.emitted is True
    assert len(emotions) == 1
    assert emotions[0]["primary"] == "happy"


@pytest.mark.asyncio
async def test_emit_vtuber_neutral():
    """VTuberEmitter defaults to neutral for plain text."""
    emitter = VTuberEmitter()
    state = PipelineState()
    state.final_text = "The database schema has three tables."

    result = await emitter.emit(state)
    assert result.metadata["emotion"]["primary"] == "neutral"


@pytest.mark.asyncio
async def test_emit_chain():
    """EmitterChain runs all emitters."""
    text_received = []
    callback_received = []

    chain = EmitterChain(
        [
            TextEmitter(callback=lambda t: text_received.append(t)),
            CallbackEmitter(callback=lambda s: callback_received.append(s.final_text)),
        ]
    )

    state = PipelineState()
    state.final_text = "Chain test"

    results = await chain.emit_all(state)
    assert len(results) == 2
    assert text_received == ["Chain test"]
    assert callback_received == ["Chain test"]


@pytest.mark.asyncio
async def test_emit_stage_full_pipeline():
    """EmitStage works in a full pipeline."""
    emitted = []
    provider = MockProvider(default_text="Pipeline emit test")
    pipeline = Pipeline(PipelineConfig(name="emit"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(provider=provider, retry=NoRetry()))
    pipeline.register_stage(ParseStage())
    pipeline.register_stage(
        EmitStage(
            emitters=[
                TextEmitter(callback=lambda t: emitted.append(t)),
            ]
        )
    )
    pipeline.register_stage(YieldStage())

    result = await pipeline.run("Test emit")
    assert result.success is True
    assert emitted == ["Pipeline emit test"]


# ── Presets ──


def test_preset_minimal():
    """Minimal preset creates a valid pipeline."""
    pipeline = PipelinePresets.minimal(api_key="test-key")
    desc = pipeline.describe()
    stage_names = [s.name for s in desc]
    assert "input" in stage_names
    assert "api" in stage_names
    assert "parse" in stage_names
    assert "yield" in stage_names


def test_preset_chat():
    """Chat preset includes context, system, loop, memory."""
    pipeline = PipelinePresets.chat(api_key="test-key", system_prompt="Hi")
    desc = pipeline.describe()
    stage_names = [s.name for s in desc]
    assert "context" in stage_names
    assert "system" in stage_names
    assert "loop" in stage_names
    assert "memory" in stage_names


def test_preset_agent():
    """Agent preset includes all core stages."""
    pipeline = PipelinePresets.agent(api_key="test-key")
    desc = pipeline.describe()
    stage_names = [s.name for s in desc]
    assert "think" in stage_names
    assert "evaluate" in stage_names
    assert "loop" in stage_names


def test_preset_evaluator():
    """Evaluator preset is lightweight."""
    pipeline = PipelinePresets.evaluator(api_key="test-key")
    desc = pipeline.describe()
    stage_names = [s.name for s in desc]
    active_names = [s.name for s in desc if s.is_active]
    assert "system" in stage_names
    assert "evaluate" in stage_names
    # loop and memory should NOT be actively registered
    assert "loop" not in active_names
    assert "memory" not in active_names


# ── Builder ──


def test_builder_fluent_api():
    """PipelineBuilder fluent API creates pipeline."""
    pipeline = (
        PipelineBuilder("test", api_key="test-key")
        .with_system(prompt="You are helpful.")
        .with_cache(strategy="system")
        .with_think()
        .with_evaluate()
        .with_loop(max_turns=10)
        .build()
    )
    desc = pipeline.describe()
    stage_names = [s.name for s in desc]
    assert "system" in stage_names
    assert "cache" in stage_names
    assert "think" in stage_names
    assert "evaluate" in stage_names
    assert "loop" in stage_names


def test_builder_with_emit():
    """Builder supports emit configuration."""
    pipeline = (
        PipelineBuilder("emit-test", api_key="test-key").with_emit(emitters=[TextEmitter()]).build()
    )
    desc = pipeline.describe()
    stage_names = [s.name for s in desc]
    assert "emit" in stage_names


# ── MCP ──


def test_mcp_server_config():
    """MCPServerConfig stores configuration."""
    config = MCPServerConfig(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    )
    assert config.name == "filesystem"
    assert config.transport == "stdio"


def test_mcp_manager_list_servers():
    """MCPManager tracks connected servers."""
    manager = MCPManager()
    assert manager.list_servers() == []


@pytest.mark.asyncio
async def test_mcp_manager_connect_disconnect():
    """MCPManager connect/disconnect lifecycle."""
    manager = MCPManager()
    config = MCPServerConfig(name="test", command="echo")

    await manager.connect("test", config)
    assert manager.is_connected("test")

    await manager.disconnect("test")
    assert not manager.is_connected("test")


@pytest.mark.asyncio
async def test_mcp_tool_adapter():
    """MCPToolAdapter wraps MCP tool definition."""
    from geny_executor.tools.mcp.manager import MCPServerConnection

    config = MCPServerConfig(name="test", command="echo")
    conn = MCPServerConnection(config)
    await conn.connect()

    adapter = MCPToolAdapter(
        server=conn,
        definition={
            "name": "read_file",
            "description": "Read a file",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
            },
        },
    )

    # v0.22.0: MCP tools are always namespaced as mcp__{server}__{tool}
    assert adapter.name == "mcp__test__read_file"
    assert adapter.raw_name == "read_file"
    assert adapter.server_name == "test"
    assert adapter.description == "Read a file"
    api_format = adapter.to_api_format()
    assert api_format["name"] == "mcp__test__read_file"
    assert "properties" in api_format["input_schema"]
