"""Self-modifying environment — pipeline wiring + live refresh.

Verifies that ``from_manifest_async`` builds a ``PipelineEnvironment``
controller, injects it into the Tool stage's ToolContext (so the built-in
``env`` tool can reach it), and that edits made through it take effect: the
tool registry version bumps (Stage 3 re-derives the tools array next turn) and
the prompt builder is editable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from geny_executor.core.environment import (
    EnvironmentManifest,
    StageManifestEntry,
    ToolsSnapshot,
)
from geny_executor.core.pipeline import Pipeline
from geny_executor.stages.s03_system.builders import MutablePromptBuilder
from geny_executor.tools.base import Tool, ToolContext, ToolResult
from tests._fixtures.manifest_entries import required_stage_entries


class _NamedTool(Tool):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"available:{self._name}"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult(content=self._name)


class _Provider:
    def __init__(self, tools: Dict[str, Tool]) -> None:
        self._tools = tools

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)


def _manifest(built_in: List[str], external: List[str]) -> EnvironmentManifest:
    # required stages (input/api/parse/yield) + system (so there's a prompt
    # builder) + tool (so the env controller can be injected into its context).
    stages = required_stage_entries() + [
        StageManifestEntry(
            order=3, name="system", active=True, config={"prompt": "base prompt"}
        ).to_dict(),
        StageManifestEntry(order=10, name="tool", active=True).to_dict(),
    ]
    return EnvironmentManifest(
        stages=stages,
        tools=ToolsSnapshot(built_in=list(built_in), external=list(external)),
    )


async def _build(provider: _Provider) -> Pipeline:
    return await Pipeline.from_manifest_async(
        _manifest(built_in=["env"], external=["read"]),
        api_key="sk-test",
        adhoc_providers=[provider],
        strict=False,
    )


@pytest.mark.asyncio
async def test_pipeline_builds_and_injects_env_controller() -> None:
    prov = _Provider({"read": _NamedTool("read"), "web_search": _NamedTool("web_search")})
    pipeline = await _build(prov)

    env = pipeline.environment
    assert env is not None, "pipeline should build a PipelineEnvironment controller"

    # Injected into the Tool stage's ToolContext.
    tool_stage = next(s for s in pipeline._stages.values() if getattr(s, "name", "") == "tool")
    assert getattr(tool_stage._context, "environment", None) is env

    snap = env.snapshot()
    assert "env" in snap["active_tools"]                 # built-in env tool active
    assert "read" in snap["active_tools"]                # external read selected
    assert "web_search" in snap["available_tools"]       # available, not active


@pytest.mark.asyncio
async def test_enable_tool_bumps_registry_version() -> None:
    prov = _Provider({"read": _NamedTool("read"), "web_search": _NamedTool("web_search")})
    pipeline = await _build(prov)
    env = pipeline.environment

    v0 = pipeline.tool_registry.version
    ok, _ = env.enable_tool("web_search")
    assert ok
    assert pipeline.tool_registry.version > v0          # → Stage 3 rebuilds next turn
    assert "web_search" in env.active_tools()


@pytest.mark.asyncio
async def test_attach_mutable_prompt_makes_prompt_editable() -> None:
    prov = _Provider({"read": _NamedTool("read")})
    pipeline = await _build(prov)
    # Host installs a mutable prompt builder (Geny does this at session setup).
    pipeline.attach_runtime(system_builder=MutablePromptBuilder("base prompt"))
    env = pipeline.environment

    assert env.get_prompt() == "base prompt"
    ok, _ = env.set_prompt("rewritten prompt")
    assert ok
    assert env.get_prompt() == "rewritten prompt"
    # The Stage 3 builder is the one we edit.
    assert pipeline._current_system_builder().build(None) == "rewritten prompt"


@pytest.mark.asyncio
async def test_env_save_calls_host_persistence_with_overlay() -> None:
    prov = _Provider({"read": _NamedTool("read"), "web_search": _NamedTool("web_search")})
    pipeline = await _build(prov)
    pipeline.attach_runtime(system_builder=MutablePromptBuilder("base prompt"))

    saved: Dict[str, Any] = {}

    async def _persist(overlay: Dict[str, Any]) -> None:
        saved.update(overlay)

    pipeline.attach_runtime(env_persistence=_persist)
    env = pipeline.environment

    env.set_prompt("evolved prompt")
    env.enable_tool("web_search")
    ok, _ = await env.save()
    assert ok
    assert saved["prompt"] == "evolved prompt"
    assert "web_search" in saved["active_tools"]
    actions = {c["action"] for c in saved["changelog"]}
    assert {"set_prompt", "enable_tool"} <= actions
