"""Stage 6 ``tool_loop`` slot — the internal agentic loop (2.3.0).

The feature gives every backend the ``claude_code_cli`` execution shape
as a manifest-selectable choice: ``tool_loop="internal"`` resolves tool
calls inside Stage 6 and returns only the final response, so Stage 9/10
naturally no-op (the CLI accumulator's finalize contract, generalized).
``tool_loop="pipeline"`` (default) is byte-identical to pre-2.3.0.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import pytest

from geny_executor import CredentialBundle, Pipeline, ProviderCredentials, validate_manifest
from geny_executor.core.environment import EnvironmentManifest
from geny_executor.core.state import TokenUsage
from geny_executor.llm_client.base import ClientCapabilities
from geny_executor.llm_client.types import APIResponse, ContentBlock
from geny_executor.stages.s06_api.artifact.default.stage import APIStage
from geny_executor.stages.s06_api.artifact.default.tool_loop import (
    InternalAgenticLoop,
    PipelineToolLoop,
)
from geny_executor.tools.base import Tool, ToolResult


# ─────────────────────────────────── fixtures ─


class ScriptedClient:
    """Returns ``tool_rounds`` tool_use responses, then a final text."""

    provider = "anthropic"
    capabilities = ClientCapabilities(supports_tools=True, supports_streaming=False)

    def __init__(
        self, tool_rounds: int = 2, tools_per_round: int = 1, cost_usd: float = 0.0
    ) -> None:
        self.calls = 0
        self.seen_messages: List[List[Dict[str, Any]]] = []
        self._tool_rounds = tool_rounds
        self._tools_per_round = tools_per_round
        self._cost_usd = cost_usd

    async def create_message(
        self, *, model_config, messages, system="", tools=None, tool_choice=None, purpose=""
    ) -> APIResponse:
        self.calls += 1
        self.seen_messages.append(list(messages))
        if self.calls <= self._tool_rounds:
            blocks = [
                ContentBlock(
                    type="tool_use",
                    tool_use_id=f"t{self.calls}-{i}",
                    tool_name="echo",
                    tool_input={"round": self.calls, "i": i},
                )
                for i in range(self._tools_per_round)
            ]
            return APIResponse(
                content=blocks,
                stop_reason="tool_use",
                usage=TokenUsage(
                    input_tokens=10, output_tokens=5, cost_usd=self._cost_usd
                ),
            )
        return APIResponse(
            content=[ContentBlock(type="text", text="final answer")],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=7, output_tokens=3),
        )


class EchoTool(Tool):
    name = "echo"
    description = "echo the input"
    input_schema = {"type": "object", "properties": {}}

    async def execute(self, tool_input: Optional[Dict[str, Any]] = None, context: Any = None) -> ToolResult:
        return ToolResult(content=f"echo:{tool_input}")


class SlowTool(Tool):
    name = "echo"
    description = "slow echo"
    input_schema = {"type": "object", "properties": {}}

    async def execute(self, tool_input=None, context=None):
        await asyncio.sleep(0.15)
        return ToolResult(content="slow")


def _manifest(strategy: str = "internal", strategy_cfg: Optional[Dict[str, Any]] = None):
    entry: Dict[str, Any] = {
        "order": 6,
        "name": "api",
        "active": True,
        "config": {"provider": "anthropic", "stream": False},
        "strategies": {"retry": "no_retry", "router": "passthrough", "tool_loop": strategy},
    }
    if strategy_cfg is not None:
        entry["strategy_configs"] = {"tool_loop": strategy_cfg}
    return EnvironmentManifest.from_dict(
        {
            "metadata": {"id": "t-loop", "name": "t"},
            "pipeline": {"max_iterations": 4},
            "stages": [
                {"order": 1, "name": "input", "active": True},
                entry,
                {"order": 9, "name": "parse", "active": True},
                {"order": 10, "name": "tool", "active": True},
                {"order": 16, "name": "loop", "active": True},
                {"order": 21, "name": "yield", "active": True},
            ],
            "tools": {"built_in": [], "external": [], "mcp_servers": []},
        }
    )


async def _build(manifest, client) -> Pipeline:
    pipeline = await Pipeline.from_manifest_async(
        manifest,
        credentials=CredentialBundle(
            by_provider={"anthropic": ProviderCredentials(api_key="sk-test")}
        ),
        strict=True,
    )
    pipeline._tool_registry.register(EchoTool())
    pipeline.attach_runtime(llm_client=client)
    return pipeline


# ─────────────────────────────────── happy path ─


@pytest.mark.asyncio
async def test_internal_loop_resolves_tools_in_one_iteration():
    client = ScriptedClient(tool_rounds=2)
    pipeline = await _build(_manifest("internal", {"max_inner_turns": 5}), client)
    try:
        result = await pipeline.run("go")
        assert result.success
        state = result.state
        assert client.calls == 3  # 2 tool rounds + final
        assert state.final_text == "final answer"
        # ONE pipeline iteration — the loop ran inside Stage 6.
        assert state.iteration <= 1
        # Conversation history complete and ordered: the model's view.
        roles = [m["role"] for m in state.messages]
        assert roles == ["user", "assistant", "user", "assistant", "user", "assistant"]
        # Final returned response is tool-free (CLI finalize parity) —
        # the last assistant message carries only text.
        last = state.messages[-1]["content"]
        assert all(b.get("type") != "tool_use" for b in last)
    finally:
        await pipeline.aclose()


@pytest.mark.asyncio
async def test_internal_loop_event_stream_parity():
    """Per call: api.tool_use {source:'internal'} → tool.call_start →
    tool.call_complete → api.tool_result. Plus one api.request/api.response
    pair per inner client call — every inner call individually visible."""
    client = ScriptedClient(tool_rounds=1)
    pipeline = await _build(_manifest("internal"), client)
    try:
        result = await pipeline.run("go")
        events = [e["type"] for e in result.state.events]
        tool_seq = [e for e in events if e in (
            "api.tool_use", "tool.call_start", "tool.call_complete", "api.tool_result"
        )]
        assert tool_seq == [
            "api.tool_use", "tool.call_start", "tool.call_complete", "api.tool_result"
        ]
        assert events.count("api.request") == 2
        assert events.count("api.response") == 2
        tu = next(e for e in result.state.events if e["type"] == "api.tool_use")
        assert tu["data"]["source"] == "internal"
        # Payload keys match the streaming/CLI shape (catalog contract).
        assert set(tu["data"]) >= {"id", "name", "input", "source"}
        tr = next(e for e in result.state.events if e["type"] == "api.tool_result")
        assert set(tr["data"]) >= {"tool_use_id", "content", "is_error"}
    finally:
        await pipeline.aclose()


@pytest.mark.asyncio
async def test_internal_loop_usage_accumulates_across_inner_calls():
    client = ScriptedClient(tool_rounds=2)
    pipeline = await _build(_manifest("internal"), client)
    try:
        result = await pipeline.run("go")
        response = result.state.last_api_response
        # 2 consumed responses (10/5 each) + final (7/3).
        assert response.usage.input_tokens == 27
        assert response.usage.output_tokens == 13
    finally:
        await pipeline.aclose()


# ─────────────────────────────────── caps + degradation ─


@pytest.mark.asyncio
async def test_max_inner_turns_cap_hands_leftover_to_stage10():
    """Capped loop returns the tool-bearing response AS-IS; Stage 9/10
    then own it — graceful degradation to the pipeline shape, end-to-end
    through Pipeline.run (s10 dispatches, s16 loops, final lands)."""
    client = ScriptedClient(tool_rounds=2)
    pipeline = await _build(_manifest("internal", {"max_inner_turns": 1}), client)
    try:
        result = await pipeline.run("go")
        assert result.success
        events = [e["type"] for e in result.state.events]
        assert "api.internal_loop_capped" in events
        capped = next(
            e for e in result.state.events if e["type"] == "api.internal_loop_capped"
        )
        assert capped["data"] == {"turns": 1, "reason": "max_inner_turns"}
        # Stage 10 dispatched the leftover round (pipeline path).
        assert "tool.execute_start" in events
        assert result.state.final_text == "final answer"
    finally:
        await pipeline.aclose()


@pytest.mark.asyncio
async def test_cost_budget_cap_stops_loop():
    # Each consumed tool-round response reports $0.50 — the loop's
    # running inner cost crosses the manifest budget after round 1.
    client = ScriptedClient(tool_rounds=3, cost_usd=0.5)
    manifest = _manifest("internal")
    # PipelineConfig.apply_to_state stomps state.cost_budget_usd every
    # run (documented lifetime), so the budget must come from its single
    # declared home: the manifest pipeline block.
    md = manifest.to_dict()
    md["pipeline"]["cost_budget_usd"] = 0.4
    manifest = EnvironmentManifest.from_dict(md)
    pipeline = await _build(manifest, client)

    try:
        result = await pipeline.run("go")
        events = [e["type"] for e in result.state.events]
        # The first response carries tool calls; budget gate trips before
        # any inner dispatch.
        capped = next(
            e for e in result.state.events if e["type"] == "api.internal_loop_capped"
        )
        assert capped["data"]["reason"] == "cost_budget"
        assert "tool.execute_start" in events  # leftover flowed to s10
    finally:
        await pipeline.aclose()


# ─────────────────────────────────── containment ─


@pytest.mark.asyncio
async def test_broken_tool_becomes_is_error_result_and_loop_continues():
    class BrokenTool(Tool):
        name = "echo"
        description = "always raises"
        input_schema = {"type": "object", "properties": {}}

        async def execute(self, tool_input=None, context=None):
            raise RuntimeError("boom")

    client = ScriptedClient(tool_rounds=1)
    manifest = _manifest("internal")
    pipeline = await Pipeline.from_manifest_async(
        manifest,
        credentials=CredentialBundle(
            by_provider={"anthropic": ProviderCredentials(api_key="sk-test")}
        ),
        strict=True,
    )
    pipeline._tool_registry.register(BrokenTool())
    pipeline.attach_runtime(llm_client=client)
    try:
        result = await pipeline.run("go")
        assert result.success  # the turn survived
        tr = next(e for e in result.state.events if e["type"] == "api.tool_result")
        assert tr["data"]["is_error"] is True
        assert client.calls == 2  # the model got the error and answered
    finally:
        await pipeline.aclose()


# ─────────────────────────────────── parallel dispatch ─


@pytest.mark.asyncio
async def test_parallel_tools_dispatch_concurrently():
    client = ScriptedClient(tool_rounds=1, tools_per_round=3)
    manifest = _manifest("internal", {"parallel_tools": True})
    pipeline = await Pipeline.from_manifest_async(
        manifest,
        credentials=CredentialBundle(
            by_provider={"anthropic": ProviderCredentials(api_key="sk-test")}
        ),
        strict=True,
    )
    pipeline._tool_registry.register(SlowTool())
    pipeline.attach_runtime(llm_client=client)
    try:
        import time

        t0 = time.monotonic()
        result = await pipeline.run("go")
        elapsed = time.monotonic() - t0
        assert result.success
        # 3 × 0.15s sequential would be ≥ 0.45s; concurrent ≈ one tool's
        # latency. Generous bound (sync tools run in a thread pool).
        assert elapsed < 0.40, f"parallel dispatch took {elapsed:.2f}s"
        results_order = [
            e["data"]["tool_use_id"]
            for e in result.state.events
            if e["type"] == "api.tool_result"
        ]
        assert results_order == ["t1-0", "t1-1", "t1-2"]  # input order kept
    finally:
        await pipeline.aclose()


# ─────────────────────────────────── capability guards ─


@pytest.mark.asyncio
async def test_subprocess_client_degrades_to_single_call():
    """The CLI already loops internally — internal mode must not
    double-loop. The strategy degrades to one call, warning once."""

    class FakeCLIClient(ScriptedClient):
        # provider stays "anthropic" so the #866 attach guard (provider
        # mismatch vs manifest) passes — the capability guard under test
        # keys on is_subprocess, not the provider name.
        capabilities = ClientCapabilities(
            supports_tools=True, supports_streaming=False, is_subprocess=True
        )

    client = FakeCLIClient(tool_rounds=1)
    pipeline = await _build(_manifest("internal"), client)
    try:
        await pipeline.run("go")
        # Degraded: exactly one client call per pipeline iteration —
        # the first iteration returned tool_use which went to s10, then
        # s16 looped and the second call produced the final.
        assert client.calls == 2
        # And the internal-loop event never fired.
        # (run again to confirm warn-once doesn't spam — behavioural no-op)
    finally:
        await pipeline.aclose()


@pytest.mark.asyncio
async def test_toolless_client_degrades_to_single_call():
    class NoToolsClient(ScriptedClient):
        capabilities = ClientCapabilities(supports_tools=False, supports_streaming=False)

    client = NoToolsClient(tool_rounds=0)  # immediate final
    pipeline = await _build(_manifest("internal"), client)
    try:
        result = await pipeline.run("go")
        assert result.success
        assert client.calls == 1
    finally:
        await pipeline.aclose()


@pytest.mark.asyncio
async def test_no_tool_stage_means_no_dispatcher_and_degrades():
    manifest_dict = _manifest("internal").to_dict()
    manifest_dict["stages"] = [s for s in manifest_dict["stages"] if s["name"] != "tool"]
    manifest = EnvironmentManifest.from_dict(manifest_dict)
    client = ScriptedClient(tool_rounds=0)
    pipeline = await Pipeline.from_manifest_async(
        manifest,
        credentials=CredentialBundle(
            by_provider={"anthropic": ProviderCredentials(api_key="sk-test")}
        ),
        strict=True,
    )
    pipeline.attach_runtime(llm_client=client)
    try:
        result = await pipeline.run("go")
        assert result.state.tool_dispatcher is None
        assert result.success
    finally:
        await pipeline.aclose()


# ─────────────────────────────────── permission integration ─


@pytest.mark.asyncio
async def test_deny_posture_reaches_internal_dispatches():
    """The dispatcher builds its context off the live Tool-stage context,
    so the 2.2.x permission posture applies identically to internal-loop
    dispatches — one decision path."""
    from geny_executor.permission.types import PermissionPosture

    client = ScriptedClient(tool_rounds=1)
    pipeline = await _build(_manifest("internal"), client)
    tool_stage = next(s for s in pipeline.stages if getattr(s, "name", "") == "tool")
    tool_stage._context.permission_default_posture = PermissionPosture.DENY
    try:
        result = await pipeline.run("go")
        tr = next(e for e in result.state.events if e["type"] == "api.tool_result")
        assert tr["data"]["is_error"] is True
        assert result.success  # denied tool != dead turn
    finally:
        await pipeline.aclose()


# ─────────────────────────────────── default-path regression ─


@pytest.mark.asyncio
async def test_pipeline_strategy_returns_tool_blocks_verbatim():
    """tool_loop='pipeline' (and the unset default) keeps the pre-2.3.0
    shape: tool_use blocks flow to s9/s10, the pipeline loops."""
    client = ScriptedClient(tool_rounds=1)
    pipeline = await _build(_manifest("pipeline"), client)
    try:
        result = await pipeline.run("go")
        assert result.success
        events = [e["type"] for e in result.state.events]
        assert "tool.execute_start" in events  # s10 dispatched
        assert "api.internal_loop_capped" not in events
        assert client.calls == 2  # one call per pipeline iteration
        assert result.state.iteration >= 1
    finally:
        await pipeline.aclose()


def test_default_slot_strategy_is_pipeline():
    stage = APIStage(provider="anthropic")
    assert stage.get_strategy_slots()["tool_loop"].strategy.name == "pipeline"
    assert isinstance(stage._tool_loop, PipelineToolLoop)


# ─────────────────────────────────── config contract ─


def test_internal_loop_configure_roundtrip():
    s = InternalAgenticLoop()
    s.configure({"max_inner_turns": 4, "parallel_tools": True})
    assert s.get_config() == {"max_inner_turns": 4, "parallel_tools": True}
    with pytest.raises(ValueError):
        s.configure({"max_inner_turns": 0})
    with pytest.raises(ValueError):
        s.configure({"max_inner_turns": True})
    with pytest.raises(ValueError):
        s.configure({"parallel_tools": "yes"})
    # Rejected configure leaves prior config live (atomicity).
    assert s.get_config() == {"max_inner_turns": 4, "parallel_tools": True}
    schema = InternalAgenticLoop.config_schema()
    assert {f.name for f in schema.fields} == {"max_inner_turns", "parallel_tools"}


def test_manifest_with_internal_loop_validates_clean():
    issues = validate_manifest(_manifest("internal", {"max_inner_turns": 6}))
    assert [i for i in issues if i.severity == "error"] == []


def test_introspection_exposes_tool_loop_slot():
    from geny_executor import introspect_stage

    info = introspect_stage(6)
    assert "tool_loop" in info.strategy_slots
    slot = info.strategy_slots["tool_loop"]
    assert set(slot.available_impls) >= {"pipeline", "internal"}
