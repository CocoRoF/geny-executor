"""TTFT program (2.50.0) — prompt-cache overhaul + api.ttft probe.

Pins the four findings of the 2026-07-12 TTFT audit, group A:

A1  cache gate is provider-based — alias models ("opus"/"sonnet") no
    longer silently disable ALL caching; non-anthropic providers are
    still bypassed.
A2  volatile prompt blocks (clock, retrieved memory) leave the cached
    prefix: Stage 3 splits them off as turn context, Stage 6 attaches
    them to a COPY of the newest user message at request build.
A3  the tools array gets its own breakpoint (largest stable block).
A4  markers are stripped before re-apply — the moving history
    breakpoint must not accumulate past Anthropic's 4-block limit.

Plus Phase 0: the ``api.ttft`` event stamps first-content latency on
both the streaming and non-streaming paths.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

import pytest

from geny_executor.core.state import PipelineState, TokenUsage
from geny_executor.llm_client import APIRequest, APIResponse, BaseClient, ClientCapabilities
from geny_executor.llm_client.types import ContentBlock
from geny_executor.stages.s03_system import SystemStage
from geny_executor.stages.s03_system.artifact.default.builders import (
    ComposablePromptBuilder,
    DateTimeBlock,
    MutablePromptBuilder,
    PersonaBlock,
    PinnedFactsBlock,
    RetrievedMemoryBlock,
)
from geny_executor.stages.s05_cache.artifact.default.strategies import (
    AggressiveCacheStrategy,
    SystemCacheStrategy,
    _supports_cache_control,
)
from geny_executor.stages.s06_api import APIStage


class _StubClient:
    def __init__(self, provider: str):
        self.provider = provider


# ---------------------------------------------------------------------------
# A1 — provider-based gate
# ---------------------------------------------------------------------------


class TestProviderGate:
    def test_alias_model_with_anthropic_client_applies_markers(self):
        """The Geny prod shape: model="opus", provider resolved later.

        Pre-2.50 the startswith("claude-") gate returned False here and
        every request went out with zero cache markers — full prefill of
        tools + system + history each turn."""
        state = PipelineState()
        state.model = "opus"
        state.llm_client = _StubClient("anthropic")
        state.system = "You are helpful."

        SystemCacheStrategy().apply_cache_markers(state)

        assert isinstance(state.system, list)
        assert state.system[0]["cache_control"] == {"type": "ephemeral"}

    def test_non_anthropic_provider_is_bypassed_even_for_claude_model(self):
        state = PipelineState()
        state.model = "claude-sonnet-4-6"  # e.g. claude via a proxy gateway
        state.llm_client = _StubClient("openai")
        state.system = "You are helpful."

        SystemCacheStrategy().apply_cache_markers(state)

        assert state.system == "You are helpful."  # untouched string

    def test_cli_backend_is_bypassed(self):
        state = PipelineState()
        state.model = "opus"
        state.llm_client = _StubClient("claude_code_cli")
        assert _supports_cache_control(state) is False

    def test_no_client_falls_back_to_model_heuristic_alias_aware(self):
        state = PipelineState()
        state.model = "sonnet"
        assert _supports_cache_control(state) is True
        state.model = "claude-opus-4-7"
        assert _supports_cache_control(state) is True
        state.model = "gpt-4o"
        assert _supports_cache_control(state) is False


# ---------------------------------------------------------------------------
# A3 + A4 — tools breakpoint, marker hygiene
# ---------------------------------------------------------------------------


def _anthropic_state(*, msgs: int = 0) -> PipelineState:
    state = PipelineState()
    state.model = "opus"
    state.llm_client = _StubClient("anthropic")
    state.system = "Persona."
    state.tools = [
        {"name": "read", "description": "r", "input_schema": {"type": "object"}},
        {"name": "write", "description": "w", "input_schema": {"type": "object"}},
    ]
    for i in range(msgs):
        role = "user" if i % 2 == 0 else "assistant"
        state.messages.append({"role": role, "content": f"m{i}"})
    return state


def _count_markers(state: PipelineState) -> int:
    n = 0
    if isinstance(state.system, list):
        n += sum(1 for b in state.system if isinstance(b, dict) and "cache_control" in b)
    for tool in state.tools or []:
        if isinstance(tool, dict) and "cache_control" in tool:
            n += 1
    for msg in state.messages:
        content = msg.get("content")
        if isinstance(content, list):
            n += sum(1 for b in content if isinstance(b, dict) and "cache_control" in b)
    return n


class TestAggressiveStrategy:
    def test_tools_get_a_breakpoint_on_the_last_entry(self):
        state = _anthropic_state()
        AggressiveCacheStrategy().apply_cache_markers(state)
        assert "cache_control" not in state.tools[0]
        assert state.tools[-1]["cache_control"] == {"type": "ephemeral"}

    def test_markers_do_not_accumulate_across_turns(self):
        """The moving history breakpoint re-applies every turn; without
        the strip pass, turn N carries N markers and the API rejects the
        request at >4. Simulate 6 turns on one growing state."""
        strategy = AggressiveCacheStrategy(stable_history_offset=2)
        state = _anthropic_state(msgs=2)

        for turn in range(6):
            state.messages.append({"role": "user", "content": f"u{turn}"})
            state.messages.append({"role": "assistant", "content": f"a{turn}"})
            # Stage 3 rebuilds the system string every turn.
            state.system = "Persona."
            strategy.apply_cache_markers(state)
            assert _count_markers(state) <= 4, f"marker leak on turn {turn}"

        # Exactly: tools(1) + system(1) + history(1)
        assert _count_markers(state) == 3

    def test_stable_system_split_marks_only_the_stable_block(self):
        state = _anthropic_state()
        stable, volatile = "Persona.", "Current date: now"
        state.system = f"{stable}\n\n{volatile}"
        state.shared["system_parts"] = {
            "stable_text": stable,
            "volatile_text": volatile,
        }

        AggressiveCacheStrategy().apply_cache_markers(state)

        assert isinstance(state.system, list) and len(state.system) == 2
        assert state.system[0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in state.system[1]
        # Byte-equivalence with the joined string is preserved.
        assert state.system[0]["text"] + state.system[1]["text"] == f"{stable}\n\n{volatile}"


# ---------------------------------------------------------------------------
# A2 — Stage 3 volatile split + Stage 6 turn-context injection
# ---------------------------------------------------------------------------


class TestVolatileSplit:
    @pytest.mark.asyncio
    async def test_composable_volatile_tail_leaves_system(self):
        stage = SystemStage(
            builder=ComposablePromptBuilder(
                blocks=[
                    PersonaBlock("Persona."),
                    PinnedFactsBlock(),
                    DateTimeBlock(),
                    RetrievedMemoryBlock(),
                ]
            )
        )
        state = PipelineState()
        state.metadata["memory_pinned"] = "User is hrjang."
        state.metadata["memory_context"] = "- [note] k: retrieved fact"

        await stage.execute("in", state)

        # Stable prefix: persona + pinned facts stay in system.
        assert "Persona." in state.system
        assert "Pinned Facts" in state.system
        # Volatile tail: clock + retrieved memory ride as turn context.
        assert "Current date" not in state.system
        assert "Relevant Knowledge" not in state.system
        ctx = state.shared["turn_context_text"]
        assert "Current date" in ctx and "Relevant Knowledge" in ctx

    @pytest.mark.asyncio
    async def test_static_builder_unchanged(self):
        """Builders without parts keep the legacy path byte-for-byte."""
        stage = SystemStage(prompt="Just a prompt.")
        state = PipelineState()
        await stage.execute("in", state)
        assert state.system == "Just a prompt."
        assert "turn_context_text" not in state.shared

    @pytest.mark.asyncio
    async def test_mutable_builder_splits_dynamic_blocks(self):
        builder = MutablePromptBuilder(prompt="Base.").add_block(DateTimeBlock())
        stage = SystemStage(builder=builder)
        state = PipelineState()
        await stage.execute("in", state)
        assert state.system == "Base."
        assert "Current date" in state.shared["turn_context_text"]

    @pytest.mark.asyncio
    async def test_system_placement_keeps_legacy_layout_and_records_parts(self):
        stage = SystemStage(
            builder=ComposablePromptBuilder(
                blocks=[PersonaBlock("Persona."), DateTimeBlock()]
            ),
            volatile_placement="system",
        )
        state = PipelineState()
        await stage.execute("in", state)
        assert "Persona." in state.system and "Current date" in state.system
        parts = state.shared["system_parts"]
        assert parts["stable_text"] == "Persona."
        assert "Current date" in parts["volatile_text"]

    @pytest.mark.asyncio
    async def test_stale_turn_context_cleared_when_volatile_empty(self):
        stage = SystemStage(
            builder=ComposablePromptBuilder(
                blocks=[PersonaBlock("Persona."), RetrievedMemoryBlock()]
            )
        )
        state = PipelineState()
        state.shared["turn_context_text"] = "stale from last turn"
        await stage.execute("in", state)  # no memory_context set → no volatile text
        assert "turn_context_text" not in state.shared


class TestTurnContextInjection:
    def test_injected_into_copy_of_last_user_message(self):
        stage = APIStage()
        state = PipelineState()
        state.messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "latest question"},
        ]
        state.shared["turn_context_text"] = "Current date: now"

        kwargs = stage._call_kwargs(stage.resolve_model_config(state), state)

        sent = kwargs["messages"]
        blocks = sent[-1]["content"]
        assert blocks[0] == {"type": "text", "text": "latest question"}
        assert "<session-context>" in blocks[-1]["text"]
        assert "Current date: now" in blocks[-1]["text"]
        # state.messages must stay untouched — request-only injection.
        assert state.messages[-1]["content"] == "latest question"

    def test_injection_lands_before_extra_messages(self):
        """Tool-loop inner calls append the pending exchange AFTER the
        history — the context must ride on the user turn, not on a
        tool_result message."""
        stage = APIStage()
        state = PipelineState()
        state.messages = [{"role": "user", "content": "q"}]
        state.shared["turn_context_text"] = "ctx"

        extra = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1"}]}]
        kwargs = stage._call_kwargs(
            stage.resolve_model_config(state), state, extra_messages=extra
        )

        sent = kwargs["messages"]
        assert "<session-context>" in sent[0]["content"][-1]["text"]
        assert sent[1]["content"][0]["type"] == "tool_result"

    def test_no_user_message_skips_injection(self):
        stage = APIStage()
        state = PipelineState()
        state.messages = [{"role": "assistant", "content": "only assistant"}]
        state.shared["turn_context_text"] = "ctx"
        kwargs = stage._call_kwargs(stage.resolve_model_config(state), state)
        assert kwargs["messages"] == state.messages


# ---------------------------------------------------------------------------
# Phase 0 — api.ttft probe
# ---------------------------------------------------------------------------


def _response(text: str = "done") -> APIResponse:
    return APIResponse(
        content=[ContentBlock(type="text", text=text)],
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=3, output_tokens=2),
        model="fake-model",
        message_id="msg_fake",
    )


class _StreamClient(BaseClient):
    provider = "fake"
    capabilities = ClientCapabilities(
        supports_thinking=True, supports_tools=True, supports_streaming=True
    )

    def __init__(self):
        super().__init__(api_key="k")

    async def _send(self, request: APIRequest, *, purpose: str = "") -> APIResponse:
        return _response()

    async def create_message_stream(
        self,
        *,
        model_config: Any,
        messages: List[Dict[str, Any]],
        system: Any = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        purpose: str = "",
    ) -> AsyncIterator[Dict[str, Any]]:
        yield {"type": "text_delta", "text": "hi"}
        yield {"type": "text_delta", "text": " there"}
        yield {"type": "message_complete", "response": _response("hi there")}


class TestTtftProbe:
    @pytest.mark.asyncio
    async def test_streaming_emits_single_ttft_on_first_content(self):
        stage = APIStage(stream=True)
        state = PipelineState()
        state.llm_client = _StreamClient()
        state.messages = [{"role": "user", "content": "q"}]

        await stage.execute("in", state)

        ttft = [e for e in state.events if e["type"] == "api.ttft"]
        assert len(ttft) == 1
        payload = ttft[0]["data"]
        assert payload["stream"] is True
        assert payload["first_visible"] == "text_delta"
        assert payload["ttft_ms"] >= 0

    @pytest.mark.asyncio
    async def test_non_streaming_emits_ttft_as_full_latency(self):
        stage = APIStage()
        stage.update_config({"stream": False})  # explicit operator setting wins
        state = PipelineState()
        state.llm_client = _StreamClient()
        state.messages = [{"role": "user", "content": "q"}]

        await stage.execute("in", state)

        ttft = [e for e in state.events if e["type"] == "api.ttft"]
        assert len(ttft) == 1
        assert ttft[0]["data"]["first_visible"] == "complete"

    @pytest.mark.asyncio
    async def test_api_response_carries_cache_token_split(self):
        stage = APIStage(stream=True)
        state = PipelineState()
        state.llm_client = _StreamClient()
        state.messages = [{"role": "user", "content": "q"}]

        await stage.execute("in", state)

        responses = [e for e in state.events if e["type"] == "api.response"]
        assert responses, "api.response missing"
        data = responses[-1]["data"]
        assert "cache_read_input_tokens" in data
        assert "cache_creation_input_tokens" in data
