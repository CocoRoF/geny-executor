"""TTFT program (2.50.0) — warmup + streaming perception, groups C/D.

C2/C3  Pipeline.warmup(): eager client build + backend pre-warm, memoized
       so turn 1 reuses the warmed instance; dropped on generation bumps.
D1     Anthropic streaming surfaces thinking deltas live (the text-only
       iterator made the whole thinking budget dead air).
D3     Pre-generation lifecycle hooks fire without blocking the pipeline
       but keep their delivery order and the PIPELINE_END flush.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, List

import pytest

from geny_executor.core.state import PipelineState
from geny_executor.llm_client.anthropic import AnthropicClient


# ---------------------------------------------------------------------------
# C — Pipeline.warmup()
# ---------------------------------------------------------------------------


class _WarmableClient:
    provider = "fake"

    def __init__(self):
        self.warmup_calls = 0

    async def warmup(self, *, timeout_s: float = 8.0) -> bool:
        self.warmup_calls += 1
        return True


class TestPipelineWarmup:
    @pytest.mark.asyncio
    async def test_warmup_calls_backend_and_reports(self):
        from geny_executor import Pipeline

        pipeline = Pipeline()
        client = _WarmableClient()
        pipeline.attach_runtime(llm_client=client)

        report = await pipeline.warmup()

        assert report == {"provider": "fake", "warmed": True}
        assert client.warmup_calls == 1

    @pytest.mark.asyncio
    async def test_warmup_memo_feeds_first_state_and_invalidate_drops_it(self):
        from geny_executor import Pipeline

        pipeline = Pipeline()
        client = _WarmableClient()
        # Simulate a pipeline-resolved client (no attach): plant the memo
        # the way warmup() does when resolution built a fresh client.
        pipeline._warm_llm_client = client

        assert pipeline._resolve_llm_client() is client

        pipeline.invalidate_client()
        assert pipeline._warm_llm_client is None
        assert pipeline._resolve_llm_client() is not client

    @pytest.mark.asyncio
    async def test_warmup_never_raises_without_client(self):
        from geny_executor import Pipeline

        report = await Pipeline().warmup()
        assert report["provider"] is None and report["warmed"] is False


# ---------------------------------------------------------------------------
# D1 — Anthropic thinking-delta streaming
# ---------------------------------------------------------------------------


def _raw_event(dtype: str, **fields: Any) -> SimpleNamespace:
    return SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type=dtype, **fields))


class _FakeSDKStream:
    """Stands in for the SDK's MessageStream context manager."""

    def __init__(self, events: List[Any], final: Any):
        self._events = events
        self._final = final

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        async def _gen():
            for e in self._events:
                yield e

        return _gen()

    async def get_final_message(self):
        return self._final


class _FakeSDKClient:
    def __init__(self, events: List[Any], final: Any):
        self.messages = SimpleNamespace(stream=lambda **kw: _FakeSDKStream(events, final))


def _fake_final_message(text: str = "answer") -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        stop_reason="end_turn",
        model="claude-test",
        id="msg_1",
    )


class TestAnthropicThinkingStream:
    @pytest.mark.asyncio
    async def test_thinking_deltas_surface_before_text(self):
        """Pre-2.50 the loop consumed only text_stream — a thinking turn
        produced no chunk at all until the first text token."""
        events = [
            SimpleNamespace(type="message_start"),
            _raw_event("thinking_delta", thinking="사고 중..."),
            _raw_event("thinking_delta", thinking=" 더 사고"),
            _raw_event("text_delta", text="답"),
            SimpleNamespace(type="content_block_stop"),
        ]
        client = AnthropicClient(api_key="k")
        client._client = _FakeSDKClient(events, _fake_final_message("답"))

        chunks = []
        async for chunk in client.create_message_stream(
            model_config=SimpleNamespace(
                model="claude-test", max_tokens=100, temperature=None,
                top_p=None, top_k=None, stop_sequences=None,
                thinking_enabled=False, thinking_type="enabled",
                thinking_budget_tokens=0, thinking_display=None,
            ),
            messages=[{"role": "user", "content": "q"}],
        ):
            chunks.append(chunk)

        types = [c["type"] for c in chunks]
        assert types[0] == "thinking_delta", f"first chunk was {types[0]}"
        assert "text_delta" in types
        assert types[-1] == "message_complete"
        # thinking arrives BEFORE the first text token
        assert types.index("thinking_delta") < types.index("text_delta")
        assert chunks[0]["text"] == "사고 중..."


# ---------------------------------------------------------------------------
# D3 — ordered non-blocking lifecycle hooks
# ---------------------------------------------------------------------------


class TestLifecycleHookOrdering:
    @pytest.mark.asyncio
    async def test_slow_hook_does_not_block_run_but_order_is_kept(self):
        from geny_executor import Pipeline
        from geny_executor.hooks.events import HookEvent

        seen: List[Any] = []

        class _SlowRunner:
            async def fire(self, event, payload):
                await asyncio.sleep(0.01)
                seen.append(payload.event)
                return []

        from geny_executor.stages.s01_input import InputStage
        from geny_executor.stages.s21_yield import YieldStage
        from tests.unit.test_ttft_cache_overhaul import _StreamClient

        pipeline = Pipeline()
        pipeline.register_stage(InputStage())
        from geny_executor.stages.s06_api import APIStage

        pipeline.register_stage(APIStage())
        pipeline.register_stage(YieldStage())
        pipeline.attach_runtime(hook_runner=_SlowRunner(), llm_client=_StreamClient())

        result = await pipeline.run("hello", PipelineState(session_id="s"))
        assert result.success

        # All fires delivered by the time run() returned (END flush) …
        assert seen[0] == HookEvent.PIPELINE_START
        assert seen[-1] == HookEvent.PIPELINE_END
        # … and stage enter/exit kept their relative order.
        stage_events = [e for e in seen if e in (HookEvent.STAGE_ENTER, HookEvent.STAGE_EXIT)]
        assert stage_events[0] == HookEvent.STAGE_ENTER
