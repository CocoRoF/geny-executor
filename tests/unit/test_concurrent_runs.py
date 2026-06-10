"""Concurrent run_stream calls on ONE pipeline — the never-reproduced case.

Audit 2026-06-09 completeness gap: overlapping runs on a shared
pipeline were reasoned about (the wave-2 run_id filter, the counter
semantics of ``_runs_in_flight``) but never actually reproduced in the
suite. Both hosts can hit this shape — GAPT keeps one pipeline per
workspace and a second SSE request can land while the first turn still
streams.

Pinned here, with two concurrent ``run_stream`` calls on one pipeline
and two separate states:

  (a) run_id-filtered streams don't cross-leak events — each consumer
      sees only its own run's traffic (the wave-2 collector filter);
  (b) both runs complete, each with its own text and its own state;
  (c) the pipeline-scoped ``events()`` tap sees BOTH run_ids (it is
      the cross-run observer; ``run_stream`` is the per-run one);
  (d) the in-flight counter brackets the overlap correctly and drains
      to 0.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from geny_executor import Pipeline, PipelineConfig, PipelineState
from geny_executor.stages.s01_input import InputStage
from geny_executor.stages.s06_api import APIStage, MockProvider
from geny_executor.stages.s09_parse import ParseStage
from geny_executor.stages.s21_yield import YieldStage


def _last_user_text(request: Any) -> str:
    """Extract the latest user text from an APIRequest, both content shapes."""
    for message in reversed(list(request.messages or [])):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
    return ""


class _EchoStreamProvider(MockProvider):
    """Streams deltas derived from the REQUEST input, with real awaits.

    Echoing the input back makes cross-leak detection content-level:
    if run B's deltas ever surface in run A's stream, the assertion
    failure names the leaked text, not just a mismatched run_id. The
    per-delta sleep forces the two runs to interleave on the loop.
    """

    def __init__(self, *, delay_s: float = 0.01) -> None:
        super().__init__()
        self._delay_s = delay_s

    async def create_message_stream(self, request):  # noqa: ANN001
        from geny_executor.llm_client.types import APIResponse, ContentBlock, TokenUsage

        text = f"echo {_last_user_text(request)}"
        self._call_history.append(request)
        self._call_count += 1
        for word in text.split(" "):
            await asyncio.sleep(self._delay_s)
            yield {"type": "text_delta", "text": word + " "}
        response = APIResponse(
            content=[ContentBlock(type="text", text=text)],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=10, output_tokens=10),
            model=request.model,
            message_id=f"echo_{self._call_count}",
        )
        yield {"type": "message_complete", "response": response}


def _pipeline() -> Pipeline:
    pipeline = Pipeline(PipelineConfig(name="concurrent-runs"))
    pipeline.register_stage(InputStage())
    pipeline.register_stage(APIStage(provider=_EchoStreamProvider()))
    pipeline.register_stage(ParseStage())  # populates state.final_text
    pipeline.register_stage(YieldStage())
    return pipeline


async def _consume(pipeline: Pipeline, text: str, state: PipelineState) -> Dict[str, Any]:
    """Drive one run_stream to completion, recording what it saw."""
    events: List[Any] = []
    max_in_flight = 0
    async for event in pipeline.run_stream(text, state):
        events.append(event)
        max_in_flight = max(max_in_flight, pipeline._runs_in_flight)
    run_ids = {e.run_id for e in events if e.run_id}
    return {
        "events": events,
        "run_ids": run_ids,
        "deltas": "".join(
            e.data.get("text", "") for e in events if e.type == "text.delta"
        ),
        "completes": [e for e in events if e.type == "pipeline.complete"],
        "max_in_flight": max_in_flight,
    }


@pytest.mark.asyncio
async def test_two_concurrent_run_streams_do_not_cross_leak():
    pipeline = _pipeline()
    state_a = PipelineState(session_id="session-a")
    state_b = PipelineState(session_id="session-b")

    tap_events: List[Any] = []

    async def tap_consumer() -> None:
        async for event in pipeline.events():
            tap_events.append(event)

    tap_task = asyncio.create_task(tap_consumer())
    await asyncio.sleep(0)

    result_a, result_b = await asyncio.gather(
        _consume(pipeline, "alpha", state_a),
        _consume(pipeline, "beta", state_b),
    )

    # (d) the runs actually overlapped — otherwise this test proves nothing.
    assert max(result_a["max_in_flight"], result_b["max_in_flight"]) == 2
    assert pipeline._runs_in_flight == 0
    assert pipeline.run_in_progress is False

    # (a) run_id isolation: every correlated event in a stream belongs
    # to exactly that stream's run.
    assert len(result_a["run_ids"]) == 1
    assert len(result_b["run_ids"]) == 1
    assert result_a["run_ids"] != result_b["run_ids"]

    # …and content-level: no leaked deltas from the other run.
    assert "alpha" in result_a["deltas"] and "beta" not in result_a["deltas"]
    assert "beta" in result_b["deltas"] and "alpha" not in result_b["deltas"]

    # Session correlation rode along correctly too.
    assert {e.session_id for e in result_a["events"]} == {"session-a"}
    assert {e.session_id for e in result_b["events"]} == {"session-b"}

    # (b) both runs completed with their own text and their own state.
    assert [c.data["result"] for c in result_a["completes"]] == ["echo alpha"]
    assert [c.data["result"] for c in result_b["completes"]] == ["echo beta"]
    assert state_a.final_text == "echo alpha"
    assert state_b.final_text == "echo beta"
    assert state_a.messages is not state_b.messages

    # (c) the events() tap is the cross-run observer: both run_ids, with
    # a start and a complete each.
    await asyncio.sleep(0.05)  # let the tap drain
    tap_starts = {e.run_id for e in tap_events if e.type == "pipeline.start"}
    tap_completes = {e.run_id for e in tap_events if e.type == "pipeline.complete"}
    expected = result_a["run_ids"] | result_b["run_ids"]
    assert tap_starts == expected
    assert tap_completes == expected

    await pipeline.aclose()
    await asyncio.wait_for(tap_task, 2)


@pytest.mark.asyncio
async def test_concurrent_runs_keep_per_state_message_history_separate():
    """True state corruption would show up as one state's transcript
    containing the other's turn — pin that it does not."""
    pipeline = _pipeline()
    state_a = PipelineState(session_id="hist-a")
    state_b = PipelineState(session_id="hist-b")

    await asyncio.gather(
        _consume(pipeline, "alpha", state_a),
        _consume(pipeline, "beta", state_b),
    )

    def _texts(state: PipelineState) -> str:
        chunks: List[str] = []
        for message in state.messages:
            content = message.get("content")
            if isinstance(content, str):
                chunks.append(content)
            elif isinstance(content, list):
                chunks.extend(str(b.get("text", "")) for b in content if isinstance(b, dict))
        return " ".join(chunks)

    assert "alpha" in _texts(state_a) and "beta" not in _texts(state_a)
    assert "beta" in _texts(state_b) and "alpha" not in _texts(state_b)


@pytest.mark.asyncio
async def test_overlapping_runs_unlock_only_after_both_drain():
    """The counter (not bool) semantics: while EITHER run is in flight
    the mutation lock holds; it releases only when both finish."""
    pipeline = _pipeline()

    async def _run(text: str, session: str) -> None:
        async for _event in pipeline.run_stream(text, PipelineState(session_id=session)):
            if pipeline._runs_in_flight == 2:
                # Mid-overlap: between-turn maintenance must refuse.
                with pytest.raises(RuntimeError):
                    pipeline.refresh_runtime(session_runtime=object())

    await asyncio.gather(_run("alpha", "lock-a"), _run("beta", "lock-b"))

    assert pipeline.run_in_progress is False
    pipeline.refresh_runtime(session_runtime=object())  # legal again
