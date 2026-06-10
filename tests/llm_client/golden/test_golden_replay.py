"""Golden wire-transcript replay (audit §2.3).

Every fixture in this directory is a REAL recorded Claude Code CLI
transcript (see the sibling ``.meta`` files for provenance + the exact
command). Pre-2.2.0, every wire fixture in the suite was a dict invented
inside a test body — so the tests pinned the *authors' assumptions*
about the wire, not the wire. The v2.1.4 incident (``stream_event``
lines silently ignored, streaming collapsed to one terminal delta for
weeks) shipped through 3,276 green tests for exactly that reason.

Replaying the recorded bytes asserts three invariants per transcript:

  (a) per-token ``text_delta`` events come out of the accumulator for
      the ``stream_event`` delta lines (true streaming, not envelope
      collapse),
  (b) ``finalize()`` text equals the terminal envelope / ``result``
      text with NO duplication (the envelope repeats the streamed text
      in full),
  (c) zero unknown / malformed lines — i.e. the parser actually knows
      every line shape this CLI version emits. When a future CLI bumps
      the wire, re-record a golden here and this suite becomes the
      failing test *before* prod does.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import pytest

from geny_executor.llm_client._cli_runtime import parse_stream_json_line
from geny_executor.llm_client.translators._cli import (
    StreamJsonAccumulator,
    parse_json_output_to_response,
)


GOLDEN_DIR = Path(__file__).resolve().parent
STREAM_GOLDENS = sorted(GOLDEN_DIR.glob("cli-*-stream.jsonl"))
JSON_GOLDENS = sorted(GOLDEN_DIR.glob("cli-*-json.json"))


def _meta(path: Path) -> Dict[str, Any]:
    return json.loads(path.with_suffix(".meta").read_text(encoding="utf-8"))


def _lines(path: Path) -> List[bytes]:
    return [l for l in path.read_bytes().splitlines() if l.strip()]


def _is_stream_event_text_delta(obj: Dict[str, Any]) -> bool:
    if str(obj.get("type", "")) != "stream_event":
        return False
    ev = obj.get("event") or {}
    if str(ev.get("type", "")) != "content_block_delta":
        return False
    return str((ev.get("delta") or {}).get("type", "")) == "text_delta"


def test_goldens_exist() -> None:
    """The replay suite is pointless if the fixtures vanish silently."""
    assert STREAM_GOLDENS, "no cli-*-stream.jsonl goldens found"
    assert JSON_GOLDENS, "no cli-*-json.json goldens found"


@pytest.mark.parametrize("golden", STREAM_GOLDENS, ids=lambda p: p.name)
def test_stream_golden_replay(golden: Path) -> None:
    meta = _meta(golden)
    expected_text = meta["expected_text"]

    accum = StreamJsonAccumulator(model="golden-replay")
    text_delta_events: List[str] = []
    stream_event_chunks: List[str] = []
    terminal_result: Dict[str, Any] = {}

    for raw in _lines(golden):
        obj = parse_stream_json_line(raw)
        assert obj is not None, f"unparseable golden line: {raw[:120]!r}"
        events = accum.feed(obj)

        if _is_stream_event_text_delta(obj):
            chunk = obj["event"]["delta"]["text"]
            stream_event_chunks.append(chunk)
            # (a) the per-token line must surface as a per-token event —
            # not be silently absorbed and replayed via the envelope.
            assert events == [{"type": "text_delta", "text": chunk}], (
                f"stream_event text_delta did not stream: {events!r}"
            )
        if str(obj.get("type", "")) == "result":
            terminal_result = obj

        for e in events:
            if e.get("type") == "text_delta":
                text_delta_events.append(e["text"])

    # The recorded CLI streamed at least one real per-token chunk.
    assert stream_event_chunks, "golden carries no stream_event text deltas"

    # (b) streamed deltas reassemble to the terminal text, and finalize()
    # agrees — the duplicate-bearing assistant envelope must NOT double it.
    assert "".join(text_delta_events) == expected_text
    assert terminal_result.get("result") == expected_text
    resp = accum.finalize()
    assert resp.text == expected_text, (
        f"finalize() text diverged/duplicated: {resp.text!r}"
    )

    # (c) zero tolerated-but-unknown lines: the parser fully understands
    # this CLI version's wire. A failure here after re-recording means
    # the wire moved — teach the parser, don't relax the assertion.
    assert accum.unknown_line_count == 0, accum.unknown_samples
    assert accum.malformed_line_count == 0, accum.unknown_samples

    # Cost / usage flow from the terminal result envelope.
    assert resp.usage.cost_usd == pytest.approx(terminal_result["total_cost_usd"])
    assert resp.usage.output_tokens == terminal_result["usage"]["output_tokens"]
    assert resp.stop_reason == "end_turn"
    assert resp.message_id  # session/message correlation handle present


def test_stream_golden_2_1_149_thinking_not_dropped() -> None:
    """The 2.1.149 transcript carries thinking via the ``thinking`` wire
    key — the shape that exposed the silent-drop bug (the accumulator
    read only ``text``, every thinking token vanished, and the terminal
    envelope masked the loss by re-recording the full block)."""
    golden = GOLDEN_DIR / "cli-2.1.149-stream.jsonl"
    accum = StreamJsonAccumulator(model="golden-replay")

    envelope_thinking = ""
    streamed_thinking: List[str] = []
    for raw in _lines(golden):
        obj = parse_stream_json_line(raw)
        assert obj is not None
        for e in accum.feed(obj):
            if e.get("type") == "thinking_delta":
                streamed_thinking.append(e["text"])
        if str(obj.get("type", "")) == "assistant":
            for block in (obj.get("message") or {}).get("content", []):
                if block.get("type") == "thinking":
                    envelope_thinking = block["thinking"]

    assert envelope_thinking, "golden lost its thinking envelope"
    # Streamed per-token thinking reassembles to the envelope's full text…
    assert "".join(streamed_thinking) == envelope_thinking
    # …and finalize() carries it exactly once (envelope replay skipped).
    resp = accum.finalize()
    assert resp.thinking_blocks
    assert resp.thinking_blocks[0].thinking_text == envelope_thinking


@pytest.mark.parametrize("golden", JSON_GOLDENS, ids=lambda p: p.name)
def test_json_golden_parse(golden: Path) -> None:
    """The real ``--output-format json`` envelope: text in the top-level
    ``result`` string, cost in ``total_cost_usd`` (audit §3.4 — the
    pre-2.2.0 parser expected an invented ``content[]`` shape and
    returned an empty, cost-less response for exactly these bytes)."""
    meta = _meta(golden)
    raw = golden.read_bytes()
    obj = json.loads(raw)

    resp = parse_json_output_to_response(raw, model="golden-replay")
    assert resp.text == meta["expected_text"]
    assert resp.stop_reason == obj["stop_reason"]
    assert resp.usage.cost_usd == pytest.approx(obj["total_cost_usd"])
    assert resp.usage.input_tokens == obj["usage"]["input_tokens"]
    assert resp.usage.output_tokens == obj["usage"]["output_tokens"]
    assert resp.usage.duration_ms == obj["duration_ms"]
    # No message_id on the real envelope — the session id is the stable
    # correlation handle, mirroring the streaming accumulator.
    assert resp.message_id == obj["session_id"]
