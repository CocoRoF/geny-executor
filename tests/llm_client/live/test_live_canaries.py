"""Live vendor-boundary canaries — opt-in, skipped by default.

Why (audit 2026-06-09 §1-4/§2.2): every 2.1.x incident was wire drift a
mocked suite could not see — the CLI changed its stream-json shape, the
HTTP API started rejecting ``temperature`` for Opus, OpenAI streaming
aggregated $0. These canaries hit the REAL vendor surfaces with
cents-level requests so drift is caught by a nightly job instead of a
prod incident. They assert exactly the contracts the 2.1.1–2.1.3 and
wave-1 fixes restored:

  * anthropic — alias model ids resolve at the boundary; opus +
    thinking + temperature self-heals instead of 400ing; streaming
    produces real per-token deltas.
  * claude_code_cli — the local binary still speaks a wire shape the
    translator fully recognises (``unknown_line_count == 0`` is THE
    drift canary), and the version handshake answers.
  * openai — streamed usage is non-zero (the wave-1 $0-cost fix).

Gating: every test is skipped unless ``RUN_LIVE`` is set AND the
provider's credential/binary is present. ``RUN_LIVE=1`` enables all
canaries; ``RUN_LIVE=anthropic,openai`` enables per-provider (same
convention as ``tests/llm_client/conformance/harness.py``).

Suggested nightly invocation::

    RUN_LIVE=1 ANTHROPIC_API_KEY=... OPENAI_API_KEY=... \\
        .venv/bin/python -m pytest tests/llm_client/live/ -v -rs

(The claude_code_cli canaries additionally need a logged-in ``claude``
binary on PATH or CLAUDE_CODE_BINARY pointing at one.)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import pytest

from geny_executor.core.config import ModelConfig
from geny_executor.llm_client._cli_runtime import detect_binary


def _live_enabled(provider: str) -> bool:
    raw = os.environ.get("RUN_LIVE", "").strip()
    if not raw or raw == "0":
        return False
    if raw.lower() in {"1", "true", "all"}:
        return True
    return provider in {p.strip() for p in raw.split(",")}


def _anthropic_ready() -> bool:
    return _live_enabled("anthropic") and bool(os.environ.get("ANTHROPIC_API_KEY"))


def _cli_ready() -> bool:
    override = os.environ.get("CLAUDE_CODE_BINARY", "") or None
    return _live_enabled("claude_code_cli") and bool(detect_binary("claude", override))


def _openai_ready() -> bool:
    return _live_enabled("openai") and bool(os.environ.get("OPENAI_API_KEY"))


anthropic_only = pytest.mark.skipif(
    not _anthropic_ready(),
    reason="live canary: RUN_LIVE with anthropic + ANTHROPIC_API_KEY required",
)
cli_only = pytest.mark.skipif(
    not _cli_ready(),
    reason="live canary: RUN_LIVE with claude_code_cli + local claude binary required",
)
openai_only = pytest.mark.skipif(
    not _openai_ready(),
    reason="live canary: RUN_LIVE with openai + OPENAI_API_KEY required",
)


# ---------------------------------------------------------------------------
# anthropic — the 2.1.1–2.1.3 boundary surface
# ---------------------------------------------------------------------------


def _anthropic_client():
    from geny_executor.llm_client import AnthropicClient

    return AnthropicClient(api_key=os.environ["ANTHROPIC_API_KEY"])


@anthropic_only
@pytest.mark.asyncio
async def test_anthropic_alias_model_id_resolves() -> None:
    """``model='sonnet'`` must reach the API as a canonical id (2.1.1) —
    a 404 here means the alias table needs a bump."""
    client = _anthropic_client()
    response = await client.create_message(
        model_config=ModelConfig(model="sonnet", max_tokens=16, temperature=0.0),
        messages=[{"role": "user", "content": "Reply with exactly: ok"}],
    )
    assert response.text.strip()
    assert response.model.startswith("claude-"), (
        f"API reported model {response.model!r} — alias did not resolve to a "
        "canonical id"
    )


@anthropic_only
@pytest.mark.asyncio
async def test_anthropic_opus_thinking_temperature_self_heals() -> None:
    """The exact 2.1.1–2.1.3 incident shape: opus alias + extended
    thinking + an explicit temperature. The boundary must drop/migrate
    the incompatible params and return a normal completion instead of
    surfacing the vendor 400."""
    client = _anthropic_client()
    response = await client.create_message(
        model_config=ModelConfig(
            model="opus",
            max_tokens=2048,
            temperature=0.3,
            thinking_enabled=True,
            thinking_budget_tokens=1024,
        ),
        messages=[{"role": "user", "content": "Reply with exactly: ok"}],
    )
    assert response.stop_reason
    assert response.usage.output_tokens > 0


@anthropic_only
@pytest.mark.asyncio
async def test_anthropic_streaming_yields_multiple_text_deltas() -> None:
    client = _anthropic_client()
    deltas = 0
    completes = []
    async for event in client.create_message_stream(
        model_config=ModelConfig(model="sonnet", max_tokens=64, temperature=0.0),
        messages=[
            {"role": "user", "content": "Count from 1 to 10, separated by spaces."}
        ],
    ):
        if event.get("type") == "text_delta":
            deltas += 1
        elif event.get("type") == "message_complete":
            completes.append(event["response"])
    assert deltas > 1, "streaming returned a single blob — not actually streaming"
    assert completes and completes[-1].usage.output_tokens > 0


# ---------------------------------------------------------------------------
# claude_code_cli — wire-drift canary against the real local binary
# ---------------------------------------------------------------------------


def _cli_client(tmp_path) -> "object":  # noqa: ANN001
    from geny_executor.llm_client.claude_code import ClaudeCodeCLIClient

    return ClaudeCodeCLIClient(
        workspace_dir=str(tmp_path),
        timeout_s=180.0,
    )


@cli_only
@pytest.mark.asyncio
async def test_cli_streams_with_zero_unknown_lines(tmp_path) -> None:
    """THE drift canary: the local CLI's stream-json must be fully
    recognised by the translator. ``unknown_line_count > 0`` is the
    first observable symptom of the next 2.1.x-style wire change —
    catch it here, not in a host's masked-text incident."""
    client = _cli_client(tmp_path)
    deltas = 0
    completes = []
    async for event in client.create_message_stream(
        model_config=ModelConfig(model="haiku", max_tokens=64),
        messages=[{"role": "user", "content": "Reply with exactly: ok"}],
    ):
        if event.get("type") == "text_delta":
            deltas += 1
        elif event.get("type") == "message_complete":
            completes.append(event["response"])

    assert completes, "stream ended without message_complete"
    response = completes[-1]
    assert response.text.strip()
    assert deltas >= 1, "CLI produced no streamed text deltas (stream_event form)"
    raw = response.raw if isinstance(response.raw, dict) else {}
    assert int(raw.get("unknown_line_count", 0) or 0) == 0, (
        f"CLI emitted {raw.get('unknown_line_count')} unrecognised wire lines "
        f"(first: {raw.get('first_unknown_type')!r}) — translator drift"
    )
    assert int(raw.get("malformed_line_count", 0) or 0) == 0


@cli_only
@pytest.mark.asyncio
async def test_cli_version_probe_answers(tmp_path) -> None:
    client = _cli_client(tmp_path)
    version = await client._ensure_cli_version()
    assert version and version != "unknown", (
        "claude --version handshake failed — every 2.1.x incident was "
        "version skew, so an unanswerable probe is itself a finding"
    )


# ---------------------------------------------------------------------------
# openai — the wave-1 $0-cost fix
# ---------------------------------------------------------------------------


@openai_only
@pytest.mark.asyncio
async def test_openai_streamed_usage_is_nonzero() -> None:
    from geny_executor.llm_client.openai import OpenAIClient

    client = OpenAIClient(api_key=os.environ["OPENAI_API_KEY"])
    completes = []
    async for event in client.create_message_stream(
        model_config=ModelConfig(model="gpt-4o-mini", max_tokens=16, temperature=0.0),
        messages=[{"role": "user", "content": "Reply with exactly: ok"}],
    ):
        if event.get("type") == "message_complete":
            completes.append(event["response"])

    assert completes, "stream ended without message_complete"
    usage = completes[-1].usage
    assert usage.input_tokens > 0, "streamed usage input_tokens=0 — audit §2.5 is back"
    assert usage.output_tokens > 0, "streamed usage output_tokens=0 — audit §2.5 is back"
