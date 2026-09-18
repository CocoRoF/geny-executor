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
  * geny_claude_code — the local ``claude`` binary still answers in
    token mode (``-p --tools ""``) with a shape the token client reads:
    a terminal ``result`` line carrying the text, and a ``<tool_call>``
    block when THIS harness offers it a tool. Those two are the whole
    contract now that the binary is used as a pure LLM.
  * openai — streamed usage is non-zero (the wave-1 $0-cost fix).

Gating: every test is skipped unless ``RUN_LIVE`` is set AND the
provider's credential/binary is present. ``RUN_LIVE=1`` enables all
canaries; ``RUN_LIVE=anthropic,openai`` enables per-provider (same
convention as ``tests/llm_client/conformance/harness.py``).

Suggested nightly invocation::

    RUN_LIVE=1 ANTHROPIC_API_KEY=... OPENAI_API_KEY=... \\
        .venv/bin/python -m pytest tests/llm_client/live/ -v -rs

(The geny_claude_code canaries additionally need a logged-in ``claude``
binary on PATH or CLAUDE_CODE_BINARY pointing at one.)
"""

from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import pytest

from geny_executor.core.config import ModelConfig


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
    binary = override if override and os.path.exists(override) else shutil.which("claude")
    return _live_enabled("geny_claude_code") and bool(binary)


def _openai_ready() -> bool:
    return _live_enabled("openai") and bool(os.environ.get("OPENAI_API_KEY"))


anthropic_only = pytest.mark.skipif(
    not _anthropic_ready(),
    reason="live canary: RUN_LIVE with anthropic + ANTHROPIC_API_KEY required",
)
cli_only = pytest.mark.skipif(
    not _cli_ready(),
    reason="live canary: RUN_LIVE with geny_claude_code + local claude binary required",
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
# geny_claude_code — wire-drift canary against the real local binary
# ---------------------------------------------------------------------------


def _token_client(tmp_path) -> "object":  # noqa: ANN001
    from geny_executor.llm_client.claude_code_tokens import ClaudeCodeTokenClient

    return ClaudeCodeTokenClient(
        binary_path=os.environ.get("CLAUDE_CODE_BINARY", "") or "claude",
        timeout_s=180.0,
    )


@cli_only
@pytest.mark.asyncio
async def test_token_mode_streams_text(tmp_path) -> None:
    """THE drift canary: ``claude -p --tools ""`` must still answer with a
    terminal ``result`` line the token client can read. A shape change
    here is the first observable symptom of the next 2.1.x-style wire
    change — catch it nightly, not in a host's masked-text incident."""
    client = _token_client(tmp_path)
    completes = []
    async for event in client.create_message_stream(
        model_config=ModelConfig(model="haiku", max_tokens=64),
        messages=[{"role": "user", "content": "Reply with exactly: ok"}],
    ):
        if event.get("type") == "message_complete":
            completes.append(event["response"])

    assert completes, "stream ended without message_complete"
    assert completes[-1].text.strip(), "the binary answered with no text"


@cli_only
@pytest.mark.asyncio
async def test_token_mode_asks_for_our_tool(tmp_path) -> None:
    """The other half of the contract: the binary is a pure LLM here, so
    it must REQUEST a tool in this harness's ``<tool_call>`` text
    protocol rather than executing one of its own. A native tool_use
    block instead would mean the ``--tools ""`` isolation regressed."""
    client = _token_client(tmp_path)
    completes = []
    async for event in client.create_message_stream(
        model_config=ModelConfig(model="haiku", max_tokens=256),
        messages=[{"role": "user", "content": "List the files in /tmp."}],
        tools=[
            {
                "name": "ListDir",
                "description": "List the entries of a directory.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ],
    ):
        if event.get("type") == "message_complete":
            completes.append(event["response"])

    assert completes, "stream ended without message_complete"
    blocks = completes[-1].content or []
    assert any(getattr(b, "type", "") == "tool_use" for b in blocks), (
        "the model did not request our tool — either the <tool_call> "
        "protocol drifted or the binary fell back to its own tools"
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
