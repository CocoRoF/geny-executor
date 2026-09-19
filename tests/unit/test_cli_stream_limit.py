"""claude stdout stream limit — a huge line must not kill the turn.

Regression for the delegated-PPTX failure (2026-07-14): the claude CLI
emits one stream-json event per line, and the model's own text rides
INSIDE those lines; a 200K-char answer blew asyncio's default
StreamReader limit (64 KiB) and reading aborted the whole turn with
"Separator is found, but chunk is longer than limit".

The lesson survived two transport changes. It first lived in a CLI
process runner, then on this module's own subprocess reader, and since
2.69.0 it is a value we hand the Agent SDK: ``max_buffer_size``. What is
tested here is that the knob still exists, still defaults generously,
and still reaches the SDK — the SDK owns the reading now, so the reading
itself is no longer ours to test.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from geny_executor.llm_client.claude_code_tokens import _stream_limit


class TestStreamLimitConfig:
    def test_default_is_32_mib(self, monkeypatch):
        monkeypatch.delenv("GENY_CLI_STREAM_LIMIT", raising=False)
        assert _stream_limit() == 32 * 1024 * 1024

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("GENY_CLI_STREAM_LIMIT", str(2**20))
        assert _stream_limit() == 2**20

    def test_garbage_and_too_small_fall_back(self, monkeypatch):
        monkeypatch.setenv("GENY_CLI_STREAM_LIMIT", "banana")
        assert _stream_limit() == 32 * 1024 * 1024
        monkeypatch.setenv("GENY_CLI_STREAM_LIMIT", "1024")  # below the 64 KiB floor
        assert _stream_limit() == 32 * 1024 * 1024


@pytest.mark.asyncio
async def test_the_knob_reaches_the_sdk(monkeypatch):
    """Not decoration: it is what the SDK's reader is actually given.

    A default-sized buffer here is the 2026-07-14 failure coming back.
    """
    from claude_agent_sdk import ClaudeAgentOptions

    from tests.unit.test_claude_code_token_client import (  # noqa: PLC0415
        _SAYS_HELLO,
        _client,
        _run,
    )

    monkeypatch.setenv("GENY_CLI_STREAM_LIMIT", str(4 * 2**20))
    seen: Dict[str, Any] = {}
    real = ClaudeAgentOptions.__init__

    def spy(self: Any, *a: Any, **kw: Any) -> None:
        seen.update(kw)
        real(self, *a, **kw)

    ClaudeAgentOptions.__init__ = spy  # type: ignore[method-assign]
    try:
        await _run(_client(_SAYS_HELLO))
    finally:
        ClaudeAgentOptions.__init__ = real  # type: ignore[method-assign]

    assert seen["max_buffer_size"] == 4 * 2**20
