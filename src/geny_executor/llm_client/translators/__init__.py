"""Canonical ↔ vendor translation helpers.

This package owns the canonical-format → provider-native translation
logic for every LLM client (Anthropic, OpenAI, Google, vLLM, …). LLM
clients are the *only* place where vendor-specific wire formats are
constructed, so the helpers belong next to them — not under
``stages.s06_api``, which is a *consumer* of these clients.

The implementation lives in :mod:`._canonical`; this module re-exports
the public surface. ``stages.s06_api._translate`` is now a deprecated
backward-compat shim that re-exports from here.
"""

from __future__ import annotations

from geny_executor.llm_client.translators._canonical import (
    blocks_to_text,
    canonical_messages_to_anthropic,
    canonical_messages_to_google,
    canonical_messages_to_openai,
    canonical_thinking_to_google,
    canonical_thinking_to_openai,
    canonical_tool_choice_to_google,
    canonical_tool_choice_to_openai,
    canonical_tools_to_google,
    canonical_tools_to_openai,
    normalize_stop_reason,
    split_tool_results,
    split_tool_uses,
)
from geny_executor.llm_client.translators._cli import (
    StreamJsonAccumulator,
    assemble_response_from_stream_json,
    build_stream_json_stdin,
    claude_code_argv,
    parse_json_output_to_response,
    stream_json_line_to_canonical_event,
    thinking_to_effort,
)

__all__ = [
    "assemble_response_from_stream_json",
    "blocks_to_text",
    "build_stream_json_stdin",
    "canonical_messages_to_anthropic",
    "canonical_messages_to_google",
    "canonical_messages_to_openai",
    "canonical_thinking_to_google",
    "canonical_thinking_to_openai",
    "canonical_tool_choice_to_google",
    "canonical_tool_choice_to_openai",
    "canonical_tools_to_google",
    "canonical_tools_to_openai",
    "claude_code_argv",
    "normalize_stop_reason",
    "parse_json_output_to_response",
    "split_tool_results",
    "split_tool_uses",
    "stream_json_line_to_canonical_event",
    "StreamJsonAccumulator",
    "thinking_to_effort",
]
