"""Deprecated backward-compat shim.

The canonical-format translation helpers now live in
:mod:`geny_executor.llm_client.translators` — that's the layer that owns
vendor wire formats. ``stages.s06_api`` is a *consumer* of LLM clients
and should not host vendor translation logic.

This module re-exports the public surface so existing imports keep
working during the migration. It will be removed in a future cycle;
new code must import from :mod:`geny_executor.llm_client.translators`.
"""

from __future__ import annotations

from geny_executor.llm_client.translators import (
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

__all__ = [
    "blocks_to_text",
    "canonical_messages_to_anthropic",
    "canonical_messages_to_google",
    "canonical_messages_to_openai",
    "canonical_thinking_to_google",
    "canonical_thinking_to_openai",
    "canonical_tool_choice_to_google",
    "canonical_tool_choice_to_openai",
    "canonical_tools_to_google",
    "canonical_tools_to_openai",
    "normalize_stop_reason",
    "split_tool_results",
    "split_tool_uses",
]
