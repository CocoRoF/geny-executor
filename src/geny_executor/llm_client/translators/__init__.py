"""Canonical ↔ vendor translation helpers.

During the PR-3→PR-4 bridge, these are thin re-exports from the
original home at :mod:`geny_executor.stages.s06_api._translate`. PR-4
inverts the dependency: the functions move into this package and the
s06_api module (which deletes entirely) no longer owns them.
"""

from __future__ import annotations

from geny_executor.stages.s06_api._translate import (
    blocks_to_text,
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
