"""Tests for the extended ClientCapabilities (Phase A1).

Validates the 16-field shape, frozen-ness, and the supports() helper.
"""

from __future__ import annotations

import sys
import os
from dataclasses import FrozenInstanceError, fields

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor.llm_client.base import ClientCapabilities


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_capabilities_has_expected_fields() -> None:
    names = {f.name for f in fields(ClientCapabilities)}
    expected = {
        # original 7
        "supports_thinking",
        "supports_tools",
        "supports_streaming",
        "supports_tool_choice",
        "supports_stop_sequences",
        "supports_top_k",
        "supports_system_prompt",
        # extended 9
        "supports_structured_output",
        "supports_session_continuity",
        "supports_mcp_passthrough",
        "supports_budget_limit",
        "supports_token_usage",
        "supports_cost_usage",
        "is_subprocess",
        "requires_workspace",
        "streaming_granularity",
        # plus drops
        "drops",
    }
    assert names == expected, f"unexpected fields: {names ^ expected}"


def test_capabilities_defaults_are_backward_compatible() -> None:
    """Default flags must not change behaviour of pre-existing clients."""
    cap = ClientCapabilities()
    # Original defaults
    assert cap.supports_thinking is False
    assert cap.supports_tools is False
    assert cap.supports_streaming is True
    assert cap.supports_tool_choice is False
    assert cap.supports_stop_sequences is True
    assert cap.supports_top_k is False
    assert cap.supports_system_prompt is True
    assert cap.drops == ()
    # New fields: conservative defaults that don't change existing behaviour
    assert cap.supports_structured_output is False
    assert cap.supports_session_continuity is False
    assert cap.supports_mcp_passthrough is False
    assert cap.supports_budget_limit is False
    assert cap.supports_token_usage is True
    assert cap.supports_cost_usage is False
    assert cap.is_subprocess is False
    assert cap.requires_workspace is False
    assert cap.streaming_granularity == "token"


# ---------------------------------------------------------------------------
# Frozen / immutability
# ---------------------------------------------------------------------------


def test_capabilities_is_frozen() -> None:
    cap = ClientCapabilities()
    with pytest.raises(FrozenInstanceError):
        cap.supports_thinking = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# supports() helper
# ---------------------------------------------------------------------------


def test_supports_helper_known_feature_true() -> None:
    cap = ClientCapabilities(supports_structured_output=True)
    assert cap.supports("structured_output") is True


def test_supports_helper_known_feature_false() -> None:
    cap = ClientCapabilities()
    assert cap.supports("structured_output") is False


def test_supports_helper_unknown_feature_returns_false() -> None:
    cap = ClientCapabilities()
    assert cap.supports("not_a_real_feature") is False


def test_supports_helper_all_extended_flags() -> None:
    cap = ClientCapabilities(
        supports_structured_output=True,
        supports_session_continuity=True,
        supports_mcp_passthrough=True,
        supports_budget_limit=True,
        supports_cost_usage=True,
    )
    for feature in (
        "structured_output",
        "session_continuity",
        "mcp_passthrough",
        "budget_limit",
        "cost_usage",
    ):
        assert cap.supports(feature) is True, feature


# ---------------------------------------------------------------------------
# streaming_granularity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["token", "message", "none"])
def test_capabilities_streaming_granularity_values(value: str) -> None:
    cap = ClientCapabilities(streaming_granularity=value)
    assert cap.streaming_granularity == value
