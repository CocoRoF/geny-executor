"""Unit tests for :meth:`AnthropicClient._build_kwargs`.

Covers two robustness fixes shipped in 2.1.1:

  * Model-alias resolution — ``opus``/``sonnet``/``haiku`` get
    expanded to the canonical IDs the Anthropic Messages API
    expects (the SDK returns 404 for the short aliases).

  * Extended-thinking sampling-param compat — when ``thinking`` is
    on, the API rejects ``temperature``/``top_p``/``top_k`` as
    deprecated. Drop them at the boundary so an env that pins both
    a thinking budget and an explicit temperature still works
    instead of returning HTTP 400.

The CLI surface (``ClaudeCodeCLIClient`` /
``llm_client.translators._cli``) keeps short aliases intact — the
``claude`` binary resolves them itself. Verified by re-running the
existing translator test that asserts those flags pass through.
"""

from __future__ import annotations

from typing import List

import pytest

from geny_executor.llm_client.anthropic import (
    AnthropicClient,
    _ANTHROPIC_MODEL_ALIASES,
    _resolve_anthropic_model,
)
from geny_executor.llm_client.types import APIRequest


def _req(**overrides) -> APIRequest:
    """Minimal valid APIRequest. Fields under test are overrides."""
    base: dict = {
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1024,
    }
    base.update(overrides)
    return APIRequest(**base)


# ── Pure alias resolver ───────────────────────────────────────────


@pytest.mark.parametrize(
    "alias,canonical",
    list(_ANTHROPIC_MODEL_ALIASES.items()),
)
def test_resolve_known_alias_returns_canonical(alias: str, canonical: str) -> None:
    assert _resolve_anthropic_model(alias) == canonical


def test_resolve_passthrough_for_canonical_id() -> None:
    """Canonical IDs round-trip unchanged."""
    assert _resolve_anthropic_model("claude-opus-4-7") == "claude-opus-4-7"
    assert _resolve_anthropic_model("claude-sonnet-4-6") == "claude-sonnet-4-6"


def test_resolve_passthrough_for_unknown_value() -> None:
    """Unknown strings (typos, future canonical IDs, third-party
    base_url targets) are not silently rewritten."""
    assert _resolve_anthropic_model("gpt-4") == "gpt-4"
    assert _resolve_anthropic_model("claude-opus-5-0") == "claude-opus-5-0"
    assert _resolve_anthropic_model("") == ""


# ── _build_kwargs — alias resolution wires through ────────────────


def test_build_kwargs_resolves_alias_in_model_field() -> None:
    client = AnthropicClient(api_key="sk-mock")
    kwargs = client._build_kwargs(_req(model="opus"))
    assert kwargs["model"] == "claude-opus-4-7"


def test_build_kwargs_resolves_sonnet_alias() -> None:
    client = AnthropicClient(api_key="sk-mock")
    kwargs = client._build_kwargs(_req(model="sonnet"))
    assert kwargs["model"] == "claude-sonnet-4-6"


def test_build_kwargs_keeps_canonical_id_unchanged() -> None:
    client = AnthropicClient(api_key="sk-mock")
    kwargs = client._build_kwargs(_req(model="claude-opus-4-7"))
    assert kwargs["model"] == "claude-opus-4-7"


def test_build_kwargs_keeps_unknown_model_unchanged() -> None:
    """A future-dated or third-party model id passes through. Better
    to let the SDK surface a precise 404 than to silently rewrite."""
    client = AnthropicClient(api_key="sk-mock")
    kwargs = client._build_kwargs(_req(model="claude-future-9-0"))
    assert kwargs["model"] == "claude-future-9-0"


# ── _build_kwargs — thinking ↔ sampling-param conflict ────────────


def test_build_kwargs_drops_temperature_when_thinking_enabled() -> None:
    """The big one — Geny's default env ships ``temperature=0.0``
    plus ``thinking_enabled=True`` and the API used to 400 with
    ``temperature is deprecated for this model``."""
    client = AnthropicClient(api_key="sk-mock")
    kwargs = client._build_kwargs(_req(
        temperature=0.3,
        thinking={"type": "enabled", "budget_tokens": 4096},
    ))
    assert "temperature" not in kwargs
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 4096}


def test_build_kwargs_drops_top_p_when_thinking_enabled() -> None:
    client = AnthropicClient(api_key="sk-mock")
    kwargs = client._build_kwargs(_req(
        top_p=0.9,
        thinking={"type": "enabled", "budget_tokens": 4096},
    ))
    assert "top_p" not in kwargs


def test_build_kwargs_drops_top_k_when_thinking_enabled() -> None:
    client = AnthropicClient(api_key="sk-mock")
    kwargs = client._build_kwargs(_req(
        top_k=10,
        thinking={"type": "enabled", "budget_tokens": 4096},
    ))
    assert "top_k" not in kwargs


def test_build_kwargs_drops_all_three_sampling_params_at_once() -> None:
    client = AnthropicClient(api_key="sk-mock")
    kwargs = client._build_kwargs(_req(
        temperature=0.5,
        top_p=0.8,
        top_k=20,
        thinking={"type": "enabled", "budget_tokens": 8192},
    ))
    for blocked in ("temperature", "top_p", "top_k"):
        assert blocked not in kwargs
    # The non-blocked params survive intact.
    assert kwargs["max_tokens"] == 1024
    assert kwargs["thinking"]["budget_tokens"] == 8192


def test_build_kwargs_keeps_sampling_params_when_thinking_absent() -> None:
    """Without ``thinking``, the API accepts the sampling params —
    don't silently strip them."""
    client = AnthropicClient(api_key="sk-mock")
    kwargs = client._build_kwargs(_req(
        temperature=0.7,
        top_p=0.95,
        top_k=15,
    ))
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_p"] == 0.95
    assert kwargs["top_k"] == 15
    assert "thinking" not in kwargs


def test_build_kwargs_alias_resolution_and_thinking_drop_together() -> None:
    """The two fixes are independent — combining them shouldn't trip
    either path. This is the configuration Geny's VTuber env hits."""
    client = AnthropicClient(api_key="sk-mock")
    kwargs = client._build_kwargs(_req(
        model="opus",
        temperature=0.0,
        thinking={"type": "enabled", "budget_tokens": 12000},
    ))
    assert kwargs["model"] == "claude-opus-4-7"
    assert "temperature" not in kwargs
    assert kwargs["thinking"]["budget_tokens"] == 12000
