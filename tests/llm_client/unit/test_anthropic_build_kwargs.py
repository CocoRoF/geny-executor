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


import pytest

from geny_executor.llm_client.anthropic import (
    AnthropicClient,
    _ANTHROPIC_MODEL_ALIASES,
    _model_rejects_sampling_params,
    _model_requires_adaptive_thinking,
    _resolve_anthropic_model,
    _retry_kwargs_after_deprecation,
    _translate_thinking_to_adaptive,
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
    """All three fixes layered: alias → canonical (2.1.1),
    unconditional temperature drop (2.1.2), thinking shape migration
    (2.1.3). This is the exact configuration Geny's VTuber env hits —
    pinning ``opus`` with thinking enabled, the legacy v1
    budget_tokens shape, and an explicit ``temperature``."""
    client = AnthropicClient(api_key="sk-mock")
    kwargs = client._build_kwargs(_req(
        model="opus",
        temperature=0.0,
        thinking={"type": "enabled", "budget_tokens": 12000},
    ))
    assert kwargs["model"] == "claude-opus-4-7"
    assert "temperature" not in kwargs
    # Opus 4.7 demands ``adaptive``; the migration drops the now-
    # invalid ``budget_tokens`` (the API rejects it under adaptive
    # as ``thinking.adaptive.budget_tokens: Extra inputs are not
    # permitted``).
    assert kwargs["thinking"] == {"type": "adaptive"}


# ── 2.1.2 — Opus 4.7 unconditional sampling-param rejection ───────


def test_model_rejects_sampling_params_for_opus_4_7():
    assert _model_rejects_sampling_params("claude-opus-4-7") is True


def test_model_rejects_sampling_params_for_dated_opus_4_7_variant():
    """Prefix match covers future pinned variants without a code
    change — ``claude-opus-4-7-20yyyymmdd`` for any date."""
    assert _model_rejects_sampling_params("claude-opus-4-7-20260101") is True


@pytest.mark.parametrize(
    "model",
    ["claude-sonnet-4-6", "claude-haiku-4-5-20251001", "claude-opus-4-6"],
)
def test_model_rejects_sampling_params_false_for_non_opus_4_7(model: str) -> None:
    assert _model_rejects_sampling_params(model) is False


def test_build_kwargs_drops_temperature_for_opus_4_7_without_thinking() -> None:
    """The big one for 2.1.2 — Opus 4.7 refuses temperature
    regardless of whether ``thinking`` is set. AdaptiveModelRouter
    auto-promotes thinking calls to Opus, but a plain call to Opus
    (e.g. memory_distill) must also drop temperature."""
    client = AnthropicClient(api_key="sk-mock")
    kwargs = client._build_kwargs(_req(
        model="claude-opus-4-7",
        temperature=0.0,
        # no thinking
    ))
    assert "temperature" not in kwargs
    assert kwargs["model"] == "claude-opus-4-7"


def test_build_kwargs_drops_all_three_for_opus_4_7() -> None:
    client = AnthropicClient(api_key="sk-mock")
    kwargs = client._build_kwargs(_req(
        model="claude-opus-4-7",
        temperature=0.5,
        top_p=0.9,
        top_k=20,
    ))
    for blocked in ("temperature", "top_p", "top_k"):
        assert blocked not in kwargs


def test_build_kwargs_drops_temperature_after_alias_resolves_to_opus_4_7() -> None:
    """An env that pins ``opus`` (alias) should drop temperature
    after resolution to the canonical ``claude-opus-4-7``."""
    client = AnthropicClient(api_key="sk-mock")
    kwargs = client._build_kwargs(_req(
        model="opus",
        temperature=0.2,
    ))
    assert kwargs["model"] == "claude-opus-4-7"
    assert "temperature" not in kwargs


def test_sonnet_4_6_keeps_temperature_when_no_thinking() -> None:
    """Regression — only Opus 4.7 (and prefix variants) belong in
    ``_TEMPERATURE_DEPRECATED_PREFIXES``. Sonnet / Haiku still accept
    temperature."""
    client = AnthropicClient(api_key="sk-mock")
    kwargs = client._build_kwargs(_req(
        model="claude-sonnet-4-6",
        temperature=0.7,
    ))
    assert kwargs["temperature"] == 0.7


# ── 2.1.2 — Retry-on-deprecation safety net ───────────────────────


def test_retry_kwargs_strips_temperature_on_known_400_message() -> None:
    """The exact phrasing Anthropic sent on 2026-06-04 for Opus 4.7."""
    class _Fake400:
        message = (
            "Error code: 400 - {'type': 'error', 'error': {'type': "
            "'invalid_request_error', 'message': "
            "'temperature is deprecated for this model.'}}"
        )

    kwargs = {
        "model": "claude-opus-4-7",
        "messages": [{"role": "user", "content": "x"}],
        "max_tokens": 1024,
        "temperature": 0.0,
    }
    retry = _retry_kwargs_after_deprecation(kwargs, _Fake400())
    assert retry is not None
    assert "temperature" not in retry
    # Other fields survive
    assert retry["model"] == "claude-opus-4-7"
    assert retry["max_tokens"] == 1024


def test_retry_kwargs_strips_backticked_field_name() -> None:
    """Some Anthropic error payloads wrap the field name in
    backticks (``\`temperature\` is deprecated``)."""
    class _Fake400:
        message = "`temperature` is deprecated for this model."

    kwargs = {"model": "claude-opus-4-7", "temperature": 0.5, "max_tokens": 100}
    retry = _retry_kwargs_after_deprecation(kwargs, _Fake400())
    assert retry is not None
    assert "temperature" not in retry


def test_retry_kwargs_strips_top_p_when_that_is_the_deprecation() -> None:
    class _Fake400:
        message = "top_p is deprecated for this model."

    kwargs = {"model": "claude-x", "top_p": 0.9, "temperature": 0.5, "max_tokens": 100}
    retry = _retry_kwargs_after_deprecation(kwargs, _Fake400())
    assert retry is not None
    assert "top_p" not in retry
    # Other sampling params survive — only the field named in the
    # error message gets stripped.
    assert retry["temperature"] == 0.5


def test_retry_kwargs_returns_none_for_unrelated_error() -> None:
    """Non-deprecation errors must not trigger the retry path —
    let the caller re-raise with the original classification."""
    class _SomeOther:
        message = "rate limit exceeded"

    kwargs = {"model": "claude-x", "temperature": 0.5, "max_tokens": 100}
    assert _retry_kwargs_after_deprecation(kwargs, _SomeOther()) is None


def test_retry_kwargs_returns_none_when_field_already_absent() -> None:
    """Deprecation said temperature, but kwargs doesn't have it —
    nothing to strip, so don't loop."""
    class _Fake400:
        message = "temperature is deprecated for this model."

    kwargs = {"model": "claude-x", "max_tokens": 100}
    assert _retry_kwargs_after_deprecation(kwargs, _Fake400()) is None


# ── 2.1.3 — Opus 4.7 thinking.type=enabled → adaptive migration ───
# (helpers imported in the module-level block above)


def test_model_requires_adaptive_thinking_for_opus_4_7():
    assert _model_requires_adaptive_thinking("claude-opus-4-7") is True
    assert _model_requires_adaptive_thinking("claude-opus-4-7-20260101") is True


@pytest.mark.parametrize(
    "model",
    ["claude-sonnet-4-6", "claude-haiku-4-5-20251001", "claude-opus-4-6"],
)
def test_model_requires_adaptive_thinking_false_for_v1_models(model: str) -> None:
    assert _model_requires_adaptive_thinking(model) is False


def test_translate_thinking_to_adaptive_flips_type_and_drops_budget():
    out = _translate_thinking_to_adaptive(
        {"type": "enabled", "budget_tokens": 4096},
    )
    assert out == {"type": "adaptive"}


def test_translate_thinking_to_adaptive_preserves_unrelated_keys():
    out = _translate_thinking_to_adaptive(
        {"type": "enabled", "budget_tokens": 4096, "display": "summarized"},
    )
    assert out == {"type": "adaptive", "display": "summarized"}


def test_build_kwargs_translates_thinking_for_opus_4_7() -> None:
    """The exact failure we hit on 2026-06-04: VTuber's memory stage
    pinned Opus 4.7 via the router with the legacy enabled-shape
    thinking dict, and Anthropic returned
    ``thinking.type.enabled is not supported for this model``."""
    client = AnthropicClient(api_key="sk-mock")
    kwargs = client._build_kwargs(_req(
        model="claude-opus-4-7",
        thinking={"type": "enabled", "budget_tokens": 8192},
    ))
    assert kwargs["thinking"] == {"type": "adaptive"}


def test_build_kwargs_translates_thinking_after_alias_resolution() -> None:
    """An env pinning the ``opus`` alias should also hit the
    translation after the alias resolves to ``claude-opus-4-7``."""
    client = AnthropicClient(api_key="sk-mock")
    kwargs = client._build_kwargs(_req(
        model="opus",
        thinking={"type": "enabled", "budget_tokens": 4096},
    ))
    assert kwargs["model"] == "claude-opus-4-7"
    assert kwargs["thinking"] == {"type": "adaptive"}


def test_build_kwargs_does_not_translate_for_sonnet_4_6() -> None:
    """Regression — only Opus 4.7 (and prefix variants) demand
    ``adaptive``. Sonnet / Haiku still accept ``enabled``."""
    client = AnthropicClient(api_key="sk-mock")
    kwargs = client._build_kwargs(_req(
        model="claude-sonnet-4-6",
        thinking={"type": "enabled", "budget_tokens": 4096},
    ))
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 4096}


def test_build_kwargs_leaves_adaptive_thinking_alone() -> None:
    """If the caller already shipped ``adaptive``, don't reshape it."""
    client = AnthropicClient(api_key="sk-mock")
    kwargs = client._build_kwargs(_req(
        model="claude-opus-4-7",
        thinking={"type": "adaptive"},
    ))
    assert kwargs["thinking"] == {"type": "adaptive"}


def test_build_kwargs_full_opus_combo() -> None:
    """The full failure path the VTuber session hit: Opus 4.7 +
    temperature + thinking.type=enabled + budget_tokens. All three
    fixes (alias, sampling-param drop, thinking migration) layer
    cleanly on the same call."""
    client = AnthropicClient(api_key="sk-mock")
    kwargs = client._build_kwargs(_req(
        model="opus",
        temperature=0.0,
        thinking={"type": "enabled", "budget_tokens": 4096},
    ))
    assert kwargs["model"] == "claude-opus-4-7"
    assert "temperature" not in kwargs
    assert kwargs["thinking"] == {"type": "adaptive"}


# ── 2.1.3 — Retry self-heals the thinking migration too ───────────


def test_retry_kwargs_self_heals_thinking_enabled_400() -> None:
    """An env shipping ``thinking.type=enabled`` against a future
    adaptive-only model the prefix list doesn't know yet — the API
    will tell us via the 400, and the retry path self-heals."""
    class _Fake400:
        message = (
            '"thinking.type.enabled" is not supported for this model. '
            'Use "thinking.type.adaptive" and "output_config.effort"'
        )

    kwargs = {
        "model": "claude-future-thinking-v2",
        "messages": [{"role": "user", "content": "x"}],
        "max_tokens": 1024,
        "thinking": {"type": "enabled", "budget_tokens": 2048},
    }
    retry = _retry_kwargs_after_deprecation(kwargs, _Fake400())
    assert retry is not None
    assert retry["thinking"] == {"type": "adaptive"}


def test_retry_kwargs_returns_none_when_thinking_already_adaptive() -> None:
    """If the request was already adaptive, the 400 message about
    enabled must not trigger a useless retry."""
    class _Fake400:
        message = '"thinking.type.enabled" is not supported for this model.'

    kwargs = {"model": "claude-x", "thinking": {"type": "adaptive"}, "max_tokens": 100}
    assert _retry_kwargs_after_deprecation(kwargs, _Fake400()) is None
