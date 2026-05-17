"""Tests for the Phase A3 provider-location validation in
``Pipeline.from_manifest`` / ``from_manifest_async``.

Manifests must declare Stage 6 provider at ``config["provider"]``. Legacy
``strategies["provider"]`` is rejected (clean break) so the silent
divergence class of bug is impossible.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor.core.environment import (
    EnvironmentManifest,
    EnvironmentMetadata,
    StageManifestEntry,
    ToolsSnapshot,
)
from geny_executor.core.pipeline import Pipeline
from geny_executor.llm_client.credentials import (
    ConfigError,
    CredentialBundle,
    ProviderCredentials,
)


def _make_manifest(stages: list[StageManifestEntry]) -> EnvironmentManifest:
    m = EnvironmentManifest(
        metadata=EnvironmentMetadata(id="env_t", name="t"),
        model={},
        pipeline={},
        stages=[],
        tools=ToolsSnapshot(),
    )
    m.set_stage_entries(stages)
    return m


def _stage6_with_config_provider(provider: str = "anthropic") -> StageManifestEntry:
    return StageManifestEntry(
        order=6, name="api", active=True, artifact="default",
        strategies={"retry": "exponential_backoff", "router": "passthrough"},
        config={"provider": provider},
    )


def _stage6_with_strategies_provider(provider: str = "anthropic") -> StageManifestEntry:
    """Legacy shape — must be rejected by strict load."""
    return StageManifestEntry(
        order=6, name="api", active=True, artifact="default",
        strategies={"provider": provider, "retry": "exponential_backoff"},
        config={},
    )


def test_manifest_with_config_provider_loads_strict() -> None:
    m = _make_manifest([_stage6_with_config_provider("anthropic")])
    p = Pipeline.from_manifest(m, credentials=CredentialBundle(), strict=True)
    assert p.get_stage(6) is not None


def test_manifest_with_strategies_provider_rejected_strict() -> None:
    m = _make_manifest([_stage6_with_strategies_provider("anthropic")])
    with pytest.raises(ConfigError, match="strategies\\['provider'\\]"):
        Pipeline.from_manifest(m, credentials=CredentialBundle(), strict=True)


def test_manifest_with_strategies_provider_allowed_in_non_strict() -> None:
    """Non-strict mode bypasses the validator — the manifest is still
    structurally usable; the operator opted out of guard rails."""
    m = _make_manifest([_stage6_with_strategies_provider("anthropic")])
    # Should not raise:
    Pipeline.from_manifest(m, credentials=CredentialBundle(), strict=False)


def test_manifest_active_stage6_without_provider_rejected_strict() -> None:
    s6 = StageManifestEntry(
        order=6, name="api", active=True, artifact="default",
        strategies={"retry": "exponential_backoff"},
        config={},  # no provider
    )
    m = _make_manifest([s6])
    with pytest.raises(ConfigError, match="no provider is configured"):
        Pipeline.from_manifest(m, credentials=CredentialBundle(), strict=True)


def test_manifest_inactive_stage6_without_provider_ok_strict() -> None:
    """Inactive Stage 6 doesn't need a provider — the stage isn't built."""
    s6 = StageManifestEntry(
        order=6, name="api", active=False, artifact="default",
        strategies={}, config={},
    )
    m = _make_manifest([s6])
    # Should not raise:
    p = Pipeline.from_manifest(m, credentials=CredentialBundle(), strict=True)
    assert p.get_stage(6) is None


def test_api_key_kwarg_auto_wraps_into_anthropic_bundle() -> None:
    m = _make_manifest([_stage6_with_config_provider("anthropic")])
    p = Pipeline.from_manifest(m, api_key="sk-test-key", strict=True)
    creds = p._credentials.require("anthropic")
    assert creds.api_key == "sk-test-key"


def test_credentials_bundle_kwarg_wins_over_api_key() -> None:
    m = _make_manifest([_stage6_with_config_provider("openai")])
    bundle = CredentialBundle(by_provider={
        "openai": ProviderCredentials(api_key="bundle-key"),
    })
    p = Pipeline.from_manifest(
        m, credentials=bundle, api_key="ignored-anthropic-key", strict=True,
    )
    assert p._credentials is bundle
    assert p._credentials.has("anthropic") is False
    assert p._credentials.require("openai").api_key == "bundle-key"
