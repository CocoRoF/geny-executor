"""Tests for the credential-bundle fork runner (Phase D4).

Validates that a fork-mode skill can route through any of the 6
providers via a single :class:`CredentialBundle`, replacing the
Anthropic-only default runner.
"""

from __future__ import annotations

import asyncio
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

from geny_executor.llm_client.base import BaseClient, ClientCapabilities
from geny_executor.llm_client.credentials import (
    CredentialBundle,
    ProviderCredentials,
)
from geny_executor.llm_client.types import APIResponse, ContentBlock
from geny_executor.skills.fork import (
    ForkResult,
    make_credential_bundle_fork_runner,
)
from geny_executor.skills.types import Skill, SkillMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingClient(BaseClient):
    """Fake BaseClient that records what it received and returns canned text."""

    provider = "fake"
    capabilities = ClientCapabilities()

    last_call: dict = {}

    def __init__(self, *, api_key: str = "", **kwargs):
        super().__init__(api_key=api_key)
        type(self).last_call = {"api_key": api_key, "kwargs": kwargs}

    async def _send(self, request, *, purpose: str = ""):
        type(self).last_call["request"] = request
        type(self).last_call["purpose"] = purpose
        return APIResponse(
            content=[ContentBlock(type="text", text="forked-text")],
            stop_reason="end_turn",
        )


def _skill(
    name: str = "test",
    *,
    provider: str | None = None,
    model_override: str | None = None,
) -> Skill:
    meta = SkillMetadata(
        name=name,
        description="test skill",
        provider=provider,
        model_override=model_override,
        execution_mode="fork",
    )
    return Skill(
        id=f"skill::{name}",
        metadata=meta,
        body="be terse and helpful",
    )


def _ctx_stub():
    """A minimal stand-in for ToolContext (only attributes the runner reads)."""

    class _C:
        pass

    return _C()


# ---------------------------------------------------------------------------
# Provider routing
# ---------------------------------------------------------------------------


def test_runner_returned_unconditionally() -> None:
    """Unlike make_default_fork_runner (returns None on missing key), the
    bundle variant always returns a runner — credential issues surface
    at invocation time."""
    runner = make_credential_bundle_fork_runner(CredentialBundle())
    assert runner is not None


@pytest.mark.asyncio
async def test_runner_uses_skill_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """``skill.metadata.provider="openai"`` → runner builds an OpenAI client."""
    bundle = CredentialBundle(by_provider={
        "openai": ProviderCredentials(api_key="sk-oai"),
    })
    runner = make_credential_bundle_fork_runner(bundle)

    # Hijack ClientRegistry.get to return our recorder.
    from geny_executor.llm_client.registry import ClientRegistry

    monkeypatch.setattr(
        ClientRegistry, "get", classmethod(lambda cls, p: _RecordingClient)
    )

    skill = _skill(provider="openai")
    result = await runner(
        skill=skill, rendered_body="be helpful", invoke_args={}, parent_context=_ctx_stub(),
    )
    assert isinstance(result, ForkResult)
    assert result.is_error is False
    assert result.content == "forked-text"
    # The recorder saw the openai api_key from the bundle.
    assert _RecordingClient.last_call["api_key"] == "sk-oai"
    assert result.metadata["provider"] == "openai"


@pytest.mark.asyncio
async def test_runner_falls_back_to_fallback_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``skill.metadata.provider`` is None, the runner uses the
    fallback_provider parameter."""
    bundle = CredentialBundle(by_provider={
        "anthropic": ProviderCredentials(api_key="sk-anth"),
    })
    runner = make_credential_bundle_fork_runner(
        bundle, fallback_provider="anthropic",
    )
    from geny_executor.llm_client.registry import ClientRegistry

    monkeypatch.setattr(
        ClientRegistry, "get", classmethod(lambda cls, p: _RecordingClient)
    )

    skill = _skill(provider=None)
    result = await runner(
        skill=skill, rendered_body="rules", invoke_args={}, parent_context=_ctx_stub(),
    )
    assert result.is_error is False
    assert result.metadata["provider"] == "anthropic"
    assert _RecordingClient.last_call["api_key"] == "sk-anth"


# ---------------------------------------------------------------------------
# Missing credentials → structured error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_returns_error_when_credentials_missing() -> None:
    """The runner does not crash on missing credentials — it returns a
    ForkResult with is_error=True and an informative message."""
    bundle = CredentialBundle()  # empty
    runner = make_credential_bundle_fork_runner(bundle)
    skill = _skill(provider="openai")
    result = await runner(
        skill=skill, rendered_body="x", invoke_args={}, parent_context=_ctx_stub(),
    )
    assert result.is_error is True
    assert "openai" in result.content
    assert "credentials" in result.content.lower()


@pytest.mark.asyncio
async def test_runner_returns_error_when_provider_unknown() -> None:
    bundle = CredentialBundle(by_provider={
        "imaginary": ProviderCredentials(api_key="sk-x"),
    })
    runner = make_credential_bundle_fork_runner(bundle)
    skill = _skill(provider="imaginary")
    result = await runner(
        skill=skill, rendered_body="x", invoke_args={}, parent_context=_ctx_stub(),
    )
    assert result.is_error is True
    assert "imaginary" in result.content
    assert "unknown" in result.content.lower()


# ---------------------------------------------------------------------------
# Model override + arguments
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_uses_skill_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    bundle = CredentialBundle(by_provider={
        "anthropic": ProviderCredentials(api_key="sk-a"),
    })
    runner = make_credential_bundle_fork_runner(bundle)
    from geny_executor.llm_client.registry import ClientRegistry

    monkeypatch.setattr(
        ClientRegistry, "get", classmethod(lambda cls, p: _RecordingClient)
    )

    skill = _skill(provider="anthropic", model_override="claude-opus-4-7")
    result = await runner(
        skill=skill, rendered_body="x", invoke_args={"q": "test"}, parent_context=_ctx_stub(),
    )
    assert result.metadata["model"] == "claude-opus-4-7"
    # The argument payload should be in the user message.
    req = _RecordingClient.last_call["request"]
    user_text = req.messages[0]["content"]
    assert "test" in user_text
