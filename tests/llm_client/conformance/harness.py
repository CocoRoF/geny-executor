"""Conformance harness — provider-agnostic contract tests for any
:class:`BaseClient` subclass.

Phase A3 ships the skeleton + checks that every provider can advertise
its capabilities consistently, accept a canonical :class:`APIRequest`,
and return a canonical :class:`APIResponse`. The harness will grow into
the full 32+ case suite as Phases B/C/D land each new client.

A provider's test module subclasses :class:`ConformanceTestSuite` and
provides:
  - ``provider_name`` (class attr)
  - ``make_client(mode)`` returning a :class:`BaseClient` instance
  - ``mocked_response_text`` (optional) for assertion targets

The harness's tests run in ``mocked`` mode by default — vendor SDKs and
CLI binaries are stubbed via fixtures. Pass ``--live=<provider,...>`` on
the pytest CLI to opt into live API calls (gated by ``RUN_LIVE`` env).
"""

from __future__ import annotations

import os
from typing import Literal

import pytest

from geny_executor.core.config import ModelConfig
from geny_executor.llm_client.base import BaseClient, ClientCapabilities


Mode = Literal["mocked", "live"]


# ---------------------------------------------------------------------------
# Decorator: skip a test when the client doesn't advertise the capability
# ---------------------------------------------------------------------------


def capability(feature: str):
    """Test decorator: skip when ``client.capabilities.supports_<feature>``
    is False. Used by every capability-gated case."""

    def decorator(fn):
        async def wrapped(self, *args, **kwargs):
            client = self.make_client(mode=self._mode)
            if not client.supports(feature):
                pytest.skip(f"{self.provider_name} does not support {feature!r}")
            return await fn(self, *args, **kwargs)

        wrapped.__name__ = fn.__name__
        wrapped.__doc__ = fn.__doc__
        wrapped._capability = feature  # introspectable
        return wrapped

    return decorator


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class ConformanceTestSuite:
    """Provider-agnostic contract tests. Subclass and supply ``make_client``."""

    #: Name of the provider under test. Subclasses MUST override.
    provider_name: str = ""

    #: Default mode. Tests may override per-instance.
    _mode: Mode = "mocked"

    # -------------------------------------------------------------- fixture
    def make_client(self, *, mode: Mode = "mocked") -> BaseClient:
        """Subclasses MUST override and return a configured client."""
        raise NotImplementedError(f"{type(self).__name__} must override make_client()")

    def make_model_config(self) -> ModelConfig:
        return ModelConfig(model="default", max_tokens=64, temperature=0.0)

    # -------------------------------------------------------- foundational
    def test_provider_attr_is_set(self) -> None:
        client = self.make_client()
        assert client.provider, f"{self.provider_name}.provider must be set"
        assert client.provider == self.provider_name

    def test_capabilities_present(self) -> None:
        client = self.make_client()
        assert isinstance(client.capabilities, ClientCapabilities)

    def test_capabilities_has_all_extended_fields(self) -> None:
        """Every provider must explicitly populate the extended capability
        flags (no hidden defaults). Subclasses can override expected_capabilities."""
        client = self.make_client()
        caps = client.capabilities
        # Just touch every field — AttributeError would fire if missing.
        for attr in (
            "supports_thinking", "supports_tools", "supports_streaming",
            "supports_tool_choice", "supports_stop_sequences", "supports_top_k",
            "supports_system_prompt", "supports_structured_output",
            "supports_session_continuity", "supports_mcp_passthrough",
            "supports_budget_limit", "supports_token_usage",
            "supports_cost_usage", "is_subprocess", "requires_workspace",
            "streaming_granularity",
        ):
            assert hasattr(caps, attr), attr

    def test_supports_helper_works(self) -> None:
        client = self.make_client()
        # supports("thinking") must match supports_thinking
        for feature in ("thinking", "tools", "streaming", "structured_output"):
            attr = f"supports_{feature}"
            assert client.supports(feature) == getattr(client.capabilities, attr)

    # ------------------------------------------------- streaming usage
    #
    # Audit §2.5: OpenAI's streaming path aggregated $0 for months
    # because the request never asked for the usage chunk — the
    # harvesting branch existed, the flag didn't, and no test pinned
    # the contract. ``supports_token_usage=True`` is a *promise*, so
    # the harness enforces it: every provider declaring it must return
    # non-zero usage from a mocked streaming call. A suite that cannot
    # mock its stream FAILS (not skips) — a new provider must wire
    # ``make_usage_stream_client()`` or stop advertising the flag.

    def make_usage_stream_client(self) -> BaseClient:
        """Return a client whose vendor streaming surface is stubbed to
        emit at least one text delta AND a usage payload. Suites for
        providers with ``supports_token_usage=True`` MUST override."""
        raise NotImplementedError

    async def test_streaming_usage_nonzero_when_supported(self) -> None:
        probe = self.make_client()
        if not probe.capabilities.supports_token_usage:
            pytest.skip(
                f"{self.provider_name} does not declare supports_token_usage"
            )
        try:
            client = self.make_usage_stream_client()
        except NotImplementedError:
            pytest.fail(
                f"{self.provider_name} declares supports_token_usage=True but "
                "its conformance suite does not override "
                "make_usage_stream_client() — either wire a mocked stream "
                "or stop advertising the capability."
            )

        completes = []
        async for evt in client.create_message_stream(
            model_config=self.make_model_config(),
            messages=[{"role": "user", "content": "usage probe"}],
        ):
            if evt.get("type") == "message_complete":
                completes.append(evt["response"])

        assert completes, "stream must terminate with a message_complete event"
        usage = completes[-1].usage
        assert usage.input_tokens > 0, (
            f"{self.provider_name} streamed usage has input_tokens=0 — "
            "the $0-aggregation bug (audit §2.5) is back"
        )
        assert usage.output_tokens > 0, (
            f"{self.provider_name} streamed usage has output_tokens=0"
        )


# ---------------------------------------------------------------------------
# Live mode gate
# ---------------------------------------------------------------------------


def live_mode_enabled(provider: str) -> bool:
    raw = os.environ.get("RUN_LIVE", "")
    if not raw:
        return False
    return provider in {p.strip() for p in raw.split(",")}
