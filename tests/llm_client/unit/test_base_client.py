"""Unit tests for the 2.2.0 :class:`BaseClient` boundary machinery.

Three mechanisms shipped together (audit §3.5 + Tier 2):

  * ``capabilities.drops`` is now authoritative — declared fields are
    stripped from the outgoing request and reported via
    ``llm_client.parameter_dropped``. Through 2.1.x the list was a decoy:
    a manifest-pinned ``temperature`` on the CLI backend was accepted,
    validated, serialized — and ignored in total silence.

  * ``_heal_request_kwargs`` / ``_invoke_with_heal`` — the retry-once
    self-heal born in AnthropicClient (2.1.2/2.1.3), promoted so every
    SDK boundary gets the same reflex. Successful heals must be LOUD
    (``llm_client.drift_healed`` + WARNING) because a heal firing in
    prod means a static compatibility table is stale.

  * ``_provenance`` — ``{'provider', 'sdk_version'}`` stamps for
    ``APIResponse.raw``; every 2.1.x incident was version skew that
    post-mortems had to reconstruct from infra logs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pytest

from geny_executor.core.config import ModelConfig
from geny_executor.core.errors import APIError, ErrorCategory
from geny_executor.llm_client.base import (
    BaseClient,
    ClientCapabilities,
    _resolve_sdk_version,
)
from geny_executor.llm_client.types import APIRequest, APIResponse


# ---------------------------------------------------------------------------
# Synthetic clients
# ---------------------------------------------------------------------------


class _DroppyClient(BaseClient):
    """Declares an aggressive drops list, including one unknown name —
    a stale declaration must not crash request assembly. Every declared
    field's ``supports_*`` flag (where one exists) is False, matching
    how every shipped client writes its tuple: since review B3 a drop
    whose instance capability flag is True is SKIPPED (capability
    upgrades win over the static declaration — see
    ``_CapabilityUpgradedClient`` below)."""

    provider = "droppy"
    capabilities = ClientCapabilities(
        supports_thinking=False,
        drops=("temperature", "max_tokens", "tools", "thinking_enabled",
               "warp_drive"),
    )

    async def _send(self, request: APIRequest, *, purpose: str = "") -> APIResponse:
        return APIResponse()


class _CapabilityUpgradedClient(BaseClient):
    """Drops tuple says strip tools/thinking, capability flags say the
    instance supports them — the B3 shape (a stale declaration relative
    to a ``configure_capabilities``-style upgrade). Capability wins."""

    provider = "upgraded"
    capabilities = ClientCapabilities(
        supports_thinking=True,
        supports_tools=True,
        supports_tool_choice=True,
        drops=("temperature", "tools", "tool_choice", "thinking_enabled"),
    )

    async def _send(self, request: APIRequest, *, purpose: str = "") -> APIResponse:
        return APIResponse()


class _GateAndDropClient(BaseClient):
    """top_k is BOTH capability-gated (supports_top_k=False) and declared
    in drops — the overlap case that must not double-emit."""

    provider = "gate_and_drop"
    capabilities = ClientCapabilities(
        supports_top_k=False,
        drops=("top_k", "top_k"),  # duplicate declaration on purpose
    )

    async def _send(self, request: APIRequest, *, purpose: str = "") -> APIResponse:
        return APIResponse()


class _PlainClient(BaseClient):
    """No drops — everything passes through."""

    provider = "plain"
    capabilities = ClientCapabilities(supports_top_k=True)

    async def _send(self, request: APIRequest, *, purpose: str = "") -> APIResponse:
        return APIResponse()


def _build(client: BaseClient, model_config: ModelConfig, **kwargs: Any) -> APIRequest:
    return client._build_request(
        model_config=model_config,
        messages=[{"role": "user", "content": "ping"}],
        system=kwargs.get("system", ""),
        tools=kwargs.get("tools"),
        tool_choice=kwargs.get("tool_choice"),
        stream=False,
    )


def _dropped(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        e["field"]: e
        for e in events
        if e["type"] == "llm_client.parameter_dropped"
    }


# ---------------------------------------------------------------------------
# drops is authoritative
# ---------------------------------------------------------------------------


def test_declared_drops_are_stripped_from_request() -> None:
    events: List[Dict[str, Any]] = []
    client = _DroppyClient(event_sink=events.append)
    request = _build(
        client,
        ModelConfig(model="m", temperature=0.7, max_tokens=2048,
                    thinking_enabled=True),
        tools=[{"name": "read"}],
    )
    assert request.temperature is None
    assert request.max_tokens is None
    assert request.tools is None
    assert request.thinking is None


def test_declared_drops_emit_parameter_dropped_with_value() -> None:
    events: List[Dict[str, Any]] = []
    client = _DroppyClient(event_sink=events.append)
    _build(
        client,
        ModelConfig(model="m", temperature=0.7, max_tokens=2048,
                    thinking_enabled=True),
        tools=[{"name": "read"}],
    )
    dropped = _dropped(events)
    assert dropped["temperature"]["value"] == 0.7
    assert dropped["temperature"]["provider"] == "droppy"
    assert dropped["max_tokens"]["value"] == 2048
    assert dropped["tools"]["value"] == [{"name": "read"}]
    assert dropped["thinking_enabled"]["value"] is True


def test_temperature_zero_still_counts_as_supplied() -> None:
    """0.0 is an explicit sampling choice (greedy), not 'unset' — the
    exact value Geny's default env pins. It must be reported."""
    events: List[Dict[str, Any]] = []
    client = _DroppyClient(event_sink=events.append)
    _build(client, ModelConfig(model="m", temperature=0.0))
    assert "temperature" in _dropped(events)


def test_absent_values_do_not_emit() -> None:
    """thinking_enabled=False / tools=None — declared in drops but never
    supplied, so there is nothing to report."""
    events: List[Dict[str, Any]] = []
    client = _DroppyClient(event_sink=events.append)
    _build(client, ModelConfig(model="m", thinking_enabled=False), tools=None)
    dropped = _dropped(events)
    assert "thinking_enabled" not in dropped
    assert "tools" not in dropped


def test_unknown_drop_name_is_ignored() -> None:
    """A stale/typo'd declaration ('warp_drive') neither crashes nor
    emits — the conformance suite is where stale vocab gets caught."""
    events: List[Dict[str, Any]] = []
    client = _DroppyClient(event_sink=events.append)
    _build(client, ModelConfig(model="m"))
    assert "warp_drive" not in _dropped(events)


def test_undeclared_fields_pass_through() -> None:
    """top_p is not in _DroppyClient.drops — it must survive intact."""
    events: List[Dict[str, Any]] = []
    client = _DroppyClient(event_sink=events.append)
    request = _build(client, ModelConfig(model="m", top_p=0.9))
    assert request.top_p == 0.9
    assert "top_p" not in _dropped(events)


def test_no_drops_declared_nothing_stripped() -> None:
    events: List[Dict[str, Any]] = []
    client = _PlainClient(event_sink=events.append)
    request = _build(
        client, ModelConfig(model="m", temperature=0.5, top_k=10)
    )
    assert request.temperature == 0.5
    assert request.top_k == 10
    assert _dropped(events) == {}


def test_gate_overlap_emits_each_event_type_exactly_once() -> None:
    """top_k is capability-gated AND declared (twice!) in drops: exactly
    one ``feature_unsupported`` (compat — hosts key on it) and exactly
    one ``parameter_dropped``. Double emission would make event-counting
    dashboards lie."""
    events: List[Dict[str, Any]] = []
    client = _GateAndDropClient(event_sink=events.append)
    request = _build(client, ModelConfig(model="m", top_k=5))
    assert request.top_k is None
    unsupported = [
        e for e in events
        if e["type"] == "llm_client.feature_unsupported" and e["field"] == "top_k"
    ]
    dropped = [
        e for e in events
        if e["type"] == "llm_client.parameter_dropped" and e["field"] == "top_k"
    ]
    assert len(unsupported) == 1
    assert len(dropped) == 1
    assert dropped[0]["value"] == 5


def test_no_event_sink_still_strips() -> None:
    """Stripping is the contract; the event is the courtesy. A client
    without a sink must still honour the declaration."""
    client = _DroppyClient()
    request = _build(client, ModelConfig(model="m", temperature=0.7))
    assert request.temperature is None


# ── Capability flags beat stale drop declarations (review B3) ─────────


def test_supported_capability_skips_declared_drop() -> None:
    """drops=('tools','tool_choice','thinking_enabled') with the matching
    supports_* flags True: the fields survive, no parameter_dropped —
    the declaration is stale relative to the instance's capabilities."""
    events: List[Dict[str, Any]] = []
    client = _CapabilityUpgradedClient(event_sink=events.append)
    request = _build(
        client,
        ModelConfig(model="m", thinking_enabled=True),
        tools=[{"name": "read"}],
        tool_choice={"type": "auto"},
    )
    assert request.tools == [{"name": "read"}]
    assert request.tool_choice == {"type": "auto"}
    assert request.thinking is not None
    dropped = _dropped(events)
    assert "tools" not in dropped
    assert "tool_choice" not in dropped
    assert "thinking_enabled" not in dropped


def test_flagless_drops_still_apply_on_upgraded_client() -> None:
    """temperature has no supports_* flag — its declaration always
    bites, capability upgrades or not."""
    events: List[Dict[str, Any]] = []
    client = _CapabilityUpgradedClient(event_sink=events.append)
    request = _build(client, ModelConfig(model="m", temperature=0.7))
    assert request.temperature is None
    assert _dropped(events)["temperature"]["value"] == 0.7


# ── Per-shipped-provider: the declarations actually bite ──────────────


def test_claude_code_cli_strips_manifest_pinned_temperature() -> None:
    """The audit's headline decoy (§3.5): CLI backend declared it drops
    temperature/max_tokens and nothing consumed the declaration. Now the
    pinned values produce observable events instead of silence."""
    from geny_executor.llm_client.claude_code import ClaudeCodeCLIClient

    events: List[Dict[str, Any]] = []
    client = ClaudeCodeCLIClient(
        binary_path="/nonexistent/claude",  # _build_request never spawns
        event_sink=events.append,
    )
    request = _build(
        client, ModelConfig(model="sonnet", temperature=0.3, max_tokens=4096)
    )
    assert request.temperature is None
    assert request.max_tokens is None
    dropped = _dropped(events)
    assert dropped["temperature"]["value"] == 0.3
    assert dropped["temperature"]["provider"] == "claude_code_cli"
    assert dropped["max_tokens"]["value"] == 4096


def test_openai_declared_drops_emit() -> None:
    pytest.importorskip("openai")
    from geny_executor.llm_client.openai import OpenAIClient

    events: List[Dict[str, Any]] = []
    client = OpenAIClient(api_key="sk-mock", event_sink=events.append)
    request = _build(
        client, ModelConfig(model="gpt-4o", thinking_enabled=True, top_k=3)
    )
    assert request.thinking is None
    assert request.top_k is None
    dropped = _dropped(events)
    assert dropped["thinking_enabled"]["value"] is True
    assert dropped["top_k"]["value"] == 3


def test_vllm_declared_tool_drops_strip_tools_from_request() -> None:
    """vLLM declares drops=('…','tools','tool_choice') — before 2.2.0
    the tools rode through to the server anyway (decoy declaration)."""
    pytest.importorskip("openai")
    from geny_executor.llm_client.vllm import VLLMClient

    events: List[Dict[str, Any]] = []
    client = VLLMClient(
        base_url="http://localhost:8000/v1", event_sink=events.append
    )
    request = _build(
        client, ModelConfig(model="local"), tools=[{"name": "read"}]
    )
    assert request.tools is None
    assert "tools" in _dropped(events)


def test_vllm_configure_capabilities_restores_tools() -> None:
    """Review B3: the documented configure_capabilities(supports_tools=
    True) upgrade must survive the wave-1 authoritative-drops
    enforcement — the drops tuple stays conservative, the instance flag
    wins. 2.1.x honoured this; 2.2.0 must too."""
    pytest.importorskip("openai")
    from geny_executor.llm_client.vllm import VLLMClient

    events: List[Dict[str, Any]] = []
    client = VLLMClient(
        base_url="http://localhost:8000/v1", event_sink=events.append
    )
    client.configure_capabilities(supports_tools=True, supports_tool_choice=True)
    request = _build(
        client,
        ModelConfig(model="local"),
        tools=[{"name": "read"}],
        tool_choice={"type": "auto"},
    )
    assert request.tools == [{"name": "read"}]
    assert request.tool_choice == {"type": "auto"}
    dropped = _dropped(events)
    assert "tools" not in dropped
    assert "tool_choice" not in dropped
    # The class-level capabilities are untouched — a second, default
    # client still strips (instance-scoped upgrade).
    fresh_events: List[Dict[str, Any]] = []
    fresh = VLLMClient(
        base_url="http://localhost:8000/v1", event_sink=fresh_events.append
    )
    fresh_request = _build(fresh, ModelConfig(model="local"), tools=[{"name": "read"}])
    assert fresh_request.tools is None
    assert "tools" in _dropped(fresh_events)


def test_anthropic_declares_no_drops_temperature_survives() -> None:
    from geny_executor.llm_client.anthropic import AnthropicClient

    events: List[Dict[str, Any]] = []
    client = AnthropicClient(api_key="sk-mock", event_sink=events.append)
    request = _build(client, ModelConfig(model="claude-sonnet-4-6", temperature=0.7))
    assert request.temperature == 0.7
    assert _dropped(events) == {}


# ---------------------------------------------------------------------------
# _invoke_with_heal — the generalized retry-once wrapper
# ---------------------------------------------------------------------------


class _HealingClient(BaseClient):
    provider = "healing"
    capabilities = ClientCapabilities()

    async def _send(self, request: APIRequest, *, purpose: str = "") -> APIResponse:
        return APIResponse()

    def _heal_request_kwargs(
        self, kwargs: Dict[str, Any], exc: BaseException
    ) -> Optional[Dict[str, Any]]:
        if "rename me" in str(exc) and "old_name" in kwargs:
            retry = dict(kwargs)
            retry["new_name"] = retry.pop("old_name")
            return retry
        return None


async def test_invoke_with_heal_retries_once_and_emits() -> None:
    events: List[Dict[str, Any]] = []
    client = _HealingClient(event_sink=events.append)
    calls: List[Dict[str, Any]] = []

    async def vendor_call(**kwargs: Any) -> str:
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("rename me")
        return "ok"

    result = await client._invoke_with_heal(
        vendor_call, {"model": "m-1", "old_name": 42}, purpose="unit"
    )
    assert result == "ok"
    assert len(calls) == 2
    assert calls[1] == {"model": "m-1", "new_name": 42}

    drift = [e for e in events if e["type"] == "llm_client.drift_healed"]
    assert len(drift) == 1
    assert drift[0]["provider"] == "healing"
    assert drift[0]["model"] == "m-1"
    assert drift[0]["field"] == "old_name"
    assert "rename me" in drift[0]["message"]


async def test_invoke_with_heal_logs_warning_not_info(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """WARNING on purpose — INFO is how the 2.1.x masked degradations
    stayed invisible. Operators must learn the static tables are stale."""
    client = _HealingClient()
    flag = {"n": 0}

    async def vendor_call(**kwargs: Any) -> str:
        flag["n"] += 1
        if flag["n"] == 1:
            raise RuntimeError("rename me")
        return "ok"

    with caplog.at_level(logging.WARNING, logger="geny_executor.llm_client.base"):
        await client._invoke_with_heal(
            vendor_call, {"model": "m", "old_name": 1}, purpose="unit"
        )
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("self-healed" in r.getMessage() for r in warnings)
    assert any("stale" in r.getMessage() for r in warnings)


async def test_invoke_with_heal_unhealable_raises_classified() -> None:
    client = _HealingClient()

    async def vendor_call(**kwargs: Any) -> str:
        raise RuntimeError("totally unrelated explosion")

    with pytest.raises(APIError) as ei:
        await client._invoke_with_heal(vendor_call, {"model": "m"})
    assert ei.value.category is ErrorCategory.UNKNOWN


async def test_invoke_with_heal_failed_retry_raises_second_error() -> None:
    """One retry per send — a heal whose retry also fails surfaces the
    second error, never loops."""
    events: List[Dict[str, Any]] = []
    client = _HealingClient(event_sink=events.append)
    calls = {"n": 0}

    async def vendor_call(**kwargs: Any) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("rename me")
        raise RuntimeError("still broken after heal")

    with pytest.raises(APIError) as ei:
        await client._invoke_with_heal(
            vendor_call, {"model": "m", "old_name": 1}
        )
    assert calls["n"] == 2
    assert "still broken" in str(ei.value)
    # No drift_healed for an unsuccessful heal.
    assert not [e for e in events if e["type"] == "llm_client.drift_healed"]


async def test_base_heal_hook_defaults_to_none() -> None:
    client = _PlainClient()
    assert client._heal_request_kwargs({"model": "m"}, RuntimeError("x")) is None


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def test_resolve_sdk_version_known_package() -> None:
    anthropic = pytest.importorskip("anthropic")
    assert _resolve_sdk_version("anthropic") == anthropic.__version__


def test_resolve_sdk_version_unknown_package_degrades() -> None:
    assert _resolve_sdk_version("definitely_not_installed_xyz") == "unknown"
    assert _resolve_sdk_version("") == "unknown"


def test_provenance_shape() -> None:
    client = _PlainClient()
    prov = client._provenance()
    assert prov["provider"] == "plain"
    assert prov["sdk_version"] == "unknown"  # _sdk_module unset on synthetic
