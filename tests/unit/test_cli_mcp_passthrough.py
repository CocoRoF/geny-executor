"""CLI MCP passthrough (2.2.1) — manifest ``tools.mcp_servers`` must reach
subprocess backends through their own ``--mcp-config`` channel.

The incident this pins: a user attached an MCP server (``gapt-service``,
stdio, ``npx gapt-mcp``) to a ``claude_code_cli`` environment through the
host UI. The manifest carried it correctly, the pipeline built cleanly —
and the agent had no idea the tools existed. Root cause: the pipeline
connected the server HOST-side (MCPManager) and registered its tools into
the pipeline ToolRegistry, but Stage 10 never dispatches for subprocess
backends (the CLI runs its own agentic loop) and the CLI subprocess only
sees servers passed via ``--mcp-config`` (which carried only the host's
session bridge). The host even spawned the MCP child process for nothing.

The fix routes manifest MCP servers to the client instead:
``from_manifest_async`` skips the host-side connect for passthrough-capable
providers and ``_build_client_for`` merges the servers into the client's
``mcp_config`` (host config wins on name collision) plus auto-allows
``mcp__<server>`` so ``--print`` mode (no human to answer permission
prompts) can actually call them.
"""

from __future__ import annotations

import json

import pytest

from geny_executor import CredentialBundle, ProviderCredentials
from geny_executor.core.environment import EnvironmentManifest
from geny_executor.core.pipeline import (
    Pipeline,
    _mcp_servers_to_cli_config,
    _merge_cli_mcp_config,
    _provider_wants_mcp_passthrough,
)
from geny_executor.llm_client.types import APIRequest
from geny_executor.llm_client.translators._cli import claude_code_argv
from geny_executor.tools.mcp.manager import MCPServerConfig


# ─────────────────────────────────── provider detection ─


def test_claude_code_cli_wants_passthrough() -> None:
    assert _provider_wants_mcp_passthrough("claude_code_cli") is True


@pytest.mark.parametrize("provider", ["anthropic", "openai", "google", "vllm"])
def test_sdk_providers_do_not_want_passthrough(provider: str) -> None:
    assert _provider_wants_mcp_passthrough(provider) is False


def test_unknown_or_empty_provider_is_safe_default() -> None:
    assert _provider_wants_mcp_passthrough("") is False
    assert _provider_wants_mcp_passthrough("no-such-provider") is False


# ─────────────────────────────────── shape translation ─


def test_stdio_server_translates_with_env() -> None:
    cfg, skipped = _mcp_servers_to_cli_config(
        {
            "gapt-service": MCPServerConfig(
                name="gapt-service",
                command="npx",
                args=["gapt-mcp"],
                env={"GAPT_BASE_URL": "https://gapt.example/"},
                transport="stdio",
            )
        }
    )
    assert skipped == []
    entry = cfg["mcpServers"]["gapt-service"]
    assert entry == {
        "type": "stdio",
        "command": "npx",
        "args": ["gapt-mcp"],
        "env": {"GAPT_BASE_URL": "https://gapt.example/"},
    }


def test_sse_and_http_servers_translate_with_headers() -> None:
    cfg, skipped = _mcp_servers_to_cli_config(
        {
            "a": MCPServerConfig(
                name="a", transport="sse", url="https://x/sse",
                headers={"Authorization": "Bearer t"},
            ),
            "b": MCPServerConfig(name="b", transport="http", url="https://x/mcp"),
        }
    )
    assert skipped == []
    assert cfg["mcpServers"]["a"] == {
        "type": "sse", "url": "https://x/sse",
        "headers": {"Authorization": "Bearer t"},
    }
    assert cfg["mcpServers"]["b"] == {"type": "http", "url": "https://x/mcp"}


def test_unknown_transport_is_skipped_and_reported() -> None:
    cfg, skipped = _mcp_servers_to_cli_config(
        {"weird": MCPServerConfig(name="weird", transport="carrier-pigeon")}
    )
    assert skipped == ["weird"]
    assert cfg["mcpServers"] == {}


# ─────────────────────────────────── merge semantics ─


_MANIFEST_CFG = {"mcpServers": {"gapt-service": {"type": "stdio", "command": "npx"}}}


def test_merge_into_empty_host_config() -> None:
    assert _merge_cli_mcp_config(None, _MANIFEST_CFG) == _MANIFEST_CFG
    assert _merge_cli_mcp_config({}, _MANIFEST_CFG) == _MANIFEST_CFG


def test_merge_host_dict_keeps_both_and_host_wins_collisions() -> None:
    host = {
        "mcpServers": {
            "geny": {"type": "stdio", "command": "python"},
            "gapt-service": {"type": "stdio", "command": "host-pinned"},
        }
    }
    merged = _merge_cli_mcp_config(host, _MANIFEST_CFG)
    assert set(merged["mcpServers"]) == {"geny", "gapt-service"}
    # Host definition survives the collision — session bridges and host
    # overrides must not be silently replaced by manifest entries.
    assert merged["mcpServers"]["gapt-service"]["command"] == "host-pinned"


def test_merge_host_path_readable(tmp_path) -> None:
    p = tmp_path / "host-mcp.json"
    p.write_text(json.dumps({"mcpServers": {"geny": {"type": "stdio", "command": "python"}}}))
    merged = _merge_cli_mcp_config(str(p), _MANIFEST_CFG)
    assert set(merged["mcpServers"]) == {"geny", "gapt-service"}


def test_merge_host_path_unreadable_keeps_host_path(tmp_path, caplog) -> None:
    bogus = str(tmp_path / "missing.json")
    with caplog.at_level("WARNING"):
        out = _merge_cli_mcp_config(bogus, _MANIFEST_CFG)
    assert out == bogus  # the single --mcp-config slot stays with the host
    assert any("could not be read" in r.message for r in caplog.records)


# ─────────────────────────────────── end-to-end pipeline ─


def _cli_manifest_with_mcp() -> EnvironmentManifest:
    return EnvironmentManifest.from_dict(
        {
            "metadata": {"id": "t-cli-mcp", "name": "t"},
            "pipeline": {"max_iterations": 1},
            "model": {},
            "stages": [
                {"order": 1, "name": "input", "active": True},
                {
                    "order": 6,
                    "name": "api",
                    "active": True,
                    "config": {"provider": "claude_code_cli"},
                    "strategies": {"retry": "no_retry", "router": "passthrough"},
                },
                {"order": 9, "name": "parse", "active": True},
                {"order": 21, "name": "yield", "active": True},
            ],
            "tools": {
                "built_in": [],
                "external": [],
                "mcp_servers": [
                    {
                        # A command that CANNOT spawn — if the pipeline
                        # tried the host-side connect, build would raise.
                        "name": "gapt-service",
                        "transport": "stdio",
                        "command": "/nonexistent-mcp-binary",
                        "args": ["gapt-mcp"],
                        "env": {"GAPT_BASE_URL": "https://gapt.example/"},
                    }
                ],
            },
        }
    )


def _cli_bundle(extras: dict | None = None) -> CredentialBundle:
    # binary_path points at a real executable so the credentials are
    # non-empty (require() gate) — the client never spawns it in these
    # tests.
    return CredentialBundle(
        by_provider={
            "claude_code_cli": ProviderCredentials(
                api_key="", binary_path="/bin/sh", extras=dict(extras or {})
            )
        }
    )


@pytest.mark.asyncio
async def test_cli_provider_skips_host_connect_and_stashes_passthrough() -> None:
    pipeline = await Pipeline.from_manifest_async(
        _cli_manifest_with_mcp(), credentials=_cli_bundle(), strict=False
    )
    try:
        # Host-side connect was skipped: a /nonexistent binary would have
        # raised MCPConnectionError during build.
        assert pipeline._cli_mcp_passthrough["mcpServers"]["gapt-service"]["command"] == (
            "/nonexistent-mcp-binary"
        )
        assert pipeline._cli_mcp_passthrough_provider == "claude_code_cli"

        client = pipeline._build_client_for("claude_code_cli")
        assert "gapt-service" in client._mcp_config["mcpServers"]
        assert "mcp__gapt-service" in client._allow_tools
    finally:
        await pipeline.aclose()


@pytest.mark.asyncio
async def test_cli_passthrough_merges_with_host_bridge_config() -> None:
    """Geny's per-session bridge (extras['mcp_config']) and the manifest
    server must BOTH reach the CLI in one --mcp-config."""
    bridge = {"mcpServers": {"geny": {"type": "stdio", "command": "python", "args": ["bridge.py"]}}}
    pipeline = await Pipeline.from_manifest_async(
        _cli_manifest_with_mcp(),
        credentials=_cli_bundle(extras={"mcp_config": bridge, "allow_tools": ("mcp__geny",)}),
        strict=False,
    )
    try:
        client = pipeline._build_client_for("claude_code_cli")
        servers = client._mcp_config["mcpServers"]
        assert set(servers) == {"geny", "gapt-service"}
        assert "mcp__geny" in client._allow_tools
        assert "mcp__gapt-service" in client._allow_tools

        # And the argv the subprocess actually receives carries both.
        argv = claude_code_argv(
            APIRequest(model="m", messages=[], system="", stream=False),
            mcp_config=client._mcp_config,
            allow_tools=client._allow_tools,
        )
        blob = argv[argv.index("--mcp-config") + 1]
        assert set(json.loads(blob)["mcpServers"]) == {"geny", "gapt-service"}
        assert "--strict-mcp-config" in argv
        allowed = argv[argv.index("--allowedTools") + 1]
        assert "mcp__gapt-service" in allowed
    finally:
        await pipeline.aclose()


@pytest.mark.asyncio
async def test_sdk_provider_still_host_connects(monkeypatch) -> None:
    """Regression: SDK providers keep the host-side MCPManager path."""
    from geny_executor.tools.mcp import manager as mcp_manager_mod

    calls = {}

    async def fake_connect_all(self, configs):
        calls["configs"] = dict(configs)

    async def fake_discover_all(self):
        return []

    monkeypatch.setattr(mcp_manager_mod.MCPManager, "connect_all", fake_connect_all)
    monkeypatch.setattr(mcp_manager_mod.MCPManager, "discover_all", fake_discover_all)

    manifest_dict = _cli_manifest_with_mcp().to_dict()
    manifest_dict["stages"][1]["config"]["provider"] = "anthropic"
    manifest = EnvironmentManifest.from_dict(manifest_dict)

    pipeline = await Pipeline.from_manifest_async(
        manifest,
        credentials=CredentialBundle(
            by_provider={"anthropic": ProviderCredentials(api_key="sk-test")}
        ),
        strict=False,
    )
    try:
        assert "gapt-service" in calls["configs"]
        assert pipeline._cli_mcp_passthrough == {}
    finally:
        await pipeline.aclose()


@pytest.mark.asyncio
async def test_cli_manifest_without_mcp_servers_untouched() -> None:
    manifest_dict = _cli_manifest_with_mcp().to_dict()
    manifest_dict["tools"]["mcp_servers"] = []
    pipeline = await Pipeline.from_manifest_async(
        EnvironmentManifest.from_dict(manifest_dict),
        credentials=_cli_bundle(),
        strict=False,
    )
    try:
        assert pipeline._cli_mcp_passthrough == {}
        client = pipeline._build_client_for("claude_code_cli")
        assert client._mcp_config is None
    finally:
        await pipeline.aclose()
