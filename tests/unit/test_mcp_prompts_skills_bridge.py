"""Unit tests for MCP prompts → Skills bridge (S8.4)."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

from geny_executor.skills import (
    MCP_SKILL_ID_PREFIX,
    MCP_SKILL_SOURCE_TAG,
    Skill,
    mcp_prompts_to_skills,
    mcp_skill_id,
)
from geny_executor.tools.mcp.manager import (
    MCPManager,
    MCPServerConfig,
    MCPServerConnection,
)
from geny_executor.tools.mcp.state import MCPConnectionState


# ── helpers ─────────────────────────────────────────────────────────────


def _make_conn(
    name: str,
    *,
    connected: bool = True,
    prompts: List[Dict[str, Any]] | None = None,
    list_raises: bool = False,
    prompt_responses: Dict[str, List[Dict[str, Any]]] | None = None,
) -> MCPServerConnection:
    conn = MCPServerConnection(MCPServerConfig(name=name))
    if connected:
        conn._state = MCPConnectionState.CONNECTED

    if list_raises:
        conn.list_prompts = AsyncMock(side_effect=RuntimeError("list boom"))
    else:
        conn.list_prompts = AsyncMock(return_value=list(prompts or []))

    responses = dict(prompt_responses or {})

    async def _get(prompt_name: str, args: Dict[str, Any] | None = None):
        return responses.get(prompt_name)

    conn.get_prompt = _get  # type: ignore[assignment]
    return conn


def _make_manager(*conns: MCPServerConnection) -> MCPManager:
    mgr = MCPManager()
    for conn in conns:
        mgr._servers[conn.config.name] = conn
        mgr._configs[conn.config.name] = conn.config
    return mgr


# ── mcp_skill_id helper ─────────────────────────────────────────────────


class TestSkillIdHelper:
    def test_canonical_format(self):
        assert mcp_skill_id("github", "summarise_pr") == "mcp__github__summarise_pr"

    def test_prefix_constant(self):
        assert MCP_SKILL_ID_PREFIX == "mcp__"

    def test_source_tag(self):
        assert MCP_SKILL_SOURCE_TAG == "mcp"

    def test_blank_args_rejected(self):
        with pytest.raises(ValueError):
            mcp_skill_id("", "x")
        with pytest.raises(ValueError):
            mcp_skill_id("x", "")


# ── Manager-level prompt API ────────────────────────────────────────────


class TestManagerPromptApi:
    @pytest.mark.asyncio
    async def test_list_all_prompts_aggregates(self):
        a = _make_conn(
            "a",
            prompts=[
                {"name": "p1", "description": "d1", "arguments": []},
            ],
        )
        b = _make_conn(
            "b",
            prompts=[
                {
                    "name": "p2",
                    "description": "d2",
                    "arguments": [
                        {"name": "k", "description": "kdesc", "required": True}
                    ],
                }
            ],
        )
        mgr = _make_manager(a, b)
        out = await mgr.list_all_prompts()
        assert len(out) == 2
        names = {(e["server"], e["name"]) for e in out}
        assert names == {("a", "p1"), ("b", "p2")}
        # arg shape preserved
        b_entry = next(e for e in out if e["server"] == "b")
        assert b_entry["arguments"] == [
            {"name": "k", "description": "kdesc", "required": True}
        ]

    @pytest.mark.asyncio
    async def test_list_all_prompts_skips_disconnected(self):
        a = _make_conn("a", connected=True, prompts=[{"name": "p1"}])
        b = _make_conn("b", connected=False, prompts=[{"name": "p2"}])
        mgr = _make_manager(a, b)
        out = await mgr.list_all_prompts()
        assert len(out) == 1
        assert out[0]["server"] == "a"

    @pytest.mark.asyncio
    async def test_get_mcp_prompt_routes_correctly(self):
        a = _make_conn(
            "a",
            prompt_responses={
                "p1": [{"role": "user", "content": "alpha"}]
            },
        )
        b = _make_conn(
            "b",
            prompt_responses={
                "p1": [{"role": "user", "content": "beta"}]
            },
        )
        mgr = _make_manager(a, b)
        assert await mgr.get_mcp_prompt("a", "p1") == [
            {"role": "user", "content": "alpha"}
        ]
        assert await mgr.get_mcp_prompt("b", "p1") == [
            {"role": "user", "content": "beta"}
        ]

    @pytest.mark.asyncio
    async def test_get_mcp_prompt_unknown_server(self):
        mgr = _make_manager(_make_conn("a"))
        assert await mgr.get_mcp_prompt("ghost", "x") is None

    @pytest.mark.asyncio
    async def test_get_mcp_prompt_disconnected(self):
        conn = _make_conn(
            "srv",
            connected=False,
            prompt_responses={"p": [{"role": "user", "content": "x"}]},
        )
        mgr = _make_manager(conn)
        assert await mgr.get_mcp_prompt("srv", "p") is None


# ── mcp_prompts_to_skills ──────────────────────────────────────────────


class TestPromptsToSkills:
    @pytest.mark.asyncio
    async def test_empty_when_no_servers(self):
        mgr = MCPManager()
        skills = await mcp_prompts_to_skills(mgr)
        assert skills == []

    @pytest.mark.asyncio
    async def test_skips_disconnected_servers(self):
        a = _make_conn("a", connected=False, prompts=[{"name": "p1"}])
        mgr = _make_manager(a)
        skills = await mcp_prompts_to_skills(mgr)
        assert skills == []

    @pytest.mark.asyncio
    async def test_basic_bridge(self):
        a = _make_conn(
            "github",
            prompts=[
                {
                    "name": "summarise_pr",
                    "description": "Summarise a PR diff",
                    "arguments": [
                        {"name": "pr_id", "description": "PR id", "required": True}
                    ],
                }
            ],
        )
        mgr = _make_manager(a)
        skills = await mcp_prompts_to_skills(mgr)
        assert len(skills) == 1
        s = skills[0]
        assert isinstance(s, Skill)
        assert s.id == "mcp__github__summarise_pr"
        assert s.metadata.name == "summarise_pr"
        assert s.metadata.description == "Summarise a PR diff"
        # Body is a placeholder mentioning the bridge target.
        assert "github" in s.body and "summarise_pr" in s.body
        assert s.metadata.extras == {
            "server": "github",
            "prompt_name": "summarise_pr",
            "arguments": [
                {"name": "pr_id", "description": "PR id", "required": True}
            ],
            "source": "mcp",
        }

    @pytest.mark.asyncio
    async def test_multiple_prompts_across_servers(self):
        a = _make_conn(
            "a",
            prompts=[
                {"name": "p1", "description": "", "arguments": []},
                {"name": "p2", "description": "", "arguments": []},
            ],
        )
        b = _make_conn(
            "b",
            prompts=[{"name": "p3", "description": "", "arguments": []}],
        )
        mgr = _make_manager(a, b)
        skills = await mcp_prompts_to_skills(mgr)
        ids = sorted(s.id for s in skills)
        assert ids == [
            "mcp__a__p1",
            "mcp__a__p2",
            "mcp__b__p3",
        ]

    @pytest.mark.asyncio
    async def test_skips_entries_missing_name_or_server(self):
        # A malformed prompt with no name shouldn't blow up the bridge.
        a = _make_conn(
            "a",
            prompts=[
                {"name": "", "description": "no-name", "arguments": []},
                {"name": "ok", "description": "", "arguments": []},
            ],
        )
        mgr = _make_manager(a)
        skills = await mcp_prompts_to_skills(mgr)
        assert [s.id for s in skills] == ["mcp__a__ok"]

    @pytest.mark.asyncio
    async def test_list_failure_yields_empty_for_that_server(self):
        a = _make_conn("a", list_raises=True)
        b = _make_conn("b", prompts=[{"name": "p", "description": "", "arguments": []}])
        mgr = _make_manager(a, b)
        skills = await mcp_prompts_to_skills(mgr)
        assert [s.id for s in skills] == ["mcp__b__p"]

    @pytest.mark.asyncio
    async def test_default_description_when_missing(self):
        a = _make_conn(
            "a", prompts=[{"name": "noop"}]  # no description / arguments fields
        )
        mgr = _make_manager(a)
        skills = await mcp_prompts_to_skills(mgr)
        assert skills[0].metadata.description == ""
        assert skills[0].metadata.extras["arguments"] == []


# ── Round-trip with real MCPServerConnection.list_prompts surface ──────


class TestConnectionListPrompts:
    @pytest.mark.asyncio
    async def test_disconnected_returns_empty_list(self):
        conn = MCPServerConnection(MCPServerConfig(name="x"))
        # state defaults to PENDING
        assert await conn.list_prompts() == []
        assert await conn.get_prompt("anything") is None
