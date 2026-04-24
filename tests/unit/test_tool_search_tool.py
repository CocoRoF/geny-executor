"""Phase 3 Week 6 — ToolSearch tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from geny_executor.tools.base import ToolContext
from geny_executor.tools.built_in.tool_search_tool import ToolSearchTool, _rank


def _desc(name: str, description: str = "", schema: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "input_schema": schema or {"type": "object"},
    }


def _ctx_with_tools(tools: List[Dict[str, Any]]) -> ToolContext:
    state_view = SimpleNamespace(tools=tools)
    return ToolContext(session_id="s", working_dir="", state_view=state_view)


class TestRankHeuristic:
    def test_exact_name_beats_substring(self):
        a = _desc("Read", "reads files")
        b = _desc("ReaderThing", "does reading")
        assert _rank(a, "Read") > _rank(b, "Read")

    def test_name_beats_description(self):
        a = _desc("Grep", "other thing")
        b = _desc("Foo", "does grep under the hood")
        assert _rank(a, "grep") > _rank(b, "grep")

    def test_no_match_returns_zero(self):
        assert _rank(_desc("Read", "x"), "nonexistent") == 0

    def test_multi_token_requires_all(self):
        # Both tokens must land somewhere — a miss on one returns 0.
        d = _desc("WebFetch", "fetches HTTP pages")
        assert _rank(d, "web fetch") > 0
        assert _rank(d, "web unrelated") == 0

    def test_schema_property_name_matches(self):
        d = _desc(
            "Foo",
            "a thing",
            schema={
                "type": "object",
                "properties": {"sparkly_key": {"description": "something"}},
            },
        )
        assert _rank(d, "sparkly_key") > 0


class TestSchemaAndCapabilities:
    def test_capabilities_are_safe_read_only(self):
        caps = ToolSearchTool().capabilities({})
        assert caps.concurrency_safe is True
        assert caps.read_only is True
        assert caps.idempotent is True

    def test_schema_has_query(self):
        schema = ToolSearchTool().input_schema
        assert "query" in schema["required"]


class TestExecuteWithStateView:
    @pytest.mark.asyncio
    async def test_returns_ranked_matches(self):
        ctx = _ctx_with_tools(
            [
                _desc("Grep", "Search file contents with regex."),
                _desc("Glob", "Match file paths with shell globbing."),
                _desc("WebFetch", "Fetch an HTTP URL."),
                _desc("Bash", "Run a shell command."),
            ]
        )
        result = await ToolSearchTool().execute({"query": "file"}, ctx)
        assert not result.is_error
        names = [r["name"] for r in result.metadata["results"]]
        # Both Grep and Glob mention "file" in their description — they should be returned.
        assert {"Grep", "Glob"}.issubset(set(names))
        # WebFetch / Bash don't mention file — should be filtered out.
        assert "WebFetch" not in names
        assert "Bash" not in names

    @pytest.mark.asyncio
    async def test_empty_query_errors(self):
        ctx = _ctx_with_tools([])
        result = await ToolSearchTool().execute({"query": ""}, ctx)
        assert result.is_error

    @pytest.mark.asyncio
    async def test_no_matches_returns_empty_count(self):
        ctx = _ctx_with_tools([_desc("Foo", "bar"), _desc("Baz", "qux")])
        result = await ToolSearchTool().execute({"query": "nothing"}, ctx)
        assert not result.is_error
        assert "No matching tools" in result.content
        assert result.metadata["results_count"] == 0

    @pytest.mark.asyncio
    async def test_respects_limit(self):
        tools = [_desc(f"Tool{i}", "utility helper") for i in range(30)]
        ctx = _ctx_with_tools(tools)
        result = await ToolSearchTool().execute(
            {"query": "utility", "limit": 3}, ctx
        )
        assert not result.is_error
        assert result.metadata["results_count"] == 3
        assert "Matching 3 tool(s)" in result.content

    @pytest.mark.asyncio
    async def test_hard_cap_enforced(self):
        tools = [_desc(f"Tool{i}", "utility helper") for i in range(200)]
        ctx = _ctx_with_tools(tools)
        result = await ToolSearchTool().execute(
            {"query": "utility", "limit": 9999}, ctx
        )
        assert not result.is_error
        assert result.metadata["results_count"] == 100  # _HARD_LIMIT

    @pytest.mark.asyncio
    async def test_exact_name_surfaces_first(self):
        ctx = _ctx_with_tools(
            [
                _desc("Explain", "Explains things"),
                _desc("Runner", "Runs things, explains sometimes"),
                _desc("explain_advanced", "More explaining"),
            ]
        )
        result = await ToolSearchTool().execute({"query": "explain"}, ctx)
        first = result.metadata["results"][0]
        assert first["name"] == "Explain"


class TestFallbackWithoutStateView:
    @pytest.mark.asyncio
    async def test_searches_builtin_catalogue_when_no_state_view(self):
        # No state_view → fall back to BUILT_IN_TOOL_CLASSES
        ctx = ToolContext(session_id="s", working_dir="")
        result = await ToolSearchTool().execute({"query": "grep"}, ctx)
        assert not result.is_error
        names = [r["name"] for r in result.metadata["results"]]
        assert "Grep" in names

    @pytest.mark.asyncio
    async def test_fallback_includes_self(self):
        """ToolSearch itself is a built-in — searching for 'tool' should find it."""
        ctx = ToolContext(session_id="s", working_dir="")
        result = await ToolSearchTool().execute({"query": "ToolSearch"}, ctx)
        names = [r["name"] for r in result.metadata["results"]]
        assert "ToolSearch" in names

    @pytest.mark.asyncio
    async def test_empty_state_view_tools_falls_back(self):
        """When state_view.tools is an empty list, we treat it as "no view"
        and fall back to the built-in catalogue — otherwise a host that
        forgot to populate state.tools would get zero results."""
        ctx = _ctx_with_tools([])
        result = await ToolSearchTool().execute({"query": "grep"}, ctx)
        names = [r["name"] for r in result.metadata["results"]]
        assert "Grep" in names


class TestRegistry:
    def test_registered_in_catalog(self):
        from geny_executor.tools.built_in import (
            BUILT_IN_TOOL_CLASSES,
            BUILT_IN_TOOL_FEATURES,
            ToolSearchTool as _Registered,
        )

        assert BUILT_IN_TOOL_CLASSES["ToolSearch"] is _Registered
        assert "ToolSearch" in BUILT_IN_TOOL_FEATURES["meta"]
