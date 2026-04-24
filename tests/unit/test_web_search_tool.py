"""Phase 3 Week 5 — WebSearch tests.

Tests stub the ``ddgs`` client so we don't hit the live network.
``_search_sync`` is monkey-patched to return canned results, so the
test suite stays deterministic and works even when the ``[web]`` extra
is not installed.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from geny_executor.tools.base import ToolContext
from geny_executor.tools.built_in.web_search_tool import (
    _DEFAULT_MAX_RESULTS,
    _HARD_MAX_RESULTS,
    WebSearchTool,
)


def _ctx() -> ToolContext:
    return ToolContext(session_id="s", working_dir="")


def _stub_search(results: List[Dict[str, Any]]):
    """Return a replacement for ``_search_sync`` that yields ``results``.

    Signature matches the real static method so monkey-patching is a
    drop-in.
    """

    def _fake(ddgs_cls, query, max_results, region, safesearch):
        return list(results[:max_results])

    return _fake


@pytest.fixture
def ddgs_available(monkeypatch):
    """Pretend ddgs is installed so tests don't depend on the [web] extra.

    The real ``_load_ddgs`` returns the actual DDGS class when the
    package is installed. Our tests monkey-patch ``_search_sync`` to
    return canned data, so we only need a non-None sentinel here —
    the sentinel never has any methods called on it.
    """
    monkeypatch.setattr(
        "geny_executor.tools.built_in.web_search_tool._load_ddgs",
        lambda: object,
    )


class TestSchemaAndCapabilities:
    def test_capabilities(self):
        caps = WebSearchTool().capabilities({"query": "x"})
        assert caps.concurrency_safe is True
        assert caps.read_only is True
        assert caps.network_egress is True
        assert caps.destructive is False

    def test_schema_has_query(self):
        schema = WebSearchTool().input_schema
        assert schema["properties"]["query"]["minLength"] == 1
        assert "query" in schema["required"]
        assert schema["properties"]["safesearch"]["enum"] == ["on", "moderate", "off"]


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_returns_formatted_hits(self, monkeypatch, ddgs_available):
        monkeypatch.setattr(
            WebSearchTool,
            "_search_sync",
            staticmethod(
                _stub_search(
                    [
                        {
                            "title": "Python docs",
                            "href": "https://docs.python.org/",
                            "body": "Official documentation",
                        },
                        {
                            "title": "PEP 8",
                            "href": "https://peps.python.org/pep-0008/",
                            "body": "Style guide",
                        },
                    ]
                )
            ),
        )
        result = await WebSearchTool().execute({"query": "python"}, _ctx())
        assert not result.is_error
        assert "Python docs" in result.content
        assert "https://peps.python.org/pep-0008/" in result.content
        assert result.metadata["results_count"] == 2
        assert result.metadata["results"][0]["url"] == "https://docs.python.org/"
        # Rank is zero-based in metadata, one-based in rendered content
        assert result.metadata["results"][0]["rank"] == 0
        assert "1. Python docs" in result.content
        assert "2. PEP 8" in result.content

    @pytest.mark.asyncio
    async def test_respects_max_results(self, monkeypatch, ddgs_available):
        many = [
            {"title": f"T{i}", "href": f"https://x/{i}", "body": f"B{i}"}
            for i in range(20)
        ]
        monkeypatch.setattr(
            WebSearchTool, "_search_sync", staticmethod(_stub_search(many))
        )
        result = await WebSearchTool().execute(
            {"query": "x", "max_results": 3}, _ctx()
        )
        assert result.metadata["results_count"] == 3
        assert "1. T0" in result.content
        assert "3. T2" in result.content
        assert "T3" not in result.content

    @pytest.mark.asyncio
    async def test_hard_cap_enforced(self, monkeypatch, ddgs_available):
        # Ask for more than the hard cap — should silently clamp.
        many = [
            {"title": f"T{i}", "href": f"https://x/{i}", "body": ""}
            for i in range(_HARD_MAX_RESULTS + 20)
        ]
        captured: Dict[str, int] = {}

        def _spy(ddgs_cls, query, max_results, region, safesearch):
            captured["max_results"] = max_results
            return many[:max_results]

        monkeypatch.setattr(WebSearchTool, "_search_sync", staticmethod(_spy))
        result = await WebSearchTool().execute(
            {"query": "x", "max_results": 9999}, _ctx()
        )
        assert captured["max_results"] == _HARD_MAX_RESULTS
        assert result.metadata["results_count"] == _HARD_MAX_RESULTS

    @pytest.mark.asyncio
    async def test_empty_results_produces_no_results_message(
        self, monkeypatch, ddgs_available
    ):
        monkeypatch.setattr(
            WebSearchTool, "_search_sync", staticmethod(_stub_search([]))
        )
        result = await WebSearchTool().execute({"query": "nothing"}, _ctx())
        assert not result.is_error
        assert "No results" in result.content
        assert result.metadata["results_count"] == 0

    @pytest.mark.asyncio
    async def test_default_max_results_applied(self, monkeypatch, ddgs_available):
        captured: Dict[str, int] = {}

        def _spy(ddgs_cls, query, max_results, region, safesearch):
            captured["max_results"] = max_results
            return []

        monkeypatch.setattr(WebSearchTool, "_search_sync", staticmethod(_spy))
        await WebSearchTool().execute({"query": "foo"}, _ctx())
        assert captured["max_results"] == _DEFAULT_MAX_RESULTS

    @pytest.mark.asyncio
    async def test_region_and_safesearch_forwarded(self, monkeypatch, ddgs_available):
        captured: Dict[str, Any] = {}

        def _spy(ddgs_cls, query, max_results, region, safesearch):
            captured["region"] = region
            captured["safesearch"] = safesearch
            return []

        monkeypatch.setattr(WebSearchTool, "_search_sync", staticmethod(_spy))
        await WebSearchTool().execute(
            {"query": "foo", "region": "kr-kr", "safesearch": "off"}, _ctx()
        )
        assert captured == {"region": "kr-kr", "safesearch": "off"}


class TestErrorPaths:
    @pytest.mark.asyncio
    async def test_empty_query_error(self):
        result = await WebSearchTool().execute({"query": ""}, _ctx())
        assert result.is_error
        assert "must not be empty" in result.content

    @pytest.mark.asyncio
    async def test_whitespace_query_error(self):
        result = await WebSearchTool().execute({"query": "   "}, _ctx())
        assert result.is_error

    @pytest.mark.asyncio
    async def test_missing_ddgs_gives_install_hint(self, monkeypatch):
        monkeypatch.setattr(
            "geny_executor.tools.built_in.web_search_tool._load_ddgs",
            lambda: None,
        )
        result = await WebSearchTool().execute({"query": "x"}, _ctx())
        assert result.is_error
        assert "pip install" in result.content
        assert "geny-executor[web]" in result.content

    @pytest.mark.asyncio
    async def test_ddg_exception_surfaces_as_error_result(
        self, monkeypatch, ddgs_available
    ):
        def _boom(ddgs_cls, query, max_results, region, safesearch):
            raise RuntimeError("rate limited")

        monkeypatch.setattr(WebSearchTool, "_search_sync", staticmethod(_boom))
        result = await WebSearchTool().execute({"query": "x"}, _ctx())
        assert result.is_error
        assert "web search failed" in result.content


class TestRegistry:
    def test_registered_in_built_in_tool_classes(self):
        from geny_executor.tools.built_in import BUILT_IN_TOOL_CLASSES, WebSearchTool

        assert "WebSearch" in BUILT_IN_TOOL_CLASSES
        assert BUILT_IN_TOOL_CLASSES["WebSearch"] is WebSearchTool
