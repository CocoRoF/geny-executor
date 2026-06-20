"""Tests for pluggable WebSearch backends (ddg / brave / tavily / searxng).

Backend selection precedence, unknown-backend error, and the HTTP-API
backends (Brave + Tavily) happy-path / missing-key paths. The API
backends are exercised against a mocked ``httpx`` transport so nothing
touches the network.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx
import pytest

from geny_executor.tools.base import ToolContext
from geny_executor.tools.built_in import _web_search_backends as backends
from geny_executor.tools.built_in.web_search_tool import WebSearchTool


def _ctx(extras: Dict[str, Any] | None = None) -> ToolContext:
    return ToolContext(session_id="s", working_dir="", extras=extras or {})


# ─────────────────────────────────────────────────────────────────
# Mocked httpx transport helpers
# ─────────────────────────────────────────────────────────────────


def _install_mock_transport(monkeypatch, handler) -> List[httpx.Request]:
    """Patch ``httpx.AsyncClient`` to route through ``MockTransport``.

    ``handler`` receives an ``httpx.Request`` and returns an
    ``httpx.Response``. Returns a list that accumulates every request
    so tests can assert on URL / headers / body.
    """
    seen: List[httpx.Request] = []

    def _routing(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    real_client = httpx.AsyncClient

    def _factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(_routing)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(backends.httpx, "AsyncClient", _factory)
    return seen


# ─────────────────────────────────────────────────────────────────
# Backend selection precedence
# ─────────────────────────────────────────────────────────────────


class TestSelection:
    def test_default_is_ddg(self, monkeypatch):
        monkeypatch.delenv("GENY_WEBSEARCH_BACKEND", raising=False)
        assert backends.select_backend_name({}, _ctx()) == "ddg"

    def test_env_overrides_default(self, monkeypatch):
        monkeypatch.setenv("GENY_WEBSEARCH_BACKEND", "brave")
        assert backends.select_backend_name({}, _ctx()) == "brave"

    def test_extras_overrides_env(self, monkeypatch):
        monkeypatch.setenv("GENY_WEBSEARCH_BACKEND", "brave")
        ctx = _ctx({"web_search": {"backend": "tavily"}})
        assert backends.select_backend_name({}, ctx) == "tavily"

    def test_input_overrides_extras(self, monkeypatch):
        monkeypatch.setenv("GENY_WEBSEARCH_BACKEND", "brave")
        ctx = _ctx({"web_search": {"backend": "tavily"}})
        assert backends.select_backend_name({"backend": "searxng"}, ctx) == "searxng"

    def test_case_and_whitespace_normalised(self, monkeypatch):
        monkeypatch.delenv("GENY_WEBSEARCH_BACKEND", raising=False)
        assert backends.select_backend_name({"backend": "  Brave "}, _ctx()) == "brave"


class TestUnknownBackend:
    def test_build_unknown_raises_config_error(self):
        with pytest.raises(backends.WebSearchConfigError) as exc:
            backends.build_backend("bing", _ctx())
        msg = str(exc.value)
        assert "Unknown WebSearch backend" in msg
        for name in backends.BACKEND_NAMES:
            assert name in msg

    @pytest.mark.asyncio
    async def test_tool_unknown_backend_returns_error(self, monkeypatch):
        monkeypatch.delenv("GENY_WEBSEARCH_BACKEND", raising=False)
        result = await WebSearchTool().execute(
            {"query": "x", "backend": "bing"}, _ctx()
        )
        assert result.is_error
        assert "Unknown WebSearch backend" in result.content
        assert "ddg" in result.content


# ─────────────────────────────────────────────────────────────────
# Brave backend
# ─────────────────────────────────────────────────────────────────


class TestBrave:
    @pytest.mark.asyncio
    async def test_happy_path(self, monkeypatch):
        def _handler(request: httpx.Request) -> httpx.Response:
            payload = {
                "web": {
                    "results": [
                        {
                            "title": "Brave Result",
                            "url": "https://example.com/brave",
                            "description": "A snippet from Brave.",
                        },
                        {
                            "title": "Second",
                            "url": "https://example.com/2",
                            "description": "Another.",
                        },
                    ]
                }
            }
            return httpx.Response(200, json=payload)

        seen = _install_mock_transport(monkeypatch, _handler)
        ctx = _ctx({"web_search": {"brave_api_key": "k-123"}})
        backend = backends.build_backend("brave", ctx)
        hits = await backend.search("foo", max_results=5, region="wt-wt", safesearch="moderate")

        assert len(hits) == 2
        assert hits[0] == {
            "rank": 0,
            "title": "Brave Result",
            "url": "https://example.com/brave",
            "snippet": "A snippet from Brave.",
        }
        # Auth header + query params went out correctly.
        req = seen[0]
        assert req.headers["X-Subscription-Token"] == "k-123"
        assert "q=foo" in str(req.url)
        assert "api.search.brave.com" in str(req.url)

    @pytest.mark.asyncio
    async def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "env-key")

        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"web": {"results": []}})

        seen = _install_mock_transport(monkeypatch, _handler)
        backend = backends.build_backend("brave", _ctx())
        await backend.search("q", 5, "wt-wt", "moderate")
        assert seen[0].headers["X-Subscription-Token"] == "env-key"

    @pytest.mark.asyncio
    async def test_missing_key_raises_config_error(self, monkeypatch):
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
        backend = backends.build_backend("brave", _ctx())
        with pytest.raises(backends.WebSearchConfigError) as exc:
            await backend.search("q", 5, "wt-wt", "moderate")
        assert "BRAVE_SEARCH_API_KEY" in str(exc.value)
        assert "brave_api_key" in str(exc.value)

    @pytest.mark.asyncio
    async def test_tool_brave_missing_key_returns_error(self, monkeypatch):
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
        monkeypatch.delenv("GENY_WEBSEARCH_BACKEND", raising=False)
        result = await WebSearchTool().execute(
            {"query": "x", "backend": "brave"}, _ctx()
        )
        assert result.is_error
        assert "Brave backend requires an API key" in result.content

    @pytest.mark.asyncio
    async def test_tool_brave_happy_path(self, monkeypatch):
        def _handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "web": {
                        "results": [
                            {
                                "title": "T",
                                "url": "https://x/1",
                                "description": "snip",
                            }
                        ]
                    }
                },
            )

        _install_mock_transport(monkeypatch, _handler)
        ctx = _ctx({"web_search": {"backend": "brave", "brave_api_key": "k"}})
        result = await WebSearchTool().execute({"query": "hi"}, ctx)
        assert not result.is_error
        assert "1. T" in result.content
        assert "https://x/1" in result.content
        assert result.metadata["results_count"] == 1
        assert result.metadata["results"][0]["snippet"] == "snip"


# ─────────────────────────────────────────────────────────────────
# Tavily backend
# ─────────────────────────────────────────────────────────────────


class TestTavily:
    @pytest.mark.asyncio
    async def test_happy_path(self, monkeypatch):
        captured: Dict[str, Any] = {}

        def _handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content.decode())
            payload = {
                "results": [
                    {
                        "title": "Tavily Hit",
                        "url": "https://example.com/t",
                        "content": "Tavily snippet.",
                    }
                ]
            }
            return httpx.Response(200, json=payload)

        seen = _install_mock_transport(monkeypatch, _handler)
        ctx = _ctx({"web_search": {"tavily_api_key": "tav-key"}})
        backend = backends.build_backend("tavily", ctx)
        hits = await backend.search("foo", 5, "wt-wt", "moderate")

        assert hits == [
            {
                "rank": 0,
                "title": "Tavily Hit",
                "url": "https://example.com/t",
                "snippet": "Tavily snippet.",
            }
        ]
        # POST body carries api_key + query.
        assert captured["body"]["api_key"] == "tav-key"
        assert captured["body"]["query"] == "foo"
        assert "api.tavily.com" in str(seen[0].url)

    @pytest.mark.asyncio
    async def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "env-tav")
        captured: Dict[str, Any] = {}

        def _handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content.decode())
            return httpx.Response(200, json={"results": []})

        _install_mock_transport(monkeypatch, _handler)
        backend = backends.build_backend("tavily", _ctx())
        await backend.search("q", 5, "wt-wt", "moderate")
        assert captured["body"]["api_key"] == "env-tav"

    @pytest.mark.asyncio
    async def test_missing_key_raises_config_error(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        backend = backends.build_backend("tavily", _ctx())
        with pytest.raises(backends.WebSearchConfigError) as exc:
            await backend.search("q", 5, "wt-wt", "moderate")
        assert "TAVILY_API_KEY" in str(exc.value)


# ─────────────────────────────────────────────────────────────────
# SearXNG backend
# ─────────────────────────────────────────────────────────────────


class TestSearxng:
    @pytest.mark.asyncio
    async def test_happy_path(self, monkeypatch):
        def _handler(request: httpx.Request) -> httpx.Response:
            payload = {
                "results": [
                    {
                        "title": "Searx Hit",
                        "url": "https://example.com/s",
                        "content": "Searx snippet.",
                    }
                ]
            }
            return httpx.Response(200, json=payload)

        seen = _install_mock_transport(monkeypatch, _handler)
        ctx = _ctx({"web_search": {"searxng_url": "https://searx.example.org/"}})
        backend = backends.build_backend("searxng", ctx)
        hits = await backend.search("foo", 5, "wt-wt", "moderate")

        assert hits[0]["title"] == "Searx Hit"
        assert hits[0]["url"] == "https://example.com/s"
        # Trailing slash stripped + /search path + json format.
        url = str(seen[0].url)
        assert "https://searx.example.org/search" in url
        assert "format=json" in url

    @pytest.mark.asyncio
    async def test_missing_url_raises_config_error(self, monkeypatch):
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        backend = backends.build_backend("searxng", _ctx())
        with pytest.raises(backends.WebSearchConfigError) as exc:
            await backend.search("q", 5, "wt-wt", "moderate")
        assert "SEARXNG_URL" in str(exc.value)


# ─────────────────────────────────────────────────────────────────
# ddg backend remains default + delegates to monkeypatchable hooks
# ─────────────────────────────────────────────────────────────────


class TestDdgDefault:
    @pytest.mark.asyncio
    async def test_ddg_used_when_no_backend_specified(self, monkeypatch):
        monkeypatch.delenv("GENY_WEBSEARCH_BACKEND", raising=False)
        # Pretend ddgs is installed.
        monkeypatch.setattr(
            "geny_executor.tools.built_in.web_search_tool._load_ddgs",
            lambda: object,
        )

        def _fake(ddgs_cls, query, max_results, region, safesearch):
            return [{"title": "D", "href": "https://d/1", "body": "snip"}]

        monkeypatch.setattr(WebSearchTool, "_search_sync", staticmethod(_fake))
        result = await WebSearchTool().execute({"query": "q"}, _ctx())
        assert not result.is_error
        assert "1. D" in result.content
        assert result.metadata["results"][0]["url"] == "https://d/1"

    @pytest.mark.asyncio
    async def test_ddg_missing_package_install_hint(self, monkeypatch):
        monkeypatch.delenv("GENY_WEBSEARCH_BACKEND", raising=False)
        monkeypatch.setattr(
            "geny_executor.tools.built_in.web_search_tool._load_ddgs",
            lambda: None,
        )
        result = await WebSearchTool().execute(
            {"query": "x", "backend": "ddg"}, _ctx()
        )
        assert result.is_error
        assert "pip install" in result.content
        assert "geny-executor[web]" in result.content
