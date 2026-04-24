"""Phase 3 Week 5 — WebFetch tests.

All network activity is mocked via ``httpx.MockTransport`` so the tests
are deterministic and offline-safe. Tests monkey-patch
``httpx.AsyncClient`` with a factory that installs the mock transport.
"""

from __future__ import annotations

from typing import Any, Dict

import httpx
import pytest

from geny_executor.tools.base import ToolContext
from geny_executor.tools.built_in.web_fetch_tool import (
    _DEFAULT_MAX_CHARS,
    WebFetchTool,
    _extract_text,
    _validate_url,
)


def _install_mock_transport(monkeypatch, handler):
    """Patch httpx.AsyncClient so every WebFetch call routes through ``handler``."""
    real_cls = httpx.AsyncClient

    def _factory(*args, **kwargs):
        kwargs.setdefault("transport", httpx.MockTransport(handler))
        return real_cls(*args, **kwargs)

    monkeypatch.setattr(
        "geny_executor.tools.built_in.web_fetch_tool.httpx.AsyncClient", _factory
    )


def _ctx() -> ToolContext:
    return ToolContext(session_id="s", working_dir="")


# ─────────────────────────────────────────────────────────────────
# URL validation
# ─────────────────────────────────────────────────────────────────


class TestURLValidation:
    def test_empty_url_raises(self):
        with pytest.raises(ValueError):
            _validate_url("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            _validate_url("   ")

    def test_missing_scheme_defaults_to_https(self):
        assert _validate_url("example.com") == "https://example.com"

    def test_rejects_file_scheme(self):
        with pytest.raises(ValueError):
            _validate_url("file:///etc/passwd")

    def test_rejects_ftp_scheme(self):
        with pytest.raises(ValueError):
            _validate_url("ftp://example.com/")

    def test_accepts_http(self):
        assert _validate_url("http://example.com/x") == "http://example.com/x"

    def test_accepts_https(self):
        assert _validate_url("https://example.com/x") == "https://example.com/x"


# ─────────────────────────────────────────────────────────────────
# HTML → text extraction
# ─────────────────────────────────────────────────────────────────


class TestTextExtraction:
    def test_basic_html_stripped(self):
        html_body = "<html><body><p>hello <b>world</b></p></body></html>"
        text, title = _extract_text(html_body, "text/html; charset=utf-8")
        assert "hello" in text and "world" in text
        assert "<p>" not in text
        assert title is None

    def test_title_captured(self):
        html_body = (
            "<html><head><title>My Page</title></head>"
            "<body><p>hi</p></body></html>"
        )
        text, title = _extract_text(html_body, "text/html")
        assert title == "My Page"
        assert "hi" in text

    def test_script_and_style_stripped(self):
        html_body = (
            "<html><head><style>body { color: red; }</style></head>"
            "<body><script>alert('bad')</script><p>only this</p></body></html>"
        )
        text, _ = _extract_text(html_body, "text/html")
        assert "only this" in text
        assert "alert" not in text
        assert "color: red" not in text

    def test_paragraph_breaks_preserved(self):
        html_body = "<p>first</p><p>second</p><p>third</p>"
        text, _ = _extract_text(html_body, "text/html")
        # Paragraphs should be on separate lines (possibly with blank
        # lines between) — just confirm order + separation
        lines = [line for line in text.splitlines() if line]
        assert lines == ["first", "second", "third"]

    def test_entity_decoded(self):
        html_body = "<p>2 &lt; 3 &amp; 4 &gt; 1</p>"
        text, _ = _extract_text(html_body, "text/html")
        assert "2 < 3 & 4 > 1" in text

    def test_non_html_content_returned_as_is(self):
        body = '{"key": "value"}'
        text, title = _extract_text(body, "application/json")
        assert text == body
        assert title is None

    def test_plain_text_returned_as_is(self):
        body = "line 1\nline 2"
        text, title = _extract_text(body, "text/plain")
        assert text == body
        assert title is None


# ─────────────────────────────────────────────────────────────────
# Capabilities
# ─────────────────────────────────────────────────────────────────


class TestCapabilities:
    def test_advertises_parallel_read_network(self):
        caps = WebFetchTool().capabilities({"url": "https://example.com"})
        assert caps.concurrency_safe is True
        assert caps.read_only is True
        assert caps.network_egress is True
        assert caps.destructive is False


# ─────────────────────────────────────────────────────────────────
# Happy-path execution (with MockTransport)
# ─────────────────────────────────────────────────────────────────


class TestExecuteSuccess:
    @pytest.mark.asyncio
    async def test_fetches_and_strips_html(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/doc"
            return httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                content=(
                    "<html><head><title>Doc</title></head>"
                    "<body><h1>Hello</h1><p>World</p></body></html>"
                ).encode("utf-8"),
            )

        _install_mock_transport(monkeypatch, handler)
        tool = WebFetchTool()
        result = await tool.execute({"url": "https://example.com/doc"}, _ctx())
        assert not result.is_error
        assert "Fetched: https://example.com/doc" in result.content
        assert "Title: Doc" in result.content
        assert "Hello" in result.content and "World" in result.content
        assert result.metadata["status_code"] == 200

    @pytest.mark.asyncio
    async def test_injects_default_user_agent(self, monkeypatch):
        captured: Dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["user_agent"] = request.headers.get("user-agent")
            return httpx.Response(200, text="ok", headers={"content-type": "text/plain"})

        _install_mock_transport(monkeypatch, handler)
        await WebFetchTool().execute({"url": "https://example.com/"}, _ctx())
        assert "geny-executor-webfetch" in captured["user_agent"]

    @pytest.mark.asyncio
    async def test_custom_headers_merged(self, monkeypatch):
        captured: Dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["accept"] = request.headers.get("accept")
            captured["auth"] = request.headers.get("authorization")
            return httpx.Response(200, text="ok")

        _install_mock_transport(monkeypatch, handler)
        await WebFetchTool().execute(
            {
                "url": "https://example.com/",
                "headers": {"Authorization": "Bearer abc", "Accept": "application/json"},
            },
            _ctx(),
        )
        assert captured["auth"] == "Bearer abc"
        # Custom Accept header overrides the default
        assert captured["accept"] == "application/json"

    @pytest.mark.asyncio
    async def test_max_chars_truncates(self, monkeypatch):
        big = "x" * 50_000

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, content=big.encode("utf-8"), headers={"content-type": "text/plain"}
            )

        _install_mock_transport(monkeypatch, handler)
        result = await WebFetchTool().execute(
            {"url": "https://example.com/", "max_chars": 100},
            _ctx(),
        )
        assert not result.is_error
        assert "[text truncated at 100 chars]" in result.content
        assert result.metadata["truncated_chars"] is True

    @pytest.mark.asyncio
    async def test_max_bytes_truncates(self, monkeypatch):
        big = b"y" * 5000

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, content=big, headers={"content-type": "text/plain"}
            )

        _install_mock_transport(monkeypatch, handler)
        result = await WebFetchTool().execute(
            {"url": "https://example.com/", "max_bytes": 200},
            _ctx(),
        )
        assert not result.is_error
        assert "[body truncated at 200 bytes]" in result.content
        assert result.metadata["truncated_bytes"] is True

    @pytest.mark.asyncio
    async def test_default_max_chars_leaves_small_page_intact(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, content=b"small", headers={"content-type": "text/plain"}
            )

        _install_mock_transport(monkeypatch, handler)
        result = await WebFetchTool().execute({"url": "https://example.com/"}, _ctx())
        assert not result.is_error
        assert result.metadata["truncated_chars"] is False
        assert result.metadata["text_chars"] <= _DEFAULT_MAX_CHARS


# ─────────────────────────────────────────────────────────────────
# Error paths
# ─────────────────────────────────────────────────────────────────


class TestExecuteErrors:
    @pytest.mark.asyncio
    async def test_invalid_url_returns_error_result(self):
        result = await WebFetchTool().execute({"url": "file:///etc/passwd"}, _ctx())
        assert result.is_error
        assert "invalid URL" in result.content

    @pytest.mark.asyncio
    async def test_empty_url_returns_error_result(self):
        result = await WebFetchTool().execute({"url": ""}, _ctx())
        assert result.is_error

    @pytest.mark.asyncio
    async def test_4xx_returns_error_result(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="not found")

        _install_mock_transport(monkeypatch, handler)
        result = await WebFetchTool().execute({"url": "https://example.com/"}, _ctx())
        assert result.is_error
        assert "HTTP 404" in result.content
        assert result.metadata["status_code"] == 404

    @pytest.mark.asyncio
    async def test_5xx_returns_error_result(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(502, text="bad gateway")

        _install_mock_transport(monkeypatch, handler)
        result = await WebFetchTool().execute({"url": "https://example.com/"}, _ctx())
        assert result.is_error
        assert "HTTP 502" in result.content

    @pytest.mark.asyncio
    async def test_timeout_returns_error_result(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("slow")

        _install_mock_transport(monkeypatch, handler)
        result = await WebFetchTool().execute(
            {"url": "https://example.com/", "timeout": 0.5}, _ctx()
        )
        assert result.is_error
        assert "timeout" in result.content.lower()

    @pytest.mark.asyncio
    async def test_connection_error_returns_error_result(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        _install_mock_transport(monkeypatch, handler)
        result = await WebFetchTool().execute({"url": "https://example.com/"}, _ctx())
        assert result.is_error
        assert "network error" in result.content


# ─────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────


def test_registered_in_built_in_tool_classes():
    from geny_executor.tools.built_in import BUILT_IN_TOOL_CLASSES

    assert "WebFetch" in BUILT_IN_TOOL_CLASSES
    assert BUILT_IN_TOOL_CLASSES["WebFetch"] is WebFetchTool
