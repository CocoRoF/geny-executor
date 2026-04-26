"""PushNotificationTool tests (PR-A.3.2)."""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock
from urllib import error

import pytest

from geny_executor.notifications import (
    NotificationEndpoint,
    NotificationEndpointRegistry,
)
from geny_executor.tools.base import ToolContext
from geny_executor.tools.built_in import PushNotificationTool, BUILT_IN_TOOL_CLASSES


def test_registered():
    assert "PushNotification" in BUILT_IN_TOOL_CLASSES


# ── Registry ─────────────────────────────────────────────────────────


class TestRegistry:
    def test_register_and_get(self):
        reg = NotificationEndpointRegistry()
        ep = NotificationEndpoint(name="alerts", url="https://example.com/hook")
        reg.register(ep)
        assert reg.get("alerts") is ep

    def test_get_missing_returns_none(self):
        assert NotificationEndpointRegistry().get("ghost") is None

    def test_list_returns_all(self):
        reg = NotificationEndpointRegistry()
        reg.register(NotificationEndpoint(name="a", url="u"))
        reg.register(NotificationEndpoint(name="b", url="v"))
        assert {e.name for e in reg.list()} == {"a", "b"}

    def test_overwrite_warns(self):
        reg = NotificationEndpointRegistry()
        reg.register(NotificationEndpoint(name="a", url="u1"))
        reg.register(NotificationEndpoint(name="a", url="u2"))
        assert reg.get("a").url == "u2"


# ── Tool ─────────────────────────────────────────────────────────────


def _ctx_with_endpoint(name="alerts", url="https://example.com/hook", headers=None):
    reg = NotificationEndpointRegistry()
    reg.register(NotificationEndpoint(name=name, url=url, headers=headers))
    return ToolContext(extras={"notification_endpoints": reg})


class TestExecute:
    @pytest.mark.asyncio
    async def test_no_registry(self):
        result = await PushNotificationTool().execute(
            {"endpoint": "x", "message": "hi"}, ToolContext(extras={}),
        )
        assert result.is_error is True
        assert result.content["error"]["code"] == "NO_REGISTRY"

    @pytest.mark.asyncio
    async def test_unknown_endpoint(self):
        ctx = _ctx_with_endpoint("known")
        result = await PushNotificationTool().execute(
            {"endpoint": "unknown", "message": "hi"}, ctx,
        )
        assert result.is_error is True
        assert result.content["error"]["code"] == "UNKNOWN_ENDPOINT"

    @pytest.mark.asyncio
    async def test_sends_webhook(self):
        ctx = _ctx_with_endpoint()
        with patch("urllib.request.urlopen") as urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = lambda self: mock_resp
            mock_resp.__exit__ = lambda *a: False
            urlopen.return_value = mock_resp
            result = await PushNotificationTool().execute(
                {"endpoint": "alerts", "title": "T", "message": "hello"}, ctx,
            )
        assert result.is_error is False
        assert result.content["status"] == 200
        # Verify the body shape.
        sent_req = urlopen.call_args[0][0]
        body = json.loads(sent_req.data)
        assert body["title"] == "T"
        assert body["message"] == "hello"

    @pytest.mark.asyncio
    async def test_includes_custom_headers(self):
        ctx = _ctx_with_endpoint(headers={"X-Auth": "secret"})
        with patch("urllib.request.urlopen") as urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.__enter__ = lambda self: mock_resp
            mock_resp.__exit__ = lambda *a: False
            urlopen.return_value = mock_resp
            await PushNotificationTool().execute(
                {"endpoint": "alerts", "message": "hi"}, ctx,
            )
        sent_req = urlopen.call_args[0][0]
        assert sent_req.headers.get("X-auth") == "secret"

    @pytest.mark.asyncio
    async def test_http_error(self):
        ctx = _ctx_with_endpoint()
        with patch("urllib.request.urlopen", side_effect=error.HTTPError(
            url="x", code=500, msg="Server Error", hdrs=None, fp=None,
        )):
            result = await PushNotificationTool().execute(
                {"endpoint": "alerts", "message": "hi"}, ctx,
            )
        assert result.is_error is True
        assert result.content["error"]["code"] == "WEBHOOK_HTTP"

    @pytest.mark.asyncio
    async def test_connection_error(self):
        ctx = _ctx_with_endpoint()
        with patch("urllib.request.urlopen", side_effect=error.URLError("connection refused")):
            result = await PushNotificationTool().execute(
                {"endpoint": "alerts", "message": "hi"}, ctx,
            )
        assert result.is_error is True
        assert result.content["error"]["code"] == "WEBHOOK_FAILED"
