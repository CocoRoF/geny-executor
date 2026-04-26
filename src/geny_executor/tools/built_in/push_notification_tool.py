"""``PushNotification`` — fire a webhook to a registered endpoint (PR-A.3.2)."""

from __future__ import annotations

import json
from typing import Any, Dict
from urllib import error, request

from geny_executor.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult


class PushNotificationTool(Tool):
    @property
    def name(self) -> str:
        return "PushNotification"

    @property
    def description(self) -> str:
        return (
            "Send a notification to a configured webhook endpoint. "
            "Endpoints are pre-registered by the host (name → URL + headers)."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "endpoint": {"type": "string"},
                "title": {"type": "string"},
                "message": {"type": "string", "minLength": 1, "maxLength": 2000},
                "metadata": {"type": "object"},
            },
            "required": ["endpoint", "message"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(
            concurrency_safe=True,
            network_egress=True,
            idempotent=False,
        )

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        registry = context.extras.get("notification_endpoints")
        if registry is None:
            return ToolResult(
                content={
                    "error": {"code": "NO_REGISTRY", "message": "notification_endpoints not wired"}
                },
                is_error=True,
            )
        endpoint_name = input.get("endpoint", "")
        endpoint = registry.get(endpoint_name)
        if endpoint is None:
            return ToolResult(
                content={
                    "error": {
                        "code": "UNKNOWN_ENDPOINT",
                        "message": f"unknown endpoint: {endpoint_name}",
                    }
                },
                is_error=True,
            )
        body = json.dumps(
            {
                "title": input.get("title", "Notification"),
                "message": input["message"],
                "metadata": input.get("metadata") or {},
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if endpoint.headers:
            headers.update(endpoint.headers)
        req = request.Request(endpoint.url, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=10) as resp:
                status = resp.status
        except error.HTTPError as exc:
            return ToolResult(
                content={
                    "error": {"code": "WEBHOOK_HTTP", "message": f"HTTP {exc.code}: {exc.reason}"}
                },
                is_error=True,
            )
        except error.URLError as exc:
            return ToolResult(
                content={"error": {"code": "WEBHOOK_FAILED", "message": str(exc.reason)}},
                is_error=True,
            )
        return ToolResult(content={"endpoint": endpoint_name, "status": status, "sent": True})


__all__ = ["PushNotificationTool"]
