"""``SendMessage`` — dispatch a message via a registered channel (PR-A.3.7)."""

from __future__ import annotations

from typing import Any, Dict

from geny_executor.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult


class SendMessageTool(Tool):
    @property
    def name(self) -> str:
        return "SendMessage"

    @property
    def description(self) -> str:
        return (
            "Send a message via a registered SendMessageChannel "
            "(Discord / Slack / SMS / etc — channel impl is host-specific)."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "channel": {"type": "string"},
                "to": {"type": "string"},
                "message": {"type": "string", "minLength": 1, "maxLength": 4000},
                "attachments": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["channel", "message"],
        }

    def capabilities(self, input):
        return ToolCapabilities(concurrency_safe=True, network_egress=True)

    async def execute(self, input, context):
        registry = context.extras.get("send_message_channels")
        if registry is None:
            return ToolResult(
                content={"error": {"code": "NO_REGISTRY", "message": "send_message_channels not wired"}},
                is_error=True,
            )
        channel_name = input["channel"]
        channel = registry.get(channel_name)
        if channel is None:
            return ToolResult(
                content={"error": {"code": "UNKNOWN_CHANNEL", "message": f"unknown channel: {channel_name}"}},
                is_error=True,
            )
        try:
            result = await channel.send(
                to=input.get("to"),
                message=input["message"],
                attachments=input.get("attachments") or [],
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                content={"error": {"code": "SEND_FAILED", "message": str(exc)}},
                is_error=True,
            )
        return ToolResult(content={"channel": channel_name, "result": result})


__all__ = ["SendMessageTool"]
