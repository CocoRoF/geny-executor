"""MCP wrapper tools — let the LLM use MCP without server-specific tools (PR-A.3.3).

Four tools that target the host's MCPManager via
``ctx.extras["mcp_manager"]``. The framework already has full
transport / FSM / OAuth / URI handling; these are just LLM-facing
wrappers so the model can:

* ``MCPTool``               — call ``server::tool`` with arguments
* ``ListMcpResources``      — discover what's available across servers
* ``ReadMcpResource``       — read a ``mcp://`` URI
* ``McpAuth``               — kick off OAuth for a server requiring auth

Each tool probes a small set of method names on the manager so it
works against varying MCPManager shapes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from geny_executor.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult

logger = logging.getLogger(__name__)


def _mgr(ctx: ToolContext) -> Optional[Any]:
    return ctx.extras.get("mcp_manager")


def _err(code: str, message: str) -> ToolResult:
    return ToolResult(
        content={"error": {"code": code, "message": message}},
        is_error=True,
    )


async def _try_call(obj: Any, candidates: List[str], *args, **kwargs):
    """Call the first available method name from ``candidates``. Awaits if
    the result is awaitable. Raises AttributeError if none match."""
    for name in candidates:
        fn = getattr(obj, name, None)
        if callable(fn):
            result = fn(*args, **kwargs)
            if hasattr(result, "__await__"):
                return await result
            return result
    raise AttributeError(f"none of {candidates} on {type(obj).__name__}")


# ── MCPTool ──────────────────────────────────────────────────────────


class MCPTool(Tool):
    @property
    def name(self) -> str:
        return "MCP"

    @property
    def description(self) -> str:
        return (
            "Call a tool exposed by a registered MCP server. "
            "Use ListMcpResources to discover available tools."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "server": {"type": "string"},
                "tool": {"type": "string"},
                "arguments": {"type": "object"},
            },
            "required": ["server", "tool"],
        }

    def capabilities(self, input):
        return ToolCapabilities(concurrency_safe=True, network_egress=True)

    async def execute(self, input, context):
        mgr = _mgr(context)
        if mgr is None:
            return _err("NO_MANAGER", "mcp_manager not wired into ctx.extras")
        try:
            result = await _try_call(
                mgr, ["call_tool", "invoke_tool", "call"],
                server_name=input["server"],
                tool_name=input["tool"],
                arguments=input.get("arguments", {}),
            )
        except AttributeError:
            # Fallback to positional.
            try:
                result = await _try_call(
                    mgr, ["call_tool", "invoke_tool", "call"],
                    input["server"], input["tool"], input.get("arguments", {}),
                )
            except Exception as exc:  # noqa: BLE001
                return _err("MCP_CALL_FAILED", str(exc))
        except Exception as exc:  # noqa: BLE001
            return _err("MCP_CALL_FAILED", str(exc))
        return ToolResult(content={"server": input["server"], "tool": input["tool"], "result": result})


# ── ListMcpResources ─────────────────────────────────────────────────


class ListMcpResourcesTool(Tool):
    @property
    def name(self) -> str:
        return "ListMcpResources"

    @property
    def description(self) -> str:
        return "List resources / tools / prompts exposed by registered MCP servers."

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "server": {"type": "string"},
                "kind": {"enum": ["tool", "resource", "prompt"]},
            },
        }

    def capabilities(self, input):
        return ToolCapabilities(concurrency_safe=True, read_only=True, idempotent=True)

    async def execute(self, input, context):
        mgr = _mgr(context)
        if mgr is None:
            return _err("NO_MANAGER", "mcp_manager not wired into ctx.extras")
        try:
            resources = await _try_call(
                mgr, ["list_resources", "list_all", "describe"],
                server=input.get("server"),
                kind=input.get("kind"),
            )
        except (AttributeError, TypeError):
            try:
                resources = await _try_call(mgr, ["list_resources", "list_all", "describe"])
            except Exception as exc:  # noqa: BLE001
                return _err("MCP_LIST_FAILED", str(exc))
        except Exception as exc:  # noqa: BLE001
            return _err("MCP_LIST_FAILED", str(exc))
        return ToolResult(content={"resources": list(resources or [])})


# ── ReadMcpResource ──────────────────────────────────────────────────


class ReadMcpResourceTool(Tool):
    @property
    def name(self) -> str:
        return "ReadMcpResource"

    @property
    def description(self) -> str:
        return "Read content of an MCP resource by mcp:// URI."

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {"uri": {"type": "string"}}, "required": ["uri"]}

    def capabilities(self, input):
        return ToolCapabilities(concurrency_safe=True, read_only=True, idempotent=True, network_egress=True)

    async def execute(self, input, context):
        mgr = _mgr(context)
        if mgr is None:
            return _err("NO_MANAGER", "mcp_manager not wired into ctx.extras")
        try:
            content = await _try_call(mgr, ["read_resource", "read", "fetch"], uri=input["uri"])
        except (AttributeError, TypeError):
            try:
                content = await _try_call(mgr, ["read_resource", "read", "fetch"], input["uri"])
            except Exception as exc:  # noqa: BLE001
                return _err("MCP_READ_FAILED", str(exc))
        except Exception as exc:  # noqa: BLE001
            return _err("MCP_READ_FAILED", str(exc))
        return ToolResult(content={"uri": input["uri"], "content": content})


# ── McpAuth ──────────────────────────────────────────────────────────


class McpAuthTool(Tool):
    @property
    def name(self) -> str:
        return "McpAuth"

    @property
    def description(self) -> str:
        return "Trigger OAuth flow for an MCP server requiring auth. Returns auth URL."

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {"server": {"type": "string"}}, "required": ["server"]}

    def capabilities(self, input):
        return ToolCapabilities(concurrency_safe=False, network_egress=True)

    async def execute(self, input, context):
        mgr = _mgr(context)
        if mgr is None:
            return _err("NO_MANAGER", "mcp_manager not wired into ctx.extras")
        try:
            status = await _try_call(
                mgr, ["start_oauth", "begin_oauth", "auth"],
                server_name=input["server"],
            )
        except (AttributeError, TypeError):
            try:
                status = await _try_call(
                    mgr, ["start_oauth", "begin_oauth", "auth"], input["server"],
                )
            except Exception as exc:  # noqa: BLE001
                return _err("MCP_AUTH_FAILED", str(exc))
        except Exception as exc:  # noqa: BLE001
            return _err("MCP_AUTH_FAILED", str(exc))
        # Status can be a dict, an object with .url/.state, or just a URL str.
        url = getattr(status, "url", None) if not isinstance(status, dict) else status.get("url")
        if url is None and isinstance(status, str):
            url = status
        return ToolResult(
            content={
                "server": input["server"],
                "auth_url": url,
                "raw": (status if isinstance(status, (dict, str)) else str(status)),
                "instructions": "Visit auth_url and complete authorization.",
            },
        )


__all__ = [
    "MCPTool",
    "ListMcpResourcesTool",
    "ReadMcpResourceTool",
    "McpAuthTool",
]
