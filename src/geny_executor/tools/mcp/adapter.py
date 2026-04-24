"""MCP tool adapter — wraps MCP server tools as Tool interface."""

from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

from geny_executor.tools.base import Tool, ToolCapabilities, ToolResult, ToolContext
from geny_executor.tools.errors import ToolFailure, ToolErrorCode

if TYPE_CHECKING:
    from geny_executor.tools.mcp.manager import MCPServerConnection


MCP_TOOL_PREFIX_FORMAT = "mcp__{server}__{tool}"
MCP_TOOL_PREFIX = "mcp__"


def _annotations_to_capabilities(annotations: Dict[str, Any]) -> ToolCapabilities:
    """Translate MCP tool annotations into :class:`ToolCapabilities`.

    The MCP spec defines four optional behavioural hints on each tool's
    ``annotations`` block. Mapping into the executor's capability
    surface:

    * ``readOnlyHint=True``  → ``read_only=True`` + ``concurrency_safe=True``.
      Read-only tools have no observable state effects; safe to fan
      out under PartitionExecutor / StreamingToolExecutor.
    * ``destructiveHint=True`` → ``destructive=True`` and
      ``concurrency_safe=False`` (overrides). Destructive tools serialise
      regardless of any other hint.
    * ``idempotentHint=True`` → ``idempotent=True``. Repeat-safe.
    * ``openWorldHint=True``  → ``network_egress=True``. Tool reaches
      out to external systems.

    Missing or absent annotations fall back to the fail-closed default
    (``concurrency_safe=False``) — same behaviour as 0.37.x and
    earlier, no regression.
    """
    if not annotations:
        return ToolCapabilities()

    read_only = bool(annotations.get("readOnlyHint", False))
    destructive = bool(annotations.get("destructiveHint", False))
    idempotent = bool(annotations.get("idempotentHint", False))
    network_egress = bool(annotations.get("openWorldHint", False))

    # Destructive overrides concurrency_safe even when the server also
    # set readOnlyHint by mistake — better safe than sorry.
    concurrency_safe = read_only and not destructive

    return ToolCapabilities(
        concurrency_safe=concurrency_safe,
        read_only=read_only,
        destructive=destructive,
        idempotent=idempotent,
        network_egress=network_egress,
    )


class MCPToolAdapter(Tool):
    """Wraps an MCP server tool as a geny-executor Tool.

    Bridges the MCP protocol to our Tool interface. The adapter exposes
    the tool to the registry under a namespaced display name —
    ``mcp__{server}__{original}`` — so no two servers (and no built-in
    or adhoc tool) can ever collide. The original, unprefixed name is
    preserved internally and used when calling the MCP server.
    """

    def __init__(
        self,
        server: MCPServerConnection,
        definition: Dict[str, Any],
    ):
        self._server = server
        self._definition = definition
        raw_name = definition.get("name", "unknown_mcp_tool")
        self._raw_name = raw_name
        self._display_name = MCP_TOOL_PREFIX_FORMAT.format(server=server.config.name, tool=raw_name)

    @property
    def raw_name(self) -> str:
        """Unprefixed name as announced by the MCP server."""
        return self._raw_name

    @property
    def server_name(self) -> str:
        return self._server.config.name

    @property
    def name(self) -> str:
        return self._display_name

    @property
    def description(self) -> str:
        return self._definition.get("description", "MCP tool")

    @property
    def input_schema(self) -> Dict[str, Any]:
        return self._definition.get(
            "inputSchema",
            self._definition.get(
                "input_schema",
                {
                    "type": "object",
                    "properties": {},
                },
            ),
        )

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        """Translate the server-supplied annotations into capabilities.

        Lets PartitionExecutor / StreamingToolExecutor parallelise
        read-only MCP tools instead of treating every external tool
        as fail-closed unsafe. See ``_annotations_to_capabilities``
        for the mapping.
        """
        annotations = self._definition.get("annotations") or {}
        return _annotations_to_capabilities(annotations)

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        """Execute the tool via the MCP server.

        MCP-side failures surface as ``ToolFailure`` (transport code) so
        the router converts them into a structured ``ToolError`` rather
        than leaking raw exception strings into the model context.
        """
        try:
            result = await self._server.call_tool(self._raw_name, input)
        except Exception as exc:
            raise ToolFailure(
                f"MCP call failed on '{self.server_name}.{self._raw_name}': {exc}",
                code=ToolErrorCode.TRANSPORT,
                details={
                    "server": self.server_name,
                    "tool": self._raw_name,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            ) from exc

        # call_tool returns str for a single text block and list[dict] for
        # multi-block / non-text content. ToolResult.content is Any and
        # ToolResult.to_api_format(...) already handles both — pass through.
        if isinstance(result, (str, list)):
            return ToolResult(content=result, is_error=False)
        return ToolResult(content=str(result), is_error=False)
