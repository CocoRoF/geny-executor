"""Default artifact routers for Stage 10: Tool."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import jsonschema

from geny_executor.tools.base import ToolContext, ToolResult
from geny_executor.tools.errors import (
    ToolError,
    ToolFailure,
    make_error_result,
    validate_input,
)
from geny_executor.tools.registry import ToolRegistry
from geny_executor.stages.s10_tool.interface import ToolRouter

logger = logging.getLogger(__name__)


class RegistryRouter(ToolRouter):
    """Routes tool calls via ToolRegistry lookup.

    Every failure mode (unknown tool, invalid input, tool-signaled
    failure, unexpected crash) is converted into a structured
    ``ToolError`` embedded in the ``ToolResult``. No free-form failure
    strings are emitted.
    """

    def __init__(self, registry: Optional[ToolRegistry] = None):
        self._registry = registry or ToolRegistry()

    def bind_registry(self, registry: ToolRegistry) -> None:
        """Swap the backing registry after construction."""
        self._registry = registry

    @property
    def name(self) -> str:
        return "registry"

    @property
    def description(self) -> str:
        return "Routes via ToolRegistry lookup"

    async def route(
        self, tool_name: str, tool_input: Dict[str, Any], context: ToolContext
    ) -> ToolResult:
        tool = self._registry.get(tool_name)
        if tool is None:
            return make_error_result(
                ToolError.unknown_tool(tool_name, known=self._registry.list_names())
            )

        try:
            validate_input(tool.input_schema, tool_input)
        except jsonschema.ValidationError as exc:
            path = ".".join(str(p) for p in exc.absolute_path) or "<root>"
            return make_error_result(ToolError.invalid_input(tool_name, exc.message, path=path))

        try:
            return await tool.execute(tool_input, context)
        except ToolFailure as failure:
            logger.info(
                "tool %s raised ToolFailure (%s): %s",
                tool_name,
                failure.error.code.value,
                failure.error.message,
            )
            return make_error_result(failure.error)
        except Exception as exc:
            logger.exception("tool %s crashed unexpectedly", tool_name)
            return make_error_result(ToolError.tool_crashed(tool_name, exc))
