"""Default artifact routers for Stage 10: Tool."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import jsonschema

from geny_executor.tools.base import Tool, ToolContext, ToolResult
from geny_executor.tools.errors import (
    ToolError,
    ToolFailure,
    make_error_result,
    validate_input,
)
from geny_executor.tools.registry import ToolRegistry
from geny_executor.stages.s10_tool.interface import ToolRouter

logger = logging.getLogger(__name__)


async def _fire_hook(
    hook_name: str,
    tool_name: str,
    tool: Any,
    *hook_args: Any,
) -> None:
    """Call ``tool.<hook_name>(*hook_args)`` defensively.

    Lifecycle hooks are observers (``on_enter`` / ``on_exit`` / ``on_error``):

    * A tool that doesn't declare the hook at all (duck-typed adapters
      that implement the structural Tool interface without inheriting
      from the :class:`Tool` ABC) simply skips the hook. The ABC's
      default no-op implementations make this a non-issue for proper
      subclasses; structural implementations just lack the attribute.
    * A hook that raises is logged at WARNING and swallowed — a
      misbehaving hook must never escalate into a failed tool call.
    * A hook that isn't callable (``on_enter = None``) is treated the
      same as missing.

    Previously this helper received the already-constructed coroutine,
    which forced the caller to do ``tool.on_enter(...)`` at the call
    site — blowing up on duck-typed tools before the try/except could
    catch it. Taking the tool + hook name here lets us look up the
    bound method with ``getattr`` first.
    """
    hook = getattr(tool, hook_name, None)
    if hook is None or not callable(hook):
        return
    try:
        result = hook(*hook_args)
        if _is_awaitable(result):
            await result
    except Exception:
        logger.warning(
            "tool %s lifecycle hook %s raised; ignored",
            tool_name,
            hook_name,
            exc_info=True,
        )


def _is_awaitable(obj: Any) -> bool:
    """True if ``obj`` is an awaitable we should ``await`` on.

    Hosts may declare sync lifecycle hooks (return ``None`` directly)
    or async hooks (return a coroutine). Both shapes are valid; only
    async return values need awaiting.
    """
    import inspect as _inspect

    return _inspect.isawaitable(obj)


class RegistryRouter(ToolRouter):
    """Routes tool calls via ToolRegistry lookup.

    Every failure mode (unknown tool, invalid input, tool-signaled
    failure, unexpected crash) is converted into a structured
    ``ToolError`` embedded in the ``ToolResult``. No free-form failure
    strings are emitted.

    Cycle 20260424 (Phase 2 Week 4 Checkpoint 3): fires the tool's
    ``on_enter`` / ``on_exit`` / ``on_error`` lifecycle hooks around
    ``execute``. Hooks see the post-validation input and run **after**
    the input schema passes — invalid inputs short-circuit before any
    hook fires so hooks can assume a well-formed payload.
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

        return await self._dispatch_with_lifecycle(tool, tool_input, context)

    async def _dispatch_with_lifecycle(
        self, tool: Tool, tool_input: Dict[str, Any], context: ToolContext
    ) -> ToolResult:
        """Execute ``tool`` with lifecycle hooks wrapped around it.

        Ordering:
            1. ``on_enter(input, ctx)`` — fired if present.
            2. ``tool.execute(...)`` — the actual body.
            3. On normal return → ``on_exit(result, ctx)`` if present.
            4. On ``ToolFailure`` or any other ``Exception`` →
               ``on_error(error, ctx)`` if present, then the error is
               mapped to a structured ``ToolError`` as before.

        Lifecycle hooks are optional: a tool that doesn't expose
        ``on_enter`` / ``on_exit`` / ``on_error`` attributes (e.g. a
        duck-typed host adapter that wasn't built against the ``Tool``
        ABC) simply skips that stage. All hook *failures* are logged
        and swallowed; the tool call's own success/failure drives the
        returned ``ToolResult``.
        """
        await _fire_hook("on_enter", tool.name, tool, tool_input, context)

        try:
            result = await tool.execute(tool_input, context)
        except ToolFailure as failure:
            logger.info(
                "tool %s raised ToolFailure (%s): %s",
                tool.name,
                failure.error.code.value,
                failure.error.message,
            )
            await _fire_hook("on_error", tool.name, tool, failure, context)
            return make_error_result(failure.error)
        except Exception as exc:
            logger.exception("tool %s crashed unexpectedly", tool.name)
            await _fire_hook("on_error", tool.name, tool, exc, context)
            return make_error_result(ToolError.tool_crashed(tool.name, exc))

        await _fire_hook("on_exit", tool.name, tool, result, context)
        return result
