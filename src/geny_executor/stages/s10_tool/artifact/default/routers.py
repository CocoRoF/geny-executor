"""Default artifact routers for Stage 10: Tool."""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Any, Dict, Optional

import jsonschema

from geny_executor.hooks.events import HookEvent, HookEventPayload
from geny_executor.permission.matrix import evaluate_permission
from geny_executor.permission.types import (
    PermissionBehavior,
    PermissionMode,
)
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


def _coerce_permission_mode(raw: Any) -> PermissionMode:
    """Best-effort coercion of ``ToolContext.permission_mode`` (str) to enum.

    The context field stays ``str`` for ergonomics; the matrix wants
    the enum. Unknown values fall back to ``DEFAULT`` rather than
    raising — mode is a soft policy hint, not a hard contract.
    """
    if isinstance(raw, PermissionMode):
        return raw
    try:
        return PermissionMode(raw or "default")
    except (ValueError, TypeError):
        logger.debug("unknown permission_mode %r — falling back to DEFAULT", raw)
        return PermissionMode.DEFAULT


def _now_iso() -> str:
    """UTC timestamp in ISO-8601 form for hook payloads."""
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _build_hook_payload(
    event: HookEvent,
    tool_name: str,
    tool_input: Dict[str, Any],
    context: ToolContext,
    *,
    tool_output: Optional[str] = None,
    extra_details: Optional[Dict[str, Any]] = None,
) -> HookEventPayload:
    """Assemble a :class:`HookEventPayload` from the dispatch context.

    Kept as a helper so the router stays declarative and tests can
    construct identical payloads when needed.
    """
    return HookEventPayload(
        event=event,
        session_id=context.session_id or "",
        timestamp=_now_iso(),
        permission_mode=getattr(context, "permission_mode", "default") or "default",
        stage_order=getattr(context, "stage_order", 0) or 0,
        stage_name=getattr(context, "stage_name", "") or "",
        tool_name=tool_name,
        tool_input=dict(tool_input),
        tool_output=tool_output,
        details=dict(extra_details or {}),
    )


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
        """Execute ``tool`` with subprocess hooks + tool lifecycle hooks
        wrapped around it.

        Ordering:
            1. Fire ``PRE_TOOL_USE`` subprocess hook (Phase 5). If the
               combined outcome is blocked, return an ``ACCESS_DENIED``
               error result without invoking ``execute``. If the
               outcome carries ``modified_input``, the rest of the
               pipeline uses it as the effective input.
            2. ``on_enter(input, ctx)`` — Tool ABC lifecycle hook,
               fired if present.
            3. ``tool.execute(...)`` — the actual body.
            4. On normal return → ``on_exit(result, ctx)`` then
               ``POST_TOOL_USE`` (or ``POST_TOOL_FAILURE`` when the
               tool returned a soft-error result with ``is_error=True``).
            5. On ``ToolFailure`` or any other ``Exception`` →
               ``on_error(error, ctx)`` then ``POST_TOOL_FAILURE``,
               then the error is mapped to a structured ``ToolError``
               as before.

        Both layers of hooks are optional and fail-open. A tool without
        ``on_enter`` / ``on_exit`` / ``on_error`` simply skips those.
        Without a ``context.hook_runner`` bound, the subprocess hook
        layer is a complete no-op. Subprocess hook failures are
        already absorbed inside ``HookRunner``; nothing here can leak.
        """
        runner = getattr(context, "hook_runner", None)

        # Phase 7 (S7.4): permission matrix consult. Fired before any
        # hooks so a DENY/ASK rule short-circuits the entire pipeline
        # — including the audit-side POST_TOOL_USE hook — and never
        # spawns a subprocess for a call we already know is blocked.
        permission_rules = getattr(context, "permission_rules", None) or []
        if permission_rules:
            mode = _coerce_permission_mode(getattr(context, "permission_mode", None))
            try:
                # Capabilities for the in-flight input — destructive
                # tools auto-escalate under PLAN mode.
                caps = tool.capabilities(tool_input)
                destructive = bool(getattr(caps, "destructive", False))
            except Exception:
                destructive = False
            decision = await evaluate_permission(
                tool=tool,
                tool_input=tool_input,
                rules=list(permission_rules),
                mode=mode,
                capabilities_destructive=destructive,
            )
            if decision.behavior is PermissionBehavior.DENY:
                reason = decision.reason or "denied by permission matrix"
                return make_error_result(ToolError.access_denied(tool.name, reason))
            if decision.behavior is PermissionBehavior.ASK:
                # No HITL stage in the 16-stage layout yet (Phase 9
                # adds Stage 15 HITL). Until that lands, ASK without
                # an explicit handler treats as DENY with a clear
                # reason — safer than silently allowing.
                reason = decision.reason or "permission matrix returned ASK; no handler bound"
                logger.info(
                    "tool %s permission ASK with no HITL handler — denying for safety",
                    tool.name,
                )
                return make_error_result(ToolError.access_denied(tool.name, reason))
            # ALLOW path may carry a rewritten input (rare today, future
            # rules might normalise paths or strip secrets).
            if decision.updated_input is not None:
                tool_input = dict(decision.updated_input)
                try:
                    validate_input(tool.input_schema, tool_input)
                except jsonschema.ValidationError as exc:
                    path = ".".join(str(p) for p in exc.absolute_path) or "<root>"
                    return make_error_result(
                        ToolError.invalid_input(tool.name, exc.message, path=path)
                    )

        if runner is not None:
            pre_payload = _build_hook_payload(
                HookEvent.PRE_TOOL_USE, tool.name, tool_input, context
            )
            pre_outcome = await runner.fire(HookEvent.PRE_TOOL_USE, pre_payload)
            if pre_outcome.blocked:
                reason = pre_outcome.stop_reason or "blocked by pre_tool_use hook"
                return make_error_result(ToolError.access_denied(tool.name, reason))
            if pre_outcome.modified_input is not None:
                tool_input = dict(pre_outcome.modified_input)
                # Re-validate after a hook rewrite so a misbehaving
                # hook can't bypass the tool's input schema.
                try:
                    validate_input(tool.input_schema, tool_input)
                except jsonschema.ValidationError as exc:
                    path = ".".join(str(p) for p in exc.absolute_path) or "<root>"
                    return make_error_result(
                        ToolError.invalid_input(tool.name, exc.message, path=path)
                    )

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
            await _fire_post_tool_hook(
                runner,
                event=HookEvent.POST_TOOL_FAILURE,
                tool_name=tool.name,
                tool_input=tool_input,
                context=context,
                output_preview=f"ToolFailure: {failure.error.message}",
            )
            return make_error_result(failure.error)
        except Exception as exc:
            logger.exception("tool %s crashed unexpectedly", tool.name)
            await _fire_hook("on_error", tool.name, tool, exc, context)
            await _fire_post_tool_hook(
                runner,
                event=HookEvent.POST_TOOL_FAILURE,
                tool_name=tool.name,
                tool_input=tool_input,
                context=context,
                output_preview=f"crash: {exc}",
            )
            return make_error_result(ToolError.tool_crashed(tool.name, exc))

        await _fire_hook("on_exit", tool.name, tool, result, context)

        # Soft errors (tool returned ``is_error=True`` without raising)
        # also count as failures from the hook subsystem's POV — gives
        # post-failure auditors a unified observation point.
        post_event = HookEvent.POST_TOOL_FAILURE if result.is_error else HookEvent.POST_TOOL_USE
        await _fire_post_tool_hook(
            runner,
            event=post_event,
            tool_name=tool.name,
            tool_input=tool_input,
            context=context,
            output_preview=_preview_result(result),
        )
        return result


def _preview_result(result: ToolResult, *, max_chars: int = 500) -> str:
    """Compact preview of a tool result for hook payloads."""
    body = result.display_text if result.display_text is not None else result.content
    if isinstance(body, str):
        text = body
    else:
        text = str(body)
    if len(text) > max_chars:
        return text[:max_chars] + f"… ({len(text)} chars total)"
    return text


async def _fire_post_tool_hook(
    runner: Any,
    *,
    event: HookEvent,
    tool_name: str,
    tool_input: Dict[str, Any],
    context: ToolContext,
    output_preview: str,
) -> None:
    """Fire ``POST_TOOL_USE`` / ``POST_TOOL_FAILURE`` if a runner is bound.

    Post-tool hooks are observational only — their outcomes do not
    feed back into the pipeline (the tool result is already final).
    Logged failures inside ``HookRunner`` are sufficient.
    """
    if runner is None:
        return
    payload = _build_hook_payload(event, tool_name, tool_input, context, tool_output=output_preview)
    try:
        await runner.fire(event, payload)
    except Exception:  # pragma: no cover - defensive
        logger.warning(
            "post-tool hook fire raised; ignored",
            exc_info=True,
        )
