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
    PermissionPosture,
    coerce_posture,
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

        Before any of the above, the permission matrix runs (when rules
        are bound or the posture is DENY). DENY → structured
        ``access_denied`` + ``PERMISSION_DENIED`` hook event; ASK →
        routed through ``_resolve_ask`` (PERMISSION_REQUEST hook, then
        the bound HITL requester, then the safe deny fallback).
        """
        runner = getattr(context, "hook_runner", None)

        # Phase 7 (S7.4): permission matrix consult. Fired before any
        # hooks so a DENY/ASK rule short-circuits the entire pipeline
        # — including the audit-side POST_TOOL_USE hook — and never
        # spawns a subprocess for a call we already know is blocked.
        #
        # 2.2.0 (audit §1-5): the matrix also runs with ZERO rules
        # bound when the configured posture is DENY — that combination
        # means "deny everything except an explicit allowlist", and
        # skipping evaluation entirely (the historical fast path) would
        # quietly turn the deny posture into a decoy setting.
        permission_rules = getattr(context, "permission_rules", None) or []
        posture = coerce_posture(getattr(context, "permission_default_posture", None))
        if permission_rules or posture is PermissionPosture.DENY:
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
                default_posture=posture,
            )
            if decision.behavior is PermissionBehavior.DENY:
                reason = decision.reason or "denied by permission matrix"
                await _fire_permission_denied(runner, tool.name, tool_input, context, reason)
                return make_error_result(ToolError.access_denied(tool.name, reason))
            if decision.behavior is PermissionBehavior.ASK:
                # 2.2.0 (audit §1-5): ASK finally routes somewhere. A
                # bound HITL requester (Stage 15 contract) gets the
                # request; a PERMISSION_REQUEST hook may also answer.
                # Without either, ASK resolves to DENY — see
                # ``_resolve_ask`` for the full ladder + rationale.
                denial = await _resolve_ask(
                    runner=runner,
                    tool=tool,
                    tool_input=tool_input,
                    context=context,
                    reason=decision.reason or "permission matrix returned ASK",
                )
                if denial is not None:
                    return denial
                # Approved (by hook or human) → continue dispatch.
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


async def _fire_permission_denied(
    runner: Any,
    tool_name: str,
    tool_input: Dict[str, Any],
    context: ToolContext,
    reason: str,
) -> None:
    """Fire ``PERMISSION_DENIED`` if a runner is bound (2.2.0).

    Observational only — a deny is final by the time this fires. The
    hook outcome is deliberately ignored: letting an external script
    un-deny a matrix verdict would invert the trust direction (hooks
    may *tighten* via PRE_TOOL_USE, never loosen a standing deny).
    """
    if runner is None:
        return
    payload = _build_hook_payload(
        HookEvent.PERMISSION_DENIED,
        tool_name,
        tool_input,
        context,
        extra_details={"reason": reason},
    )
    try:
        await runner.fire(HookEvent.PERMISSION_DENIED, payload)
    except Exception:  # pragma: no cover - defensive
        logger.warning("permission_denied hook fire raised; ignored", exc_info=True)


def _state_audit(context: ToolContext, event_type: str, payload: Dict[str, Any]) -> None:
    """Best-effort ``state.add_event`` through the context's state view.

    The router only holds a ``ToolContext``; the live ``PipelineState``
    rides on ``state_view`` when Stage 10 built the context (direct
    router invocations in tests may have none). Audit must never be
    able to fail a dispatch, hence the blanket guard.
    """
    state = getattr(context, "state_view", None)
    add_event = getattr(state, "add_event", None)
    if not callable(add_event):
        return
    try:
        add_event(event_type, payload)
    except Exception:  # pragma: no cover - defensive
        logger.warning("state audit event %s failed; ignored", event_type, exc_info=True)


async def _resolve_ask(
    *,
    runner: Any,
    tool: Tool,
    tool_input: Dict[str, Any],
    context: ToolContext,
    reason: str,
) -> Optional[ToolResult]:
    """Resolve a permission-matrix ASK into proceed-or-deny (2.2.0, audit §1-5).

    Until 2.2.0 the matrix could *return* ASK but nothing ever
    *produced* an approval request — ASK was a hard deny wearing a
    different reason string. This is the missing plumbing between the
    permission subsystem and the Stage 15 HITL contract. Ladder:

    1. ``PERMISSION_REQUEST`` hook (when a runner is bound). A hook
       outcome with ``decision='approve'`` answers the ASK without a
       human (hosts running machine policy engines — GAPT's case);
       a blocked outcome denies. Anything else falls through.
    2. The bound HITL requester — ``context.hitl_requester``, any
       object with an async ``request(request, state)`` method (the
       Stage 15 :class:`Requester` contract; ``CallbackRequester`` and
       ``PipelineResumeRequester`` slot straight in). The request /
       decision are mirrored into the Stage 15 shared keys
       (``hitl_history`` etc.) so the audit trail is identical no
       matter which stage collected the verdict. APPROVE proceeds;
       REJECT / CANCEL / no-decision denies. Requester exceptions deny
       (mirror of Stage 15's "never block the loop on requester bugs").
    3. Neither answered → **DENY.** Deliberately NOT the configured
       ``default_posture``: an ASK rule is the operator explicitly
       demanding judgement for this call, so auto-approving it because
       the *ambient* posture happens to be ``allow`` would invert the
       rule's intent. Hosts that want prompt-free sessions already
       have a config-level escape hatch — ``PermissionMode.DONT_ASK``
       / ``ACCEPT_EDITS`` promote ASK to ALLOW *before* it ever
       reaches this ladder.

    Returns ``None`` when dispatch may proceed, or the structured
    ``access_denied`` ToolResult to return verbatim.

    Note: there is no timeout ladder here — bounding the requester's
    wall-clock is Stage 15's ``TimeoutPolicy`` concern. A requester
    that wants timeouts wraps itself (or hosts use Stage 15 proper).
    """
    # Stage 15 types are imported lazily — same pattern as
    # ``Pipeline.resume`` — so this module's import graph doesn't grow
    # a hard cross-stage edge for pipelines that never use HITL.
    from geny_executor.stages.s15_hitl.interface import (
        HITL_HISTORY_KEY,
        HITL_LAST_DECISION_KEY,
    )
    from geny_executor.stages.s15_hitl.types import (
        HITLDecision,
        HITLEntry,
        HITLRequest,
        coerce_decision,
    )

    request = HITLRequest(
        reason=reason,
        severity="warn",
        tool_call_id=getattr(context, "parent_tool_use_id", None) or "",
        payload={
            "source": "permission_matrix",
            "tool_name": tool.name,
            "tool_input": dict(tool_input),
        },
    )

    async def _deny(deny_reason: str) -> ToolResult:
        await _fire_permission_denied(runner, tool.name, tool_input, context, deny_reason)
        return make_error_result(ToolError.access_denied(tool.name, deny_reason))

    def _record(decision: HITLDecision, *, via: str) -> None:
        """Mirror the Stage 15 audit contract (history + last-decision)."""
        _state_audit(
            context,
            "hitl.decision",
            {"token": request.token, "decision": decision.value, "via": via},
        )
        state = getattr(context, "state_view", None)
        shared = getattr(state, "shared", None)
        if not isinstance(shared, dict):
            return
        try:
            shared[HITL_LAST_DECISION_KEY] = decision.value
            shared.setdefault(HITL_HISTORY_KEY, []).append(
                HITLEntry(request=request, decision=decision, note=f"via {via}").to_dict()
            )
        except Exception:  # pragma: no cover - defensive
            logger.warning("hitl history audit write failed; ignored", exc_info=True)

    _state_audit(context, "hitl.request", request.to_dict())

    # 1. PERMISSION_REQUEST hook — machine policy gets first answer.
    if runner is not None:
        payload = _build_hook_payload(
            HookEvent.PERMISSION_REQUEST,
            tool.name,
            tool_input,
            context,
            extra_details={"reason": reason, "token": request.token},
        )
        try:
            outcome = await runner.fire(HookEvent.PERMISSION_REQUEST, payload)
        except Exception:  # pragma: no cover - defensive
            logger.warning("permission_request hook fire raised; ignored", exc_info=True)
            outcome = None
        if outcome is not None:
            if outcome.blocked:
                _record(HITLDecision.REJECT, via="permission_request_hook")
                return await _deny(
                    outcome.stop_reason or f"{reason} (blocked by permission_request hook)"
                )
            if outcome.decision == "approve":
                _record(HITLDecision.APPROVE, via="permission_request_hook")
                return None

    # 2. Bound HITL requester (Stage 15 Requester contract).
    requester = getattr(context, "hitl_requester", None)
    request_fn = getattr(requester, "request", None)
    if callable(request_fn):
        state = getattr(context, "state_view", None)
        try:
            raw = await request_fn(request, state)
        except Exception as exc:  # noqa: BLE001 — requester bugs must not crash dispatch
            logger.warning(
                "HITL requester %s raised %s — denying tool %s",
                getattr(requester, "name", type(requester).__name__),
                exc,
                tool.name,
            )
            _record(HITLDecision.CANCEL, via="requester_error")
            return await _deny(f"{reason} (HITL requester failed: {exc})")
        decision = coerce_decision(raw)
        if decision is HITLDecision.APPROVE:
            _record(decision, via="hitl_requester")
            return None
        if decision is None:
            # No verdict — Stage 15 would consult its TimeoutPolicy
            # here; at the dispatch site the only safe reading of "the
            # human didn't answer" is no.
            _record(HITLDecision.REJECT, via="hitl_no_decision")
            return await _deny(f"{reason} (HITL requester returned no decision)")
        _record(decision, via="hitl_requester")
        return await _deny(f"{reason} (HITL {decision.value})")

    # 3. No handler at all — deny, loudly enough to find in logs.
    logger.info(
        "tool %s permission ASK with no HITL requester bound — denying "
        "(bind one via the Tool stage context's 'hitl_requester', or use "
        "PermissionMode.DONT_ASK to promote ASKs)",
        tool.name,
    )
    _record(HITLDecision.REJECT, via="no_requester")
    return await _deny(f"{reason} (no HITL requester bound — ASK defaults to deny)")
