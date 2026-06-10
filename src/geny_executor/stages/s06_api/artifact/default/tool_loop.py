"""Tool-loop strategies — where the agentic loop runs (2.3.0).

Why this slot exists: the ``claude_code_cli`` backend runs its agentic
loop INSIDE the subprocess — Stage 6 streams the events, the terminal
:class:`APIResponse` carries only final text (tool_use blocks consumed;
see ``StreamJsonAccumulator.finalize`` in
``llm_client/translators/_cli.py`` for the contract rationale), Stage 9
finds no pending tool calls and Stage 10 naturally no-ops. SDK
providers, by contrast, paid a full pipeline iteration per tool
round-trip: every Stage 2-5/7/14 re-run, per call. This slot makes the
CLI execution shape a manifest-selectable choice for EVERY backend:

- :class:`PipelineToolLoop` (default ``"pipeline"``) — exactly the
  pre-2.3.0 behaviour: one client call, tool_use blocks returned
  verbatim, Stage 9 → Stage 10 → Stage 16 own the loop. Zero behaviour
  change for every existing manifest.
- :class:`InternalAgenticLoop` (``"internal"``) — resolves tool calls
  inside Stage 6 (call → dispatch → call …) and returns only the final
  text/thinking response, mirroring the CLI accumulator's finalize
  contract. Tool dispatch goes through the SAME
  :class:`~geny_executor.stages.s10_tool.dispatcher.ToolDispatcher`
  channel Stage 10 uses (same ToolRegistry instance, same
  ``evaluate_permission`` matrix path, same PERMISSION_* / PRE/POST
  hook firing) — there is exactly one permission decision
  implementation in the engine, not two.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set

from geny_executor.core.schema import ConfigField, ConfigSchema
from geny_executor.core.state import PipelineState, TokenUsage
from geny_executor.stages.s06_api.interface import ToolLoopCall, ToolLoopStrategy
from geny_executor.stages.s06_api.types import APIResponse

logger = logging.getLogger(__name__)


def assistant_content_blocks(response: APIResponse) -> List[Dict[str, Any]]:
    """Render a response's content as assistant-message blocks.

    Shared by ``APIStage._build_assistant_content`` (the single-call
    path) and :class:`InternalAgenticLoop` (which must record each
    intermediate assistant turn with its tool_use blocks intact, so the
    conversation history is complete and rehydratable). One renderer —
    the recorded history cannot drift between the two paths.
    """
    blocks: List[Dict[str, Any]] = []
    for block in response.content:
        if block.raw:
            blocks.append(block.raw)
        elif block.type == "text":
            blocks.append({"type": "text", "text": block.text or ""})
        elif block.type == "tool_use":
            blocks.append(
                {
                    "type": "tool_use",
                    "id": block.tool_use_id,
                    "name": block.tool_name,
                    "input": block.tool_input,
                }
            )
    return blocks


class PipelineToolLoop(ToolLoopStrategy):
    """Default: the loop belongs to the pipeline (Stage 9/10/16).

    A deliberate pass-through — one call, response returned verbatim,
    tool_use blocks included. Kept free of knobs and behaviour so the
    default path stays byte-identical to pre-2.3.0: choosing where the
    loop runs is opt-in, never a silent migration.
    """

    @property
    def name(self) -> str:
        return "pipeline"

    @property
    def description(self) -> str:
        return "One call per pipeline iteration; Stage 9/10/16 own the tool loop"

    async def run(
        self,
        *,
        call: ToolLoopCall,
        client: Any,
        state: PipelineState,
    ) -> APIResponse:
        return await call(None)


def _require_int(strategy: str, key: str, value: Any, *, minimum: int = 1) -> int:
    """configure() validation: integer >= minimum, bools rejected explicitly."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{strategy}: {key!r} must be an integer >= {minimum}, got {value!r}")
    if value < minimum:
        raise ValueError(f"{strategy}: {key!r} must be >= {minimum}, got {value!r}")
    return value


def _require_bool(strategy: str, key: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{strategy}: {key!r} must be a boolean, got {value!r}")
    return value


class InternalAgenticLoop(ToolLoopStrategy):
    """Resolve tool calls inside Stage 6 — the CLI execution shape for SDKs.

    After the first client call, while the response carries tool_use
    blocks: dispatch them through the state-carried
    :class:`~geny_executor.stages.s10_tool.dispatcher.ToolDispatcher`
    (Stage 10's registry, permission matrix, ASK→HITL ladder and hook
    events — one decision path, shared), append the exchange to a
    LOOP-LOCAL message list, and call the client again with
    ``state.messages + exchange``. ``state.messages`` is mutated only
    once, at loop end: every intermediate assistant tool_use message
    and user tool_result message is recorded in order (completeness
    over thrift — a rehydrated session must replay the same
    conversation the model actually saw), and the FINAL assistant
    content still lands via the stage's normal append — the same
    contract as the single-call path.

    The returned :class:`APIResponse` carries only final text/thinking
    blocks: the loop only exits successfully when the model stopped
    requesting tools, mirroring ``StreamJsonAccumulator.finalize``
    (which drops already-executed tool_use blocks from the terminal CLI
    response so Stage 10 "sees no tool_use blocks and naturally
    no-ops"). Its usage is the SUM of every inner call's usage, so
    Stage 7 prices the whole turn; each inner call additionally emitted
    its own ``api.request`` / ``api.response`` pair for per-call
    attribution.

    Failure containment — the model reacts, the turn never dies:

    - permission DENY → the dispatcher returns a structured
      ``access_denied`` tool_result (``is_error=True``) after firing
      ``PERMISSION_DENIED``; the loop continues.
    - ASK → ``PERMISSION_REQUEST`` hook, then the bound HITL requester,
      then safe-deny — Stage 10's exact ladder, because it IS Stage
      10's ladder.
    - tool exceptions → structured error tool_result, never a raise.

    Caps (graceful degradation to the pipeline path): when
    ``max_inner_turns`` is reached or the per-turn cost budget
    (``state.total_cost_usd`` + this loop's inner cost vs
    ``state.cost_budget_usd`` — the same fields Stage 16's budget
    controllers read) is exceeded, the strategy emits
    ``api.internal_loop_capped`` {turns, reason} and returns the last
    response AS-IS — unresolved tool_use blocks then flow to Stage 9/10
    like any pipeline-mode response, so a capped loop degrades into the
    default execution shape instead of dropping work.

    Capability guard (one-time warning, then pipeline behaviour):

    - subprocess backends (``capabilities.is_subprocess`` — the CLI
      already loops internally; looping again would re-dispatch tools
      the subprocess already executed);
    - clients without ``supports_tools`` (nothing to loop);
    - no ``state.tool_dispatcher`` (the pipeline has no Tool stage to
      share a dispatch path with).
    """

    def __init__(self, max_inner_turns: int = 10, parallel_tools: bool = False):
        self._max_inner_turns = max_inner_turns
        self._parallel_tools = parallel_tools
        self._warned_reasons: Set[str] = set()

    @property
    def name(self) -> str:
        return "internal"

    @property
    def description(self) -> str:
        return (
            f"Resolve tool calls inside Stage 6 (max {self._max_inner_turns} "
            "inner turns); only the final response reaches Stage 9"
        )

    @classmethod
    def config_schema(cls) -> ConfigSchema:
        return ConfigSchema(
            name="internal",
            fields=[
                ConfigField(
                    name="max_inner_turns",
                    type="integer",
                    label="Max inner turns",
                    description=(
                        "Tool-resolution rounds the loop may run inside one "
                        "Stage 6 execution before emitting "
                        "api.internal_loop_capped and handing leftover tool "
                        "calls back to the pipeline path."
                    ),
                    default=10,
                    min_value=1,
                ),
                ConfigField(
                    name="parallel_tools",
                    type="boolean",
                    label="Parallel tools",
                    description=(
                        "Dispatch a round's tool calls concurrently "
                        "(asyncio.gather) instead of sequentially."
                    ),
                    default=False,
                    ui_widget="toggle",
                ),
            ],
        )

    def configure(self, config: Dict[str, Any]) -> None:
        # Validate everything before applying anything — a rejected
        # configure must leave the previous (working) config live.
        turns = self._max_inner_turns
        parallel = self._parallel_tools
        if "max_inner_turns" in config:
            turns = _require_int("internal", "max_inner_turns", config["max_inner_turns"])
        if "parallel_tools" in config:
            parallel = _require_bool("internal", "parallel_tools", config["parallel_tools"])
        self._max_inner_turns = turns
        self._parallel_tools = parallel

    def get_config(self) -> Dict[str, Any]:
        return {
            "max_inner_turns": self._max_inner_turns,
            "parallel_tools": self._parallel_tools,
        }

    # ── Guards ──────────────────────────────────────────────

    def _degrade_reason(self, client: Any, state: PipelineState) -> Optional[str]:
        """Return why the internal loop cannot run here, or ``None``.

        Each reason is documented in the class docstring; a non-None
        return downgrades this execution to pipeline behaviour.
        """
        caps = getattr(client, "capabilities", None)
        if bool(getattr(caps, "is_subprocess", False)):
            return (
                "the client is a subprocess backend that runs its own "
                "agentic loop (e.g. claude_code_cli) — looping again in "
                "Stage 6 would re-dispatch tools the subprocess already "
                "executed"
            )
        if not bool(getattr(caps, "supports_tools", False)):
            return "the client's capabilities lack supports_tools — nothing to loop"
        if getattr(state, "tool_dispatcher", None) is None:
            return (
                "state.tool_dispatcher is unset — no Tool stage is "
                "registered to share a dispatch/permission path with"
            )
        return None

    def _warn_once(self, reason: str, client: Any) -> None:
        if reason in self._warned_reasons:
            return
        self._warned_reasons.add(reason)
        logger.warning(
            "tool_loop='internal' on provider %r is degrading to pipeline behaviour: %s",
            getattr(client, "provider", "") or type(client).__name__,
            reason,
        )

    def _budget_exceeded(self, state: PipelineState, inner_cost_usd: float) -> bool:
        """Same fields Stage 16's budget controllers read.

        ``state.total_cost_usd`` only grows when Stage 7 prices a
        returned response, so mid-loop the inner calls' cost (when the
        backend reports one) is added on top — otherwise an internal
        loop could spend arbitrarily far past the per-turn budget
        before Stage 16 ever saw a number.
        """
        if state.cost_budget_usd is None:
            return False
        return (state.total_cost_usd + inner_cost_usd) >= state.cost_budget_usd

    # ── Dispatch ────────────────────────────────────────────

    @staticmethod
    def _as_tool_calls(response: APIResponse) -> List[Dict[str, Any]]:
        """tool_use blocks → the dict shape Stage 9 hands Stage 10."""
        return [
            {
                "tool_use_id": block.tool_use_id or "",
                "tool_name": block.tool_name or "",
                "tool_input": block.tool_input or {},
            }
            for block in response.tool_calls
        ]

    @staticmethod
    async def _dispatch_one(
        dispatcher: Any, tc: Dict[str, Any], state: PipelineState
    ) -> Dict[str, Any]:
        """One dispatch, exception-proof.

        The dispatcher already structures every tool-level failure into
        an ``is_error`` result; this guard covers dispatcher-machinery
        crashes so a single bad call can never kill the turn.
        """
        try:
            return await dispatcher.dispatch(tc, state)
        except Exception as exc:  # noqa: BLE001 — containment is the contract
            logger.exception("internal loop dispatch for %s crashed", tc.get("tool_name"))
            return {
                "type": "tool_result",
                "tool_use_id": tc.get("tool_use_id", ""),
                "content": f"ERROR tool_dispatch_failed: {exc}",
                "is_error": True,
            }

    def _emit_tool_use(self, state: PipelineState, tc: Dict[str, Any]) -> None:
        state.add_event(
            "api.tool_use",
            {
                "id": tc.get("tool_use_id", ""),
                "name": tc.get("tool_name", ""),
                "input": tc.get("tool_input", {}),
                "source": "internal",
            },
        )

    def _emit_tool_result(self, state: PipelineState, result: Dict[str, Any]) -> None:
        state.add_event(
            "api.tool_result",
            {
                "tool_use_id": result.get("tool_use_id", ""),
                "content": result.get("content"),
                "is_error": bool(result.get("is_error", False)),
                "source": "internal",
            },
        )

    async def _dispatch_round(
        self,
        dispatcher: Any,
        tool_calls: List[Dict[str, Any]],
        state: PipelineState,
    ) -> List[Dict[str, Any]]:
        """Dispatch one round of tool calls, emitting the canonical events.

        Sequential mode interleaves ``api.tool_use`` → dispatch →
        ``api.tool_result`` per call; parallel mode announces every
        ``api.tool_use`` up front, gathers, then emits results in input
        order (results stay input-ordered either way, matching the
        Stage 10 executors' determinism contract).
        """
        results: List[Dict[str, Any]] = []
        if self._parallel_tools and len(tool_calls) > 1:
            for tc in tool_calls:
                self._emit_tool_use(state, tc)
            results = list(
                await asyncio.gather(
                    *(self._dispatch_one(dispatcher, tc, state) for tc in tool_calls)
                )
            )
            for result in results:
                self._emit_tool_result(state, result)
        else:
            for tc in tool_calls:
                self._emit_tool_use(state, tc)
                result = await self._dispatch_one(dispatcher, tc, state)
                self._emit_tool_result(state, result)
                results.append(result)
        return results

    # ── The loop ────────────────────────────────────────────

    async def run(
        self,
        *,
        call: ToolLoopCall,
        client: Any,
        state: PipelineState,
    ) -> APIResponse:
        reason = self._degrade_reason(client, state)
        if reason is not None:
            self._warn_once(reason, client)
            return await call(None)

        dispatcher = state.tool_dispatcher
        response = await call(None)

        exchange: List[Dict[str, Any]] = []  # loop-local; lands on state at the end
        consumed_usage: Optional[TokenUsage] = None  # inner calls already resolved
        inner_cost_usd = 0.0
        turns = 0

        while response.tool_calls:
            if turns >= self._max_inner_turns:
                state.add_event(
                    "api.internal_loop_capped",
                    {"turns": turns, "reason": "max_inner_turns"},
                )
                break
            if self._budget_exceeded(state, inner_cost_usd):
                state.add_event(
                    "api.internal_loop_capped",
                    {"turns": turns, "reason": "cost_budget"},
                )
                break

            tool_calls = self._as_tool_calls(response)
            results = await self._dispatch_round(dispatcher, tool_calls, state)

            exchange.append({"role": "assistant", "content": assistant_content_blocks(response)})
            exchange.append({"role": "user", "content": results})

            # This response's usage is now "consumed" — it will never be
            # the returned response, so fold it into the running sum that
            # the final response carries for Stage 7.
            consumed_usage = (
                response.usage if consumed_usage is None else consumed_usage + response.usage
            )
            inner_cost_usd += response.usage.cost_usd or 0.0
            turns += 1

            response = await call(list(exchange))

        # Record the intermediate exchange in order; the stage appends
        # the FINAL assistant content right after run() returns, so the
        # history reads exactly like a pipeline-mode multi-iteration
        # turn would.
        for message in exchange:
            state.add_message(message["role"], message["content"])

        if consumed_usage is not None:
            response.usage = consumed_usage + response.usage

        return response
