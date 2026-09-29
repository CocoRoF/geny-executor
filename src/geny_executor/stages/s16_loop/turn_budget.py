"""End a turn gracefully when it has read too much, or run out of steps.

**Input budget.** Measured on XGEN's runtime (same stack, 28 days, 7,159
turns): per-turn input p50 10k, p99 248k — yet the 0.15% of turns over 1M
tokens were 24% of all input, and the largest (37M tokens, 442 tool calls)
was 16% of a month and ended without an answer. The line is drawn high
enough not to cut normal long work. Geny's defaults are twice XGEN's
(2M / 5M): every Geny call carries a ~38k-token system prompt and tool
list, roughly four times XGEN's, so the same work reads more.

* Past ``soft`` cumulative input (cache reads and writes included — what the
  model actually processed) the last tool result gets one "wrap up" note.
* Past ``hard`` it gets "no more tools, report what is done and how to
  continue", and the turn ends with the next response.

**Step limit.** A turn that reached ``max_iterations`` stopped right after a
tool round: the model never saw the results and never answered, and the
turn's text was whatever it had said before its last call. The step before
the last now tells it to stop calling tools and report.

Both end the turn as a normal completion, with ``completion_signal`` saying
why, so the user can pick it up with a new message.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from geny_executor.core.state import PipelineState
from geny_executor.stages.s16_loop.repeat_stop import _append_note

__all__ = [
    "BUDGET_KEY",
    "DEFAULT_HARD_TOKENS",
    "DEFAULT_SOFT_TOKENS",
    "STEP_LIMIT_KEY",
    "apply_step_limit",
    "apply_turn_budget",
    "budget_stopped",
    "turn_input_tokens",
]

BUDGET_KEY = "loop.turn_budget"
STEP_LIMIT_KEY = "loop.step_limit"

DEFAULT_SOFT_TOKENS = 2_000_000
DEFAULT_HARD_TOKENS = 5_000_000

_SOFT_NOTE = (
    "[Turn budget: {used:,} of {hard:,} input tokens used. Wrap up now — finish the single "
    "most valuable remaining step, then report what is done and what is left. Do not start "
    "new exploration.]"
)
_FINAL_NOTE = (
    "[Turn budget exhausted: {used:,} input tokens (limit {hard:,}). Do not call any more tools. "
    "In this response, report what has been done, what remains, and exactly how to continue "
    "(files, commands, next step). The turn ends after this response.]"
)
_STEP_NOTE = (
    "[Step limit: the next response is the last step of this turn ({limit} steps). Do not call "
    "any more tools. Report what has been done, what remains, and exactly how to continue — "
    'the user can say "continue" to pick it up.]'
)


def turn_input_tokens(state: PipelineState) -> int:
    """Prompt tokens the model processed this turn, cache reads and writes included."""
    total = 0
    for u in state.turn_token_usage:
        total += (
            int(getattr(u, "input_tokens", 0) or 0)
            + int(getattr(u, "cache_creation_input_tokens", 0) or 0)
            + int(getattr(u, "cache_read_input_tokens", 0) or 0)
        )
    return total


def _record(state: PipelineState, key: str) -> Dict[str, Any]:
    rec = state.shared.get(key)
    if not isinstance(rec, dict):
        rec = {}
        state.shared[key] = rec
    return rec


def apply_turn_budget(state: PipelineState, decision: str, soft: int, hard: int) -> str:
    """Stage 16 calls this after its controller; returns the (possibly changed) decision."""
    if hard <= 0:
        return decision
    used = turn_input_tokens(state)
    rec = _record(state, BUDGET_KEY)
    calls = len(state.turn_token_usage)
    if rec.get("stopped"):
        return "complete"

    final_calls = rec.get("final_calls")
    if isinstance(final_calls, int) and calls > final_calls:
        # The response to the final note has arrived: end, whatever it asked for.
        rec["stopped"] = True
        rec["used"] = used
        state.completion_signal = "TURN_INPUT_BUDGET"
        state.completion_detail = f"turn input budget: {used:,} tokens (limit {hard:,})"
        state.add_event(
            "loop.turn_budget",
            {"phase": "stop", "used": used, "soft": soft, "hard": hard, "calls": calls},
        )
        return "complete"

    if decision != "continue":
        return decision

    if used >= hard and final_calls is None:
        if _append_note(state, _FINAL_NOTE.format(used=used, hard=hard)):
            rec["final_calls"] = calls
            rec["used"] = used
            state.add_event(
                "loop.turn_budget",
                {"phase": "final", "used": used, "soft": soft, "hard": hard, "calls": calls},
            )
        return decision

    if 0 < soft < hard and used >= soft and rec.get("soft_calls") is None:
        if _append_note(state, _SOFT_NOTE.format(used=used, hard=hard)):
            rec["soft_calls"] = calls
            state.add_event(
                "loop.turn_budget",
                {"phase": "soft", "used": used, "soft": soft, "hard": hard, "calls": calls},
            )
    return decision


def apply_step_limit(state: PipelineState, decision: str) -> str:
    """Before the last step, tell the model it is the last step."""
    limit = int(getattr(state, "max_iterations", 0) or 0)
    if decision != "continue" or limit <= 1:
        return decision
    rec = _record(state, STEP_LIMIT_KEY)
    if rec.get("noted"):
        return decision
    # The pipeline increments ``iteration`` after this and stops once it
    # reaches ``limit``: the next iteration is the last when it is limit-1.
    if state.iteration + 1 == limit - 1 and _append_note(state, _STEP_NOTE.format(limit=limit)):
        rec["noted"] = True
        rec["iteration"] = state.iteration
        state.add_event("loop.step_limit", {"limit": limit, "iteration": state.iteration})
    return decision


def budget_stopped(state: PipelineState) -> Optional[Dict[str, Any]]:
    """For hosts: the record, if this turn ended on its input budget."""
    rec = state.shared.get(BUDGET_KEY)
    if isinstance(rec, dict) and rec.get("stopped"):
        return rec
    return None
