"""End the turn once the harness keeps refusing the model's calls.

The repeat guard (``s10_tool.repeat_guard``) answers a hopeless call
without running it. That is not enough on its own: in the sibling runtime a
model ignored the refusal and made the same call dozens more times, each
refusal another full round-trip, until the turn's input budget ran out —
LLM 127 calls against 32 tool executions, three separate tasks, the same
shape every time. No real user turn had that shape.

So once ``stop_after`` calls have been refused in a turn, the last tool
result carries a note — report what was done, what is left, what blocks
you; call no more tools — and the model's next response ends the turn,
whatever it asks for. By then the model has already ignored the warning and
``stop_after`` refusals; a threshold only moves where the loop starts.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from geny_executor.core.state import PipelineState
from geny_executor.stages.s10_tool.repeat_guard import REFUSED_KEY

__all__ = ["DEFAULT_STOP_AFTER", "REPEAT_STOP_KEY", "apply_repeat_stop", "repeat_stopped"]

REPEAT_STOP_KEY = "loop.repeat_stop"
DEFAULT_STOP_AFTER = 3

_FINAL_NOTE = (
    "[Stopped: {refused} of your tool calls were refused this turn because they repeated a "
    "call that already returned the same result, or kept failing the same way. Repeating them "
    "will not produce anything new. Do not call any more tools. In this response, report what "
    "has been done, what remains, what is blocking you, and exactly how to continue. The turn "
    "ends after this response.]"
)


def _append_note(state: PipelineState, note: str) -> bool:
    """Put *note* where the next request will carry it: on the last tool
    result, or as a user message after an assistant one."""
    if not state.messages:
        return False
    last = state.messages[-1]
    if not isinstance(last, dict):
        return False
    content = last.get("content")
    if last.get("role") == "user" and isinstance(content, list):
        for block in reversed(content):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                inner = block.get("content")
                if isinstance(inner, list):
                    inner.append({"type": "text", "text": note})
                else:
                    block["content"] = f"{inner or ''}\n\n{note}"
                return True
        return False
    if last.get("role") == "assistant":
        state.add_message("user", note)
        return True
    return False


def apply_repeat_stop(state: PipelineState, decision: str, stop_after: int) -> str:
    """Stage 16 calls this after its controller decides. Returns the
    (possibly changed) decision."""
    if stop_after <= 0:
        return decision
    record = state.shared.get(REPEAT_STOP_KEY)
    if not isinstance(record, dict):
        record = {}
        state.shared[REPEAT_STOP_KEY] = record
    if record.get("stopped"):
        return "complete"
    refused = int(state.shared.get(REFUSED_KEY) or 0)
    final_iteration = record.get("final_iteration")

    # The response to the note has arrived: end, whatever it asked for.
    if isinstance(final_iteration, int) and state.iteration > final_iteration:
        record["stopped"] = True
        record["refused"] = refused
        state.completion_signal = "REPEAT_STOP"
        state.completion_detail = f"repeated calls refused {refused} times"
        state.add_event(
            "loop.repeat_stop",
            {"phase": "stop", "refused": refused, "iteration": state.iteration},
        )
        return "complete"

    if decision != "continue" or final_iteration is not None:
        return decision
    if refused >= stop_after and _append_note(state, _FINAL_NOTE.format(refused=refused)):
        record["final_iteration"] = state.iteration
        record["refused"] = refused
        state.add_event(
            "loop.repeat_stop",
            {"phase": "final", "refused": refused, "iteration": state.iteration},
        )
    return decision


def repeat_stopped(state: PipelineState) -> Optional[Dict[str, Any]]:
    """For hosts: the record, if this turn ended on repeated refusals."""
    record = state.shared.get(REPEAT_STOP_KEY)
    if isinstance(record, dict) and record.get("stopped"):
        return record
    return None
