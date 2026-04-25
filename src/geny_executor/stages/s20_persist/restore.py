"""Checkpoint → :class:`PipelineState` restoration helpers (S9c.2).

The inverse of :func:`stage._build_payload`. Stage 20 writes a
JSON-shaped snapshot of the non-runtime portion of the state; this
module reads one back through any :class:`Persister` and reconstructs
a fresh :class:`PipelineState` ready to be plugged into a Pipeline
run.

The runtime fields (``llm_client``, ``session_runtime``, the event
listener) are *not* restored — those are bound by the host at run
time. ``token_usage`` is restored from the dict shape; ``messages``
is copied verbatim.

A future :func:`Pipeline.from_checkpoint` could compose this with a
host-supplied stage list. For now the surface stays narrow:
``restore_state_from_checkpoint`` returns the state, hosts plug it
into whatever Pipeline they already build.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from geny_executor.core.state import PipelineState, TokenUsage
from geny_executor.stages.s20_persist.interface import Persister
from geny_executor.stages.s20_persist.types import CheckpointRecord


class CheckpointNotFound(LookupError):
    """Raised when the persister has no record for the requested id."""


def state_from_payload(payload: Dict[str, Any]) -> PipelineState:
    """Rebuild a :class:`PipelineState` from a checkpoint payload dict.

    Tolerates missing keys — anything absent falls back to the
    :class:`PipelineState` default. Unknown extra keys are ignored so
    forward-compatible writers don't break readers.
    """
    state = PipelineState(session_id=str(payload.get("session_id") or ""))
    state.iteration = int(payload.get("iteration") or 0)
    if "model" in payload and payload["model"]:
        state.model = str(payload["model"])
    state.messages = list(payload.get("messages") or [])
    state.shared = dict(payload.get("shared") or {})
    state.metadata = dict(payload.get("metadata") or {})
    state.loop_decision = str(payload.get("loop_decision") or "continue")
    completion_signal = payload.get("completion_signal")
    state.completion_signal = str(completion_signal) if completion_signal else None
    completion_detail = payload.get("completion_detail")
    state.completion_detail = str(completion_detail) if completion_detail else None
    state.total_cost_usd = float(payload.get("total_cost_usd") or 0.0)
    usage = payload.get("token_usage") or {}
    if isinstance(usage, dict):
        state.token_usage = TokenUsage(
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
        )
    state.final_text = str(payload.get("final_text") or "")
    return state


def state_from_record(record: CheckpointRecord) -> PipelineState:
    """Convenience: round-trip a :class:`CheckpointRecord` to state."""
    return state_from_payload(record.payload)


async def restore_state_from_checkpoint(persister: Persister, checkpoint_id: str) -> PipelineState:
    """Read a checkpoint and reconstruct the :class:`PipelineState`.

    Raises :class:`CheckpointNotFound` when the persister returns
    ``None`` (typically: unknown id). Persister-side errors propagate
    directly so callers can distinguish "missing" from "backend down".
    """
    record: Optional[CheckpointRecord] = await persister.read(checkpoint_id)
    if record is None:
        raise CheckpointNotFound(f"checkpoint not found: {checkpoint_id!r}")
    return state_from_payload(record.payload)


__all__ = [
    "CheckpointNotFound",
    "restore_state_from_checkpoint",
    "state_from_payload",
    "state_from_record",
]
