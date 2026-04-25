"""Stage 20: Persist — real implementation (S9b.5).

Decides via the ``frequency`` slot whether to write a checkpoint
this turn, then asks the ``persister`` slot to write it. Records the
last checkpoint id at ``state.shared['last_checkpoint']`` and
appends to an audit log at ``state.shared['checkpoint_history']``.

The default :class:`NoPersister` is a no-op so existing pipelines
pay zero cost. Hosts opt in by swapping in :class:`FilePersister`
(or their own backend) and choosing a frequency.

The serialised payload covers the *non-runtime* portion of the
state — messages, shared, metadata, iteration counters, completion
signals, token usage, total cost. The live ``llm_client`` /
``session_runtime`` references are intentionally not persisted
(they're runtime-only). ``Pipeline.resume_from_checkpoint`` for
crash-recovery is deferred to a follow-up sprint; this PR ships
the write half so hosts can start collecting checkpoints.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from geny_executor.core.slot import StrategySlot
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState
from geny_executor.stages.s20_persist.artifact.default.frequencies import (
    EveryNTurnsFrequency,
    EveryTurnFrequency,
    OnSignificantFrequency,
)
from geny_executor.stages.s20_persist.artifact.default.persisters import (
    FilePersister,
    NoPersister,
)
from geny_executor.stages.s20_persist.interface import (
    CHECKPOINT_HISTORY_KEY,
    LAST_CHECKPOINT_KEY,
    FrequencyPolicy,
    Persister,
)
from geny_executor.stages.s20_persist.types import CheckpointRecord

logger = logging.getLogger(__name__)


def _serialise_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Best-effort copy of message list — keeps strings/dicts as-is.

    Non-serialisable content (binary blobs, custom objects) gets
    converted to its string repr so the JSON layer doesn't blow up.
    Hosts that need lossless round-trip should plug their own
    persister and override the payload shape.
    """
    out: List[Dict[str, Any]] = []
    for msg in messages:
        out.append(
            {
                "role": str(msg.get("role", "")),
                "content": _safe(msg.get("content")),
            }
        )
    return out


def _safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    # Fallback: stringified repr.
    return str(value)


def _build_payload(state: PipelineState) -> Dict[str, Any]:
    return {
        "session_id": state.session_id,
        "iteration": state.iteration,
        "model": state.model,
        "messages": _serialise_messages(state.messages),
        "shared": _safe(state.shared),
        "metadata": _safe(state.metadata),
        "loop_decision": state.loop_decision,
        "completion_signal": state.completion_signal,
        "completion_detail": state.completion_detail,
        "total_cost_usd": state.total_cost_usd,
        "token_usage": {
            "input_tokens": state.token_usage.input_tokens,
            "output_tokens": state.token_usage.output_tokens,
            "total_tokens": state.token_usage.total_tokens,
        },
        "final_text": state.final_text,
    }


class PersistStage(Stage[Any, Any]):
    """Stage 20: Persist.

    Two slots:

    * ``persister`` — backend (default :class:`NoPersister`).
    * ``frequency`` — when to write (default
      :class:`EveryTurnFrequency`).
    """

    def __init__(
        self,
        persister: Optional[Persister] = None,
        frequency: Optional[FrequencyPolicy] = None,
    ):
        self._slots: Dict[str, StrategySlot] = {
            "persister": StrategySlot(
                name="persister",
                strategy=persister or NoPersister(),
                registry={
                    "no_persist": NoPersister,
                    "file": FilePersister,
                },
                description="Backend for checkpoint writes",
            ),
            "frequency": StrategySlot(
                name="frequency",
                strategy=frequency or EveryTurnFrequency(),
                registry={
                    "every_turn": EveryTurnFrequency,
                    "every_n_turns": EveryNTurnsFrequency,
                    "on_significant": OnSignificantFrequency,
                },
                description="Cadence policy for checkpoint writes",
            ),
        }

    @property
    def name(self) -> str:
        return "persist"

    @property
    def order(self) -> int:
        return 20

    @property
    def category(self) -> str:
        return "finalize"

    @property
    def _persister(self) -> Persister:
        return self._slots["persister"].strategy  # type: ignore[return-value]

    @property
    def _frequency(self) -> FrequencyPolicy:
        return self._slots["frequency"].strategy  # type: ignore[return-value]

    def get_strategy_slots(self) -> Dict[str, StrategySlot]:
        return self._slots

    def should_bypass(self, state: PipelineState) -> bool:
        # Default NoPersister never writes — short-circuit so the
        # stage doesn't even fire its events for the common no-op.
        return isinstance(self._persister, NoPersister)

    async def execute(self, input: Any, state: PipelineState) -> Any:
        if not self._frequency.should_persist(state):
            state.add_event(
                "checkpoint.skipped",
                {"frequency": self._frequency.name, "iteration": state.iteration},
            )
            return input

        record = CheckpointRecord(
            session_id=state.session_id,
            iteration=state.iteration,
            payload=_build_payload(state),
        )

        try:
            await self._persister.write(record, state)
        except Exception as exc:  # noqa: BLE001 — never wedge the loop on persister bugs
            logger.warning(
                "Persister %s raised %s; skipping checkpoint",
                self._persister.name,
                exc,
            )
            state.add_event(
                "checkpoint.persister_error",
                {"persister": self._persister.name, "error": str(exc)},
            )
            return input

        state.shared[LAST_CHECKPOINT_KEY] = record.checkpoint_id
        history: List[Any] = state.shared.setdefault(CHECKPOINT_HISTORY_KEY, [])
        history.append(
            {
                "checkpoint_id": record.checkpoint_id,
                "session_id": record.session_id,
                "iteration": record.iteration,
                "created_at": record.created_at.isoformat(),
            }
        )
        state.add_event(
            "checkpoint.written",
            {
                "checkpoint_id": record.checkpoint_id,
                "session_id": record.session_id,
                "iteration": record.iteration,
                "persister": self._persister.name,
            },
        )
        return input
