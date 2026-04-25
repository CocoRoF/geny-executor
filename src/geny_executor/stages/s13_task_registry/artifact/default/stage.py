"""Stage 13: Task Registry — real implementation (S9b.2).

Drains tasks queued on ``state.shared[PENDING_TASKS_KEY]``, registers
them in a :class:`TaskRegistry`, and runs the configured
:class:`TaskPolicy`. After the policy returns, the stage refreshes
``state.shared[TASKS_BY_STATUS_KEY]`` with the current group-by-status
view so downstream stages (Stage 2 Context, Stage 14 Evaluate, host
UI) can read the latest snapshot.

The stage is opt-in via the existing ``with_task_registry()`` builder
hook — pipelines that don't need task tracking see no behaviour
change because the queue is empty.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from geny_executor.core.slot import StrategySlot
from geny_executor.core.stage import Stage
from geny_executor.core.state import PipelineState
from geny_executor.stages.s13_task_registry.artifact.default.policies import (
    EagerWaitPolicy,
    FireAndForgetPolicy,
    TimedWaitPolicy,
)
from geny_executor.stages.s13_task_registry.artifact.default.registry import (
    InMemoryRegistry,
)
from geny_executor.stages.s13_task_registry.interface import (
    PENDING_TASKS_KEY,
    TASKS_BY_STATUS_KEY,
    TaskPolicy,
    TaskRegistry,
)
from geny_executor.stages.s13_task_registry.types import TaskRecord, TaskStatus

logger = logging.getLogger(__name__)


def _coerce_record(raw: Any) -> Optional[TaskRecord]:
    """Accept dict or TaskRecord; return a TaskRecord or None on bad shape."""
    if isinstance(raw, TaskRecord):
        return raw
    if not isinstance(raw, dict):
        return None
    task_id = str(raw.get("task_id", "") or raw.get("id", ""))
    if not task_id:
        return None
    status_raw = raw.get("status", TaskStatus.PENDING.value)
    try:
        status = TaskStatus(status_raw) if isinstance(status_raw, str) else status_raw
    except ValueError:
        status = TaskStatus.PENDING
    return TaskRecord(
        task_id=task_id,
        kind=str(raw.get("kind", "") or ""),
        payload=dict(raw.get("payload") or {}),
        status=status,
    )


class TaskRegistryStage(Stage[Any, Any]):
    """Stage 13: Task Registry.

    Two strategy slots:

    * ``registry`` — storage backend (default :class:`InMemoryRegistry`)
    * ``policy`` — what to do with newly-registered tasks each turn
      (default :class:`FireAndForgetPolicy`)
    """

    def __init__(
        self,
        registry: Optional[TaskRegistry] = None,
        policy: Optional[TaskPolicy] = None,
    ):
        self._slots: Dict[str, StrategySlot] = {
            "registry": StrategySlot(
                name="registry",
                strategy=registry or InMemoryRegistry(),
                registry={
                    "in_memory": InMemoryRegistry,
                },
                description="Backend store for TaskRecord instances",
            ),
            "policy": StrategySlot(
                name="policy",
                strategy=policy or FireAndForgetPolicy(),
                registry={
                    "fire_and_forget": FireAndForgetPolicy,
                    "eager_wait": EagerWaitPolicy,
                    "timed_wait": TimedWaitPolicy,
                },
                description="What to do with newly-drained tasks",
            ),
        }

    @property
    def name(self) -> str:
        return "task_registry"

    @property
    def order(self) -> int:
        return 13

    @property
    def category(self) -> str:
        return "orchestration"

    @property
    def _registry(self) -> TaskRegistry:
        return self._slots["registry"].strategy  # type: ignore[return-value]

    @property
    def _policy(self) -> TaskPolicy:
        return self._slots["policy"].strategy  # type: ignore[return-value]

    def get_strategy_slots(self) -> Dict[str, StrategySlot]:
        return self._slots

    def _drain_pending(self, state: PipelineState, iteration: int) -> List[TaskRecord]:
        raw_queue = state.shared.get(PENDING_TASKS_KEY) or []
        drained: List[TaskRecord] = []
        for raw in list(raw_queue):
            record = _coerce_record(raw)
            if record is None:
                state.add_event(
                    "task_registry.invalid_payload",
                    {"payload_repr": repr(raw)[:200]},
                )
                continue
            record.iteration_seen = iteration
            self._registry.register(record)
            drained.append(record)
        # Always clear the queue, even on coercion errors, so a bad
        # payload doesn't haunt subsequent iterations.
        state.shared[PENDING_TASKS_KEY] = []
        return drained

    def _publish_status_view(self, state: PipelineState) -> Dict[str, List[Dict[str, Any]]]:
        view: Dict[str, List[Dict[str, Any]]] = {}
        for status, records in self._registry.by_status().items():
            view[status] = [r.to_dict() for r in records]
        state.shared[TASKS_BY_STATUS_KEY] = view
        return view

    async def execute(self, input: Any, state: PipelineState) -> Any:
        new_tasks = self._drain_pending(state, iteration=state.iteration)

        if new_tasks:
            for record in new_tasks:
                state.add_event(
                    "task.registered",
                    {
                        "task_id": record.task_id,
                        "kind": record.kind,
                        "status": record.status.value,
                    },
                )
            try:
                await self._policy.apply(new_tasks, self._registry, state)
            except Exception as exc:  # noqa: BLE001 — never block the loop on policy bugs
                logger.warning("Task policy %s raised %s; continuing", self._policy.name, exc)
                state.add_event(
                    "task_registry.policy_error",
                    {"policy": self._policy.name, "error": str(exc)},
                )

        view = self._publish_status_view(state)
        counts = {status: len(records) for status, records in view.items()}
        state.add_event(
            "task_registry.synced",
            {"new": len(new_tasks), "by_status": counts, "total": sum(counts.values())},
        )
        return input
