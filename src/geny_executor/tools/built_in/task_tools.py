"""Six task lifecycle tools (PR-A.1.5).

Exposes the BackgroundTaskRunner + TaskRegistry surface to the LLM
as callable tools:

* ``TaskCreate`` — submit a new background task.
* ``TaskGet``    — fetch one record by id.
* ``TaskList``   — list with optional status / kind / limit filter.
* ``TaskUpdate`` — mutate user-writable fields (payload only; status
                   transitions stay with the runner / registry).
* ``TaskOutput`` — read accumulated output bytes (offset + limit).
* ``TaskStop``   — cancel a running task.

Wiring contract: hosts inject ``task_registry`` and ``task_runner``
into ``ToolContext.extras`` at startup. Tools that need only the
registry (Get / List / Update / Output) keep working even if the
runner is absent — the host may want to read state from a backend
populated by a different process.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional

from geny_executor.runtime.task_runner import BackgroundTaskRunner
from geny_executor.stages.s13_task_registry.interface import TaskRegistry
from geny_executor.stages.s13_task_registry.types import (
    TaskFilter,
    TaskRecord,
    TaskStatus,
)
from geny_executor.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult

logger = logging.getLogger(__name__)

_MAX_OUTPUT_LIMIT = 1 * 1024 * 1024  # 1 MiB cap per call so an LLM
                                     # asking for everything doesn't blow
                                     # the response budget.


def _registry(ctx: ToolContext) -> Optional[TaskRegistry]:
    return ctx.extras.get("task_registry")


def _runner(ctx: ToolContext) -> Optional[BackgroundTaskRunner]:
    return ctx.extras.get("task_runner")


def _err(code: str, message: str) -> ToolResult:
    return ToolResult(
        content={"error": {"code": code, "message": message}},
        is_error=True,
    )


# ── TaskCreate ───────────────────────────────────────────────────────


class TaskCreateTool(Tool):
    @property
    def name(self) -> str:
        return "TaskCreate"

    @property
    def description(self) -> str:
        return (
            "Create and submit a background task. Returns task_id. "
            "Built-in kinds: 'local_bash' (payload.command) / "
            "'local_agent' (payload.subagent_type + payload.prompt)."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "minLength": 1},
                "payload": {"type": "object"},
                "task_id": {
                    "type": "string",
                    "description": "Optional. Auto-generated UUID if absent.",
                },
            },
            "required": ["kind"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(concurrency_safe=True, destructive=True)

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        runner = _runner(context)
        if runner is None:
            return _err("NO_RUNNER", "task_runner not configured in ToolContext.extras")
        kind = (input.get("kind") or "").strip()
        if not kind:
            return _err("BAD_INPUT", "kind is required")
        record = TaskRecord(
            task_id=input.get("task_id") or str(uuid.uuid4()),
            kind=kind,
            payload=dict(input.get("payload") or {}),
        )
        task_id = await runner.submit(record)
        # Re-read so the LLM sees the actual status (RUNNING by now,
        # or FAILED if there was no executor for the kind).
        registry = _registry(context)
        rec = registry.get(task_id) if registry else None
        return ToolResult(
            content={
                "task_id": task_id,
                "status": (rec.status.value if rec else "submitted"),
            },
        )


# ── TaskGet ──────────────────────────────────────────────────────────


class TaskGetTool(Tool):
    @property
    def name(self) -> str:
        return "TaskGet"

    @property
    def description(self) -> str:
        return "Fetch a single task record by id."

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(concurrency_safe=True, read_only=True, idempotent=True)

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        registry = _registry(context)
        if registry is None:
            return _err("NO_REGISTRY", "task_registry not configured")
        rec = registry.get(input["task_id"])
        if rec is None:
            return _err("NOT_FOUND", f"unknown task: {input['task_id']}")
        return ToolResult(content=rec.to_dict())


# ── TaskList ─────────────────────────────────────────────────────────


class TaskListTool(Tool):
    @property
    def name(self) -> str:
        return "TaskList"

    @property
    def description(self) -> str:
        return (
            "List tasks ordered by created_at desc. Optional filter by "
            "status (pending|running|done|failed|cancelled), kind, limit."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": [s.value for s in TaskStatus],
                },
                "kind": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200},
            },
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(concurrency_safe=True, read_only=True, idempotent=True)

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        registry = _registry(context)
        if registry is None:
            return _err("NO_REGISTRY", "task_registry not configured")
        status = input.get("status")
        rows = registry.list_filtered(
            TaskFilter(
                status=TaskStatus(status) if status else None,
                kind=input.get("kind"),
                limit=input.get("limit", 20),
            )
        )
        return ToolResult(content={"tasks": [r.to_dict() for r in rows]})


# ── TaskUpdate ───────────────────────────────────────────────────────


# Whitelist of fields the LLM is allowed to overwrite. Status is
# excluded — only the runner / registry should drive status
# transitions (otherwise an LLM could mark a still-running task as
# DONE and consumers polling for completion would get a wrong answer).
_USER_MUTABLE_FIELDS = {"payload"}


class TaskUpdateTool(Tool):
    @property
    def name(self) -> str:
        return "TaskUpdate"

    @property
    def description(self) -> str:
        return (
            "Update mutable fields on an existing task. Only 'payload' "
            "is user-mutable; status transitions are driven by the "
            "runner. Returns the updated record."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "payload": {"type": "object"},
            },
            "required": ["task_id"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(concurrency_safe=False, destructive=True)

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        registry = _registry(context)
        if registry is None:
            return _err("NO_REGISTRY", "task_registry not configured")
        rec = registry.get(input["task_id"])
        if rec is None:
            return _err("NOT_FOUND", f"unknown task: {input['task_id']}")
        rejected = sorted(set(input) - {"task_id"} - _USER_MUTABLE_FIELDS)
        if rejected:
            return _err(
                "FIELD_NOT_MUTABLE",
                f"these fields are not user-mutable: {rejected}",
            )
        if "payload" in input:
            rec.payload = dict(input["payload"])
        return ToolResult(content=rec.to_dict())


# ── TaskOutput ───────────────────────────────────────────────────────


class TaskOutputTool(Tool):
    @property
    def name(self) -> str:
        return "TaskOutput"

    @property
    def description(self) -> str:
        return (
            "Read accumulated task output bytes. Returns text (decoded "
            "as UTF-8 with replacement) plus byte_count + truncated. "
            "Use offset + limit to page through long outputs."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0, "default": 0},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": _MAX_OUTPUT_LIMIT,
                    "default": _MAX_OUTPUT_LIMIT,
                },
            },
            "required": ["task_id"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(concurrency_safe=True, read_only=True, idempotent=True)

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        registry = _registry(context)
        if registry is None:
            return _err("NO_REGISTRY", "task_registry not configured")
        offset = int(input.get("offset", 0))
        # Cap the per-call limit so an LLM asking for everything doesn't
        # blow the response budget. Callers can page via offset.
        requested = int(input.get("limit", _MAX_OUTPUT_LIMIT))
        limit = min(requested, _MAX_OUTPUT_LIMIT)
        chunk = await registry.read_output(input["task_id"], offset=offset, limit=limit)
        return ToolResult(
            content={
                "task_id": input["task_id"],
                "offset": offset,
                "byte_count": len(chunk),
                "truncated": len(chunk) >= limit,
                "text": chunk.decode("utf-8", errors="replace"),
            },
        )


# ── TaskStop ─────────────────────────────────────────────────────────


class TaskStopTool(Tool):
    @property
    def name(self) -> str:
        return "TaskStop"

    @property
    def description(self) -> str:
        return "Cancel a running task. Returns whether the task was actually stopped."

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(concurrency_safe=False, destructive=True)

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        runner = _runner(context)
        if runner is None:
            return _err("NO_RUNNER", "task_runner not configured")
        stopped = await runner.stop(input["task_id"])
        return ToolResult(content={"task_id": input["task_id"], "stopped": stopped})


__all__ = [
    "TaskCreateTool",
    "TaskGetTool",
    "TaskListTool",
    "TaskUpdateTool",
    "TaskOutputTool",
    "TaskStopTool",
]
