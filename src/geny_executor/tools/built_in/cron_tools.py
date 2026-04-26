"""Three cron tools — CronCreate / CronDelete / CronList (PR-A.4.2).

Wiring: host injects ``cron_store`` into ``ToolContext.extras``.
Optional ``cron_runner`` triggers a refresh after Create/Delete so
schedule changes take effect immediately.
"""

from __future__ import annotations

from typing import Any, Dict

from geny_executor.cron.types import CronJob, CronJobStatus
from geny_executor.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult


def _err(code: str, message: str) -> ToolResult:
    return ToolResult(content={"error": {"code": code, "message": message}}, is_error=True)


async def _refresh(context: ToolContext) -> None:
    runner = context.extras.get("cron_runner")
    refresh = getattr(runner, "refresh", None) if runner else None
    if callable(refresh):
        result = refresh()
        if hasattr(result, "__await__"):
            await result


def _validate_cron(expr: str) -> str | None:
    """Returns error message or None on success."""
    try:
        from croniter import croniter, CroniterBadCronError  # type: ignore[import-not-found]
    except ImportError:
        # croniter is an optional dep; without it we accept any string
        # and let the runner fail loudly when it tries to schedule.
        return None
    try:
        croniter(expr)
    except (CroniterBadCronError, KeyError, ValueError, TypeError) as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001
        return str(exc)
    return None


# ── CronCreate ───────────────────────────────────────────────────────


class CronCreateTool(Tool):
    @property
    def name(self) -> str:
        return "CronCreate"

    @property
    def description(self) -> str:
        return (
            "Create a recurring scheduled job. cron_expr is the standard "
            "5-field expression (minute hour day month weekday). target_kind "
            "matches a registered BackgroundTaskExecutor."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "pattern": "^[a-zA-Z0-9_-]+$", "maxLength": 64},
                "cron_expr": {"type": "string"},
                "target_kind": {"type": "string"},
                "payload": {"type": "object"},
                "description": {"type": "string"},
            },
            "required": ["name", "cron_expr", "target_kind"],
        }

    def capabilities(self, input):
        return ToolCapabilities(concurrency_safe=False, destructive=True)

    async def execute(self, input, context):
        store = context.extras.get("cron_store")
        if store is None:
            return _err("NO_STORE", "cron_store not wired into ctx.extras")
        err = _validate_cron(input["cron_expr"])
        if err is not None:
            return _err("INVALID_CRON_EXPR", err)
        existing = await store.get(input["name"])
        if existing is not None:
            return _err("NAME_EXISTS", f"cron job {input['name']!r} already exists")
        job = CronJob(
            name=input["name"],
            cron_expr=input["cron_expr"],
            target_kind=input["target_kind"],
            payload=dict(input.get("payload") or {}),
            description=input.get("description"),
        )
        await store.put(job)
        await _refresh(context)
        return ToolResult(content={"name": job.name, "status": job.status.value})


# ── CronDelete ───────────────────────────────────────────────────────


class CronDeleteTool(Tool):
    @property
    def name(self) -> str:
        return "CronDelete"

    @property
    def description(self) -> str:
        return "Delete a cron job by name."

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}

    def capabilities(self, input):
        return ToolCapabilities(concurrency_safe=False, destructive=True)

    async def execute(self, input, context):
        store = context.extras.get("cron_store")
        if store is None:
            return _err("NO_STORE", "cron_store not wired into ctx.extras")
        deleted = await store.delete(input["name"])
        if not deleted:
            return _err("NOT_FOUND", f"cron job {input['name']!r} not found")
        await _refresh(context)
        return ToolResult(content={"deleted": input["name"]})


# ── CronList ─────────────────────────────────────────────────────────


class CronListTool(Tool):
    @property
    def name(self) -> str:
        return "CronList"

    @property
    def description(self) -> str:
        return "List all cron jobs (with optional only_enabled filter)."

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {"only_enabled": {"type": "boolean", "default": False}},
        }

    def capabilities(self, input):
        return ToolCapabilities(concurrency_safe=True, read_only=True, idempotent=True)

    async def execute(self, input, context):
        store = context.extras.get("cron_store")
        if store is None:
            return _err("NO_STORE", "cron_store not wired into ctx.extras")
        jobs = await store.list(only_enabled=bool(input.get("only_enabled", False)))
        return ToolResult(content={"jobs": [j.to_dict() for j in jobs]})


__all__ = ["CronCreateTool", "CronDeleteTool", "CronListTool"]
