"""``/tasks`` — list background tasks (read-only summary inline)."""

from __future__ import annotations

from geny_executor.slash_commands.types import (
    SlashCategory,
    SlashCommand,
    SlashContext,
    SlashResult,
)
from geny_executor.stages.s13_task_registry.types import TaskFilter, TaskStatus


class TasksCommand(SlashCommand):
    name = "tasks"
    description = "List background tasks. Optional arg: status filter (pending|running|done|failed|cancelled)."
    category = SlashCategory.INTROSPECTION

    async def execute(self, args, ctx: SlashContext) -> SlashResult:
        registry = ctx.extras.get("task_registry")
        if registry is None:
            return SlashResult(content="No task_registry wired into ctx.extras.", success=False)
        status: TaskStatus | None = None
        if args:
            try:
                status = TaskStatus(args[0])
            except ValueError:
                return SlashResult(
                    content=f"Unknown status: {args[0]}. Use pending|running|done|failed|cancelled.",
                    success=False,
                )
        rows = registry.list_filtered(TaskFilter(status=status, limit=20))
        if not rows:
            return SlashResult(content="No tasks." if status is None else f"No {status.value} tasks.")
        header = "**Tasks**" if status is None else f"**Tasks** ({status.value})"
        lines = [header, ""]
        for r in rows:
            duration = ""
            if r.started_at and r.completed_at:
                delta = (r.completed_at - r.started_at).total_seconds()
                duration = f" ({delta:.1f}s)"
            lines.append(f"- `{r.task_id}` [{r.status.value}] kind=`{r.kind}`{duration}")
            if r.error:
                lines.append(f"  error: {r.error[:200]}")
        return SlashResult(content="\n".join(lines))


__all__ = ["TasksCommand"]
