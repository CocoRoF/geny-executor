"""SubAgent* tools — the persistent (owned, autonomous) delegate surface.

Distinct from the ``Agent`` tool (one-shot **sub-worker**: delegate a task,
get the answer back inline). These drive a persistent **sub-agent**: spawn an
owned instance, assign it tasks it completes autonomously, and read the
completion notifications from your inbox (the alarm).

All read ``context.extras["subagent_manager"]`` (a
:class:`~geny_executor.stages.s12_agent.persistent_subagent.SubAgentManager`)
the host wires at startup — same escape-hatch pattern as ``agent_orchestrator``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from geny_executor.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult

logger = logging.getLogger(__name__)


def _manager(context: ToolContext):
    return context.extras.get("subagent_manager")


def _no_manager() -> ToolResult:
    return ToolResult(
        content={
            "error": {
                "code": "NO_SUBAGENT_MANAGER",
                "message": (
                    "subagent_manager was not wired into ToolContext.extras. "
                    "Host must construct a SubAgentManager at startup."
                ),
            }
        },
        is_error=True,
    )


class SubAgentSpawnTool(Tool):
    """Create a persistent, owned sub-agent you can assign tasks to."""

    @property
    def name(self) -> str:
        return "SubAgentSpawn"

    @property
    def description(self) -> str:
        return (
            "Create a persistent sub-agent of a registered subagent_type that "
            "you own. Unlike Agent (one-shot), a sub-agent stays alive, keeps "
            "its memory across assignments, and completes assigned tasks "
            "autonomously — you are notified via SubAgentInboxRead when done. "
            "Returns its sub_agent_id."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "subagent_type": {
                    "type": "string",
                    "description": "Registered subagent type id (e.g. 'researcher').",
                },
                "sub_agent_id": {
                    "type": "string",
                    "description": "Optional stable id to reattach an existing instance.",
                },
            },
            "required": ["subagent_type"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(concurrency_safe=True, read_only=False, network_egress=True)

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        mgr = _manager(context)
        if mgr is None:
            return _no_manager()
        subagent_type = (input.get("subagent_type") or "").strip()
        if not subagent_type:
            return ToolResult(
                content={"error": {"code": "BAD_INPUT", "message": "subagent_type is required"}},
                is_error=True,
            )
        try:
            agent = await mgr.spawn(
                subagent_type,
                context.session_id,
                sub_agent_id=input.get("sub_agent_id"),
            )
        except KeyError as exc:
            return ToolResult(
                content={"error": {"code": "UNKNOWN_TYPE", "message": f"unknown subagent_type: {exc}"}},
                is_error=True,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                content={"error": {"code": "SPAWN_FAILED", "message": str(exc)}},
                is_error=True,
            )
        return ToolResult(content=agent.summary())


class SubAgentAssignTool(Tool):
    """Fully delegate a task to a persistent sub-agent (autonomous)."""

    @property
    def name(self) -> str:
        return "SubAgentAssign"

    @property
    def description(self) -> str:
        return (
            "Assign a task to a persistent sub-agent (by sub_agent_id). It runs "
            "autonomously in the background; you get a completion notification in "
            "your inbox (SubAgentInboxRead) when done. Returns an assignment_id."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "sub_agent_id": {"type": "string", "description": "Target sub-agent id."},
                "task": {"type": "string", "description": "The task to fully delegate."},
            },
            "required": ["sub_agent_id", "task"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(concurrency_safe=True, read_only=False, network_egress=True)

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        mgr = _manager(context)
        if mgr is None:
            return _no_manager()
        sub_agent_id = (input.get("sub_agent_id") or "").strip()
        task = input.get("task") or ""
        if not sub_agent_id or not task:
            return ToolResult(
                content={"error": {"code": "BAD_INPUT", "message": "sub_agent_id and task are required"}},
                is_error=True,
            )
        try:
            out = await mgr.assign(sub_agent_id, task, background=True)
        except KeyError:
            return ToolResult(
                content={"error": {"code": "UNKNOWN_SUBAGENT", "message": f"no such sub_agent_id: {sub_agent_id}"}},
                is_error=True,
            )
        return ToolResult(content=out)


class SubAgentListTool(Tool):
    """List your persistent sub-agents."""

    @property
    def name(self) -> str:
        return "SubAgentList"

    @property
    def description(self) -> str:
        return "List the persistent sub-agents you own, with their status."

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(concurrency_safe=True, read_only=True)

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        mgr = _manager(context)
        if mgr is None:
            return _no_manager()
        return ToolResult(content={"sub_agents": mgr.list(context.session_id)})


class SubAgentStopTool(Tool):
    """Stop and release a persistent sub-agent."""

    @property
    def name(self) -> str:
        return "SubAgentStop"

    @property
    def description(self) -> str:
        return "Stop a persistent sub-agent (cancels in-flight work, frees it)."

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {"sub_agent_id": {"type": "string"}},
            "required": ["sub_agent_id"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(concurrency_safe=True, read_only=False, destructive=True)

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        mgr = _manager(context)
        if mgr is None:
            return _no_manager()
        sub_agent_id = (input.get("sub_agent_id") or "").strip()
        stopped = await mgr.stop(sub_agent_id)
        if not stopped:
            return ToolResult(
                content={"error": {"code": "UNKNOWN_SUBAGENT", "message": f"no such sub_agent_id: {sub_agent_id}"}},
                is_error=True,
            )
        return ToolResult(content={"sub_agent_id": sub_agent_id, "stopped": True})


class SubAgentInboxReadTool(Tool):
    """Read (and clear) your sub-agent completion notifications — the alarm."""

    @property
    def name(self) -> str:
        return "SubAgentInboxRead"

    @property
    def description(self) -> str:
        return (
            "Read your inbox: completion/failure notifications from your "
            "persistent sub-agents. By default this drains (clears) what it "
            "returns. This is how you learn an assigned task finished."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "drain": {
                    "type": "boolean",
                    "description": "Clear the messages returned (default true).",
                }
            },
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(concurrency_safe=True, read_only=False)

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        mgr = _manager(context)
        if mgr is None:
            return _no_manager()
        drain = input.get("drain")
        drain = True if drain is None else bool(drain)
        msgs = mgr.read_inbox(context.session_id, drain=drain)
        return ToolResult(content={"messages": msgs, "count": len(msgs)})


__all__ = [
    "SubAgentSpawnTool",
    "SubAgentAssignTool",
    "SubAgentListTool",
    "SubAgentStopTool",
    "SubAgentInboxReadTool",
]
