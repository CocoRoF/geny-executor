"""AgentTool — spawn a sub-agent of a registered SubagentType (PR-A.1.4).

The LLM calls ``Agent`` with a registered ``subagent_type`` and a
prompt; the tool dispatches to a :class:`SubagentTypeOrchestrator`
the host wired into ``context.extras["agent_orchestrator"]``.

Why ``extras`` rather than a dedicated field on :class:`ToolContext`:
the orchestrator is host-supplied (FastAPI lifespan / CLI bootstrap)
and we don't want every consumer to depend on the s12_agent module
just because they import :class:`ToolContext`. The extras bag is
already the documented escape hatch for host-specific data, and the
key is namespaced (``agent_orchestrator``) so it can't collide.

Recursion safety: ``context.extras["agent_depth"]`` is consulted; if
present and >= ``max_depth`` (default 3), the tool refuses to spawn
another sub-agent. The host increments the depth in the spawned
sub-pipeline's context.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from geny_executor.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult

logger = logging.getLogger(__name__)


_DEFAULT_MAX_DEPTH = 3


class AgentTool(Tool):
    """Spawn a sub-agent of the specified type with a prompt.

    Returns the sub-agent's final assistant message (or whatever the
    orchestrator's ``run_subagent`` / ``spawn`` returns).

    The orchestrator is read from ``context.extras["agent_orchestrator"]``;
    the host wires it at startup. Tools that recurse (``Agent`` calling
    ``Agent``) are bounded by ``max_depth`` to prevent runaway costs.
    """

    @property
    def name(self) -> str:
        return "Agent"

    @property
    def description(self) -> str:
        return (
            "Spawn a sub-agent of the specified subagent_type with a prompt. "
            "Returns the sub-agent's final result. Use sparingly — each "
            "sub-agent burns tokens and time. Recursion is depth-limited."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "subagent_type": {
                    "type": "string",
                    "description": (
                        "Registered subagent type id (e.g. 'researcher', "
                        "'code-coder'). Use ListSubagentTypes to discover."
                    ),
                },
                "prompt": {
                    "type": "string",
                    "description": "Initial user prompt for the sub-agent.",
                },
                "model": {
                    "type": "string",
                    "description": "Optional model override for this sub-agent only.",
                },
            },
            "required": ["subagent_type", "prompt"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(
            concurrency_safe=True,
            read_only=False,
            destructive=False,
            network_egress=True,
        )

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        depth = int(context.extras.get("agent_depth", 0))
        max_depth = int(context.extras.get("agent_max_depth", _DEFAULT_MAX_DEPTH))
        if depth >= max_depth:
            return ToolResult(
                content={
                    "error": {
                        "code": "MAX_DEPTH",
                        "message": (
                            f"AgentTool recursion depth {depth} >= max_depth {max_depth}; "
                            "refusing to spawn further sub-agents."
                        ),
                    },
                },
                is_error=True,
            )

        orch = context.extras.get("agent_orchestrator")
        if orch is None:
            return ToolResult(
                content={
                    "error": {
                        "code": "NO_ORCHESTRATOR",
                        "message": (
                            "agent_orchestrator was not wired into ToolContext.extras. "
                            "Host must register a SubagentTypeOrchestrator at startup."
                        ),
                    },
                },
                is_error=True,
            )

        subagent_type = input.get("subagent_type", "")
        prompt = input.get("prompt", "")
        model: Optional[str] = input.get("model")
        if not subagent_type:
            return ToolResult(
                content={"error": {"code": "BAD_INPUT", "message": "subagent_type is required"}},
                is_error=True,
            )
        if not prompt:
            return ToolResult(
                content={"error": {"code": "BAD_INPUT", "message": "prompt is required"}},
                is_error=True,
            )

        runner = getattr(orch, "run_subagent", None) or getattr(orch, "spawn", None)
        if runner is None:
            return ToolResult(
                content={
                    "error": {
                        "code": "ORCHESTRATOR_API",
                        "message": (
                            "agent_orchestrator object has no run_subagent or spawn method."
                        ),
                    },
                },
                is_error=True,
            )

        try:
            result = await runner(subagent_type, prompt, model=model)
        except KeyError as exc:
            return ToolResult(
                content={
                    "error": {
                        "code": "UNKNOWN_TYPE",
                        "message": f"unknown subagent_type: {exc}",
                    },
                },
                is_error=True,
            )
        except Exception as exc:  # noqa: BLE001 — surface to LLM
            logger.warning(
                "agent_tool_subagent_failed",
                extra={"subagent_type": subagent_type, "error": str(exc)},
            )
            return ToolResult(
                content={
                    "error": {
                        "code": "SUBAGENT_FAILED",
                        "message": str(exc),
                    },
                },
                is_error=True,
            )

        # Mirror LocalAgentExecutor's serialization shape so callers see
        # consistent output regardless of orchestrator return type.
        if isinstance(result, (str, bytes, bytearray)):
            payload = result if isinstance(result, str) else bytes(result).decode(errors="replace")
        else:
            payload = result
        return ToolResult(content={"subagent_type": subagent_type, "result": payload})


__all__ = ["AgentTool"]
