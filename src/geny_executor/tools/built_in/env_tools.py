"""Built-in self-modifying-environment tool (``env``).

A session uses this ONE tool to inspect and edit its OWN operating environment
at runtime — system prompt, active tools, active skills — and to persist the
result for itself. It is a thin dispatcher over the live
:class:`~geny_executor.core.environment_control.PipelineEnvironment` controller,
which the pipeline injects into ``ToolContext.environment``.

Why one dispatcher tool (not a tool per action): the capability must be present
by default, but a dozen always-on schemas would occupy context for every
session that never edits its environment. So the always-on footprint is a
single lean tool; the detailed how-to (every action, when/why, recipes) lives
in the bundled ``environment`` skill, loaded on demand (progressive
disclosure).

Mutations take effect on the NEXT turn: the prompt is rebuilt every turn, and
the tool registry's version bump makes Stage 3 re-derive the tools array.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from geny_executor.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult

_NO_ENV = (
    "Self-modifying environment is not enabled for this session "
    "(no environment controller attached)."
)

# action -> (controller method name, is_async, arg names). create_skill /
# edit_skill are added by the skill-authoring layer (registered on the class).
_ACTIONS = {
    "view": ("snapshot", False, ()),
    "get_prompt": ("get_prompt", False, ()),
    "set_prompt": ("set_prompt", False, ("prompt",)),
    "append_prompt": ("append_prompt", False, ("text",)),
    "enable_tool": ("enable_tool", False, ("name",)),
    "disable_tool": ("disable_tool", False, ("name",)),
    "enable_skill": ("enable_skill", False, ("skill_id",)),
    "disable_skill": ("disable_skill", False, ("skill_id",)),
    "create_skill": (
        "create_skill",
        False,
        ("skill_id", "description", "body", "allowed_tools"),
    ),
    "edit_skill": ("edit_skill", False, ("skill_id", "description", "body", "allowed_tools")),
    "forge_tool": (
        "forge_tool",
        False,
        (
            "name",
            "description",
            "entrypoint",
            "runtime",
            "input_schema",
            "argv",
            "timeout_s",
            "workdir",
            "network_egress",
            "read_only",
        ),
    ),
    "get_settings": ("get_settings", False, ("reveal",)),
    "set_setting": ("set_setting", False, ("key", "field", "value")),
    "get_config": ("get_config", False, ()),
    "set_config": ("set_config", False, ("key", "value")),
    "changelog": ("changelog", False, ("limit",)),
    "save": ("save", True, ()),
    "save_pack": ("save_pack", True, ("name", "description", "tools", "skills")),
}


class EnvTool(Tool):
    """Single dispatcher for self-environment inspection + edits."""

    # Subclasses/host layers may extend this (e.g. skill authoring adds
    # create_skill / edit_skill); kept on the class so the schema's action
    # enum stays in sync.
    actions: Dict[str, Any] = dict(_ACTIONS)

    @property
    def name(self) -> str:
        return "env"

    @property
    def description(self) -> str:
        return (
            "Inspect and edit your OWN operating environment at runtime — system "
            "prompt, active tools, and active skills — within what's available; "
            "changes apply from the next turn. One tool, many actions "
            f"(action one of: {', '.join(sorted(self.actions))}). "
            "Load the 'environment' skill for exact usage and recipes before "
            "making non-trivial edits."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": sorted(self.actions),
                    "description": "What to do. 'view' first to see current state.",
                },
                "args": {
                    "type": "object",
                    "description": (
                        "Arguments for the action, e.g. set_prompt -> "
                        '{"prompt": "..."}; enable_tool -> {"name": "web_search"}; '
                        'enable_skill -> {"skill_id": "gapt"}. See the environment '
                        "skill for each action's args."
                    ),
                    "additionalProperties": True,
                },
            },
            "required": ["action"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        action = (input or {}).get("action")
        read_only = action in ("view", "get_prompt", "changelog", "get_settings", "get_config")
        # Env edits change the active toolset/prompt — never run them in parallel
        # with other tools.
        return ToolCapabilities(concurrency_safe=False, read_only=read_only)

    async def execute(
        self, input: Dict[str, Any], context: Optional[ToolContext] = None
    ) -> ToolResult:
        env = getattr(context, "environment", None) if context else None
        if env is None:
            return ToolResult(content=_NO_ENV, is_error=True)

        inp = input or {}
        action = str(inp.get("action") or "").strip()
        spec = self.actions.get(action)
        if spec is None:
            return ToolResult(
                content=(
                    f"unknown env action {action!r}. Valid: {', '.join(sorted(self.actions))}"
                ),
                is_error=True,
            )
        method_name, is_async, arg_names = spec
        args = inp.get("args") or {}
        if not isinstance(args, dict):
            return ToolResult(content="'args' must be an object", is_error=True)

        method = getattr(env, method_name, None)
        if method is None:
            return ToolResult(
                content=f"env action {action!r} is not supported in this build",
                is_error=True,
            )

        call_kwargs = {n: args[n] for n in arg_names if n in args}
        try:
            result = (
                await method(**call_kwargs) if is_async else method(**call_kwargs)
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(content=f"env {action} failed: {exc}", is_error=True)

        return self._format(action, result)

    @staticmethod
    def _format(action: str, result: Any) -> ToolResult:
        # Read actions return data (dict/str/list) → JSON or text.
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], bool):
            ok, msg = result
            return ToolResult(content=str(msg), is_error=not ok)
        if isinstance(result, str):
            return ToolResult(content=result)
        return ToolResult(content=json.dumps(result, ensure_ascii=False, default=str))
