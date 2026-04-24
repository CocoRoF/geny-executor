"""Tool ABC uplift — Phase 1 Week 1 Checkpoint 1.

Tests for the additive extensions to the Tool base class:
- ToolCapabilities + PermissionDecision dataclasses
- ToolContext new optional fields (permission_mode, state_view, …)
- ToolResult new optional fields (display_text, persist_full, …)
- Tool optional methods (capabilities, check_permissions,
  prepare_permission_matcher, on_enter/on_exit/on_error, …)
- build_tool() factory

Backward-compat guarantees:
- Existing Tool subclasses (implementing only the 4 required members)
  continue to work.
- Existing ToolResult / ToolContext construction code keeps working.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest

from geny_executor.tools.base import (
    PermissionDecision,
    Tool,
    ToolCapabilities,
    ToolContext,
    ToolResult,
    build_tool,
)


# ─────────────────────────────────────────────────────────────────
# Existing minimal Tool — backward compat
# ─────────────────────────────────────────────────────────────────


class _LegacyTool(Tool):
    """Tool written before the uplift — should keep working."""

    @property
    def name(self) -> str:
        return "legacy"

    @property
    def description(self) -> str:
        return "Legacy tool that only implements the 4 required members."

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}

    async def execute(self, input, context):  # noqa: D401
        return ToolResult(content=f"legacy:{input['x']}")


class TestBackwardCompat:
    """Ensure the old API path still works."""

    def test_legacy_tool_instantiates(self) -> None:
        t = _LegacyTool()
        assert t.name == "legacy"
        assert t.aliases == ()
        assert t.is_mcp is False
        assert t.is_enabled() is True

    def test_legacy_tool_capabilities_default(self) -> None:
        t = _LegacyTool()
        caps = t.capabilities({"x": 1})
        assert caps == ToolCapabilities()
        # Defaults are fail-closed
        assert caps.concurrency_safe is False
        assert caps.destructive is False
        assert caps.max_result_chars == 100_000

    def test_legacy_tool_permission_default_allow(self) -> None:
        t = _LegacyTool()
        decision = asyncio.run(t.check_permissions({"x": 1}, ToolContext()))
        assert decision.behavior == "allow"
        assert decision.updated_input is None

    def test_legacy_tool_matcher_default(self) -> None:
        t = _LegacyTool()
        match = asyncio.run(t.prepare_permission_matcher({"x": 1}))
        assert match("legacy") is True
        assert match("other") is False
        assert match("legacy(sub)") is False  # default is exact-match

    def test_legacy_tool_lifecycle_hooks_no_op(self) -> None:
        t = _LegacyTool()
        ctx = ToolContext()
        # All three must be awaitable without error
        asyncio.run(t.on_enter({"x": 1}, ctx))
        asyncio.run(t.on_exit(ToolResult(content="r"), ctx))
        asyncio.run(t.on_error(RuntimeError("boom"), ctx))

    def test_legacy_tool_api_format(self) -> None:
        t = _LegacyTool()
        api = t.to_api_format()
        assert api["name"] == "legacy"
        assert "description" in api
        assert api["input_schema"]["type"] == "object"


# ─────────────────────────────────────────────────────────────────
# ToolCapabilities dataclass
# ─────────────────────────────────────────────────────────────────


class TestToolCapabilities:
    def test_defaults_are_fail_closed(self) -> None:
        c = ToolCapabilities()
        assert not c.concurrency_safe
        assert not c.read_only
        assert not c.destructive
        assert not c.idempotent
        assert not c.network_egress
        assert c.interrupt == "block"
        assert c.max_result_chars == 100_000

    def test_is_frozen(self) -> None:
        c = ToolCapabilities()
        with pytest.raises(Exception):
            c.concurrency_safe = True  # type: ignore[misc]

    def test_custom_instantiation(self) -> None:
        c = ToolCapabilities(
            concurrency_safe=True,
            read_only=True,
            max_result_chars=500_000,
        )
        assert c.concurrency_safe is True
        assert c.read_only is True
        assert c.max_result_chars == 500_000


# ─────────────────────────────────────────────────────────────────
# PermissionDecision
# ─────────────────────────────────────────────────────────────────


class TestPermissionDecision:
    def test_minimal(self) -> None:
        d = PermissionDecision(behavior="allow")
        assert d.behavior == "allow"
        assert d.updated_input is None
        assert d.reason is None

    def test_with_modified_input(self) -> None:
        d = PermissionDecision(
            behavior="allow",
            updated_input={"x": 42},
            reason="sanitized",
        )
        assert d.updated_input == {"x": 42}

    def test_is_frozen(self) -> None:
        d = PermissionDecision(behavior="deny")
        with pytest.raises(Exception):
            d.behavior = "allow"  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────
# ToolContext new optional fields
# ─────────────────────────────────────────────────────────────────


class TestToolContextExtensions:
    def test_default_values(self) -> None:
        ctx = ToolContext()
        assert ctx.permission_mode == "default"
        assert ctx.state_view is None
        assert ctx.event_emit is None
        assert ctx.parent_tool_use_id is None
        assert ctx.extras == {}

    def test_all_new_fields_settable(self) -> None:
        events: list[tuple[str, dict]] = []

        def _emit(event_type: str, payload: dict) -> None:
            events.append((event_type, payload))

        ctx = ToolContext(
            session_id="sess-1",
            working_dir="/tmp",
            permission_mode="plan",
            state_view={"ro": "view"},
            event_emit=_emit,
            parent_tool_use_id="toolu_abc",
            extras={"creature_role": "vtuber"},
        )
        ctx.event_emit("x", {"k": "v"})  # type: ignore[misc]
        assert ctx.permission_mode == "plan"
        assert ctx.state_view == {"ro": "view"}
        assert ctx.parent_tool_use_id == "toolu_abc"
        assert ctx.extras["creature_role"] == "vtuber"
        assert events == [("x", {"k": "v"})]

    def test_backward_compat_positional_construction(self) -> None:
        # Existing code constructs ToolContext with just session_id /
        # working_dir; the new fields must not break that.
        ctx = ToolContext(session_id="s", working_dir="/w")
        assert ctx.session_id == "s"
        assert ctx.working_dir == "/w"
        assert ctx.permission_mode == "default"  # default picked up


# ─────────────────────────────────────────────────────────────────
# ToolResult new optional fields
# ─────────────────────────────────────────────────────────────────


class TestToolResultExtensions:
    def test_defaults_unchanged(self) -> None:
        r = ToolResult()
        assert r.content == ""
        assert r.is_error is False
        assert r.metadata == {}
        # New fields
        assert r.display_text is None
        assert r.persist_full is None
        assert r.state_mutations == {}
        assert r.artifacts == {}
        assert r.new_messages == []
        assert r.mcp_meta is None

    def test_to_api_format_uses_display_text_when_set(self) -> None:
        r = ToolResult(
            content={"huge": "payload" * 1000},
            display_text="Summary: saved to /tmp/x.json",
        )
        api = r.to_api_format("tu_1")
        assert api["content"] == "Summary: saved to /tmp/x.json"
        assert api["type"] == "tool_result"
        assert api["tool_use_id"] == "tu_1"

    def test_to_api_format_falls_back_to_content_without_display_text(self) -> None:
        r = ToolResult(content="plain text")
        api = r.to_api_format("tu_2")
        assert api["content"] == "plain text"

    def test_to_api_format_structured_error_header(self) -> None:
        r = ToolResult(
            content={"error": {"code": "EPERM", "message": "nope"}},
            is_error=True,
        )
        api = r.to_api_format("tu_3")
        assert api["is_error"] is True
        assert "ERROR EPERM: nope" in api["content"]

    def test_to_api_format_dict_without_error_block(self) -> None:
        r = ToolResult(content={"k": "v"})
        api = r.to_api_format("tu_4")
        # Serialized JSON
        assert '"k"' in api["content"]
        assert '"v"' in api["content"]

    def test_state_mutations_and_artifacts_round_trip(self) -> None:
        r = ToolResult(
            content="done",
            state_mutations={"plugin.foo.key": 42},
            artifacts={"out_path": "/tmp/out.json"},
            new_messages=[{"role": "user", "content": [{"type": "text", "text": "next"}]}],
        )
        assert r.state_mutations == {"plugin.foo.key": 42}
        assert r.artifacts["out_path"] == "/tmp/out.json"
        assert r.new_messages[0]["role"] == "user"


# ─────────────────────────────────────────────────────────────────
# Tool optional overrides
# ─────────────────────────────────────────────────────────────────


class _RichTool(Tool):
    """Tool that exercises every optional override."""

    aliases = ("rich", "rich-tool")

    @property
    def name(self) -> str:
        return "Rich"

    @property
    def description(self) -> str:
        return "A tool that overrides everything."

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }

    def output_schema(self):
        return {"type": "object", "properties": {"ok": {"type": "boolean"}}}

    def validate_input(self, raw):
        cmd = raw.get("command", "")
        if "DROP" in cmd.upper():
            raise ValueError("SQL injection guess: DROP not allowed")
        return raw

    def capabilities(self, input):
        cmd = input.get("command", "")
        return ToolCapabilities(
            concurrency_safe=False,
            destructive=cmd.startswith("rm "),
            read_only=cmd.startswith("ls ") or cmd.startswith("cat "),
            network_egress=cmd.startswith("curl "),
        )

    async def check_permissions(self, input, context):
        if context.permission_mode == "plan" and "rm " in input.get("command", ""):
            return PermissionDecision(behavior="ask", reason="destructive in plan mode")
        return PermissionDecision(behavior="allow")

    async def prepare_permission_matcher(self, input):
        cmd = input.get("command", "")

        def _match(pattern: str) -> bool:
            if pattern == self.name:
                return True
            if pattern.startswith(f"{self.name}(") and pattern.endswith(")"):
                inner = pattern[len(self.name) + 1 : -1]
                import fnmatch

                return fnmatch.fnmatch(cmd, inner)
            return False

        return _match

    def user_facing_name(self, input):
        return f"Rich: {input.get('command', '?')[:20]}"

    def activity_description(self, input):
        return f"Running: {input.get('command', '')}"

    async def execute(self, input, context):
        return ToolResult(content={"ok": True, "cmd": input["command"]})


class TestToolOptionalOverrides:
    def test_output_schema(self) -> None:
        t = _RichTool()
        sch = t.output_schema()
        assert sch is not None
        assert sch["properties"]["ok"]["type"] == "boolean"

    def test_validate_input_normalises(self) -> None:
        t = _RichTool()
        # pass
        assert t.validate_input({"command": "ls"}) == {"command": "ls"}
        # reject
        with pytest.raises(ValueError):
            t.validate_input({"command": "DROP TABLE users"})

    def test_capabilities_input_dependent(self) -> None:
        t = _RichTool()
        ls = t.capabilities({"command": "ls -la"})
        assert ls.read_only is True
        assert ls.destructive is False

        rm = t.capabilities({"command": "rm -rf /tmp/x"})
        assert rm.destructive is True
        assert rm.read_only is False

        curl = t.capabilities({"command": "curl http://x"})
        assert curl.network_egress is True

    def test_permission_ask_in_plan_mode(self) -> None:
        t = _RichTool()
        ctx = ToolContext(permission_mode="plan")
        d = asyncio.run(t.check_permissions({"command": "rm -rf /"}, ctx))
        assert d.behavior == "ask"
        assert d.reason is not None

    def test_permission_allow_in_default_mode(self) -> None:
        t = _RichTool()
        d = asyncio.run(t.check_permissions({"command": "rm -rf /"}, ToolContext()))
        assert d.behavior == "allow"

    def test_permission_matcher_input_pattern(self) -> None:
        t = _RichTool()
        match = asyncio.run(t.prepare_permission_matcher({"command": "git status"}))
        assert match("Rich") is True
        assert match("Rich(git *)") is True
        assert match("Rich(rm *)") is False
        assert match("Other") is False

    def test_user_facing_name_and_activity(self) -> None:
        t = _RichTool()
        assert t.user_facing_name({"command": "ls"}).startswith("Rich: ")
        assert "ls" in (t.activity_description({"command": "ls"}) or "")

    def test_aliases_attribute(self) -> None:
        t = _RichTool()
        assert "rich" in t.aliases
        assert "rich-tool" in t.aliases


# ─────────────────────────────────────────────────────────────────
# build_tool() factory
# ─────────────────────────────────────────────────────────────────


class TestBuildToolFactory:
    def test_minimal_factory(self) -> None:
        async def _exec(inp, ctx):
            return ToolResult(content=f"echo:{inp.get('msg','')}")

        tool = build_tool(
            name="echo",
            description="Echo back the message.",
            input_schema={
                "type": "object",
                "properties": {"msg": {"type": "string"}},
                "required": ["msg"],
            },
            execute=_exec,
        )

        assert isinstance(tool, Tool)
        assert tool.name == "echo"
        assert tool.description == "Echo back the message."
        # Defaults
        assert tool.capabilities({"msg": "hi"}) == ToolCapabilities()
        assert tool.is_enabled() is True

        # Execution works
        out = asyncio.run(tool.execute({"msg": "hi"}, ToolContext()))
        assert out.content == "echo:hi"

    def test_factory_with_capabilities(self) -> None:
        async def _exec(inp, ctx):
            return ToolResult(content="ok")

        caps = ToolCapabilities(
            concurrency_safe=True,
            read_only=True,
            max_result_chars=1_000_000,
        )
        tool = build_tool(
            name="reader",
            description="desc",
            input_schema={"type": "object"},
            execute=_exec,
            capabilities=caps,
        )
        # capabilities() returns the passed instance
        got = tool.capabilities({})
        assert got.concurrency_safe is True
        assert got.read_only is True
        assert got.max_result_chars == 1_000_000

    def test_factory_custom_permission_hook(self) -> None:
        calls = {"n": 0}

        async def _exec(inp, ctx):
            return ToolResult(content="ok")

        async def _check(inp, ctx):
            calls["n"] += 1
            return PermissionDecision(behavior="deny", reason="blocked")

        tool = build_tool(
            name="forbidden",
            description="d",
            input_schema={"type": "object"},
            execute=_exec,
            check_permissions=_check,
        )

        d = asyncio.run(tool.check_permissions({}, ToolContext()))
        assert d.behavior == "deny"
        assert d.reason == "blocked"
        assert calls["n"] == 1

    def test_factory_aliases(self) -> None:
        async def _exec(inp, ctx):
            return ToolResult(content="")

        tool = build_tool(
            name="primary",
            description="d",
            input_schema={"type": "object"},
            execute=_exec,
            aliases=("alt1", "alt2"),
        )
        assert tool.aliases == ("alt1", "alt2")

    def test_factory_lifecycle_hooks(self) -> None:
        log: list[str] = []

        async def _exec(inp, ctx):
            return ToolResult(content="done")

        async def _enter(inp, ctx):
            log.append("enter")

        async def _exit(result, ctx):
            log.append(f"exit:{result.content}")

        async def _error(err, ctx):
            log.append(f"error:{err}")

        tool = build_tool(
            name="lifecycle",
            description="d",
            input_schema={"type": "object"},
            execute=_exec,
            on_enter=_enter,
            on_exit=_exit,
            on_error=_error,
        )

        async def _run_happy() -> None:
            ctx = ToolContext()
            await tool.on_enter({}, ctx)
            result = await tool.execute({}, ctx)
            await tool.on_exit(result, ctx)

        asyncio.run(_run_happy())
        assert log == ["enter", "exit:done"]
        log.clear()

        async def _run_err() -> None:
            ctx = ToolContext()
            try:
                raise RuntimeError("boom")
            except RuntimeError as e:
                await tool.on_error(e, ctx)

        asyncio.run(_run_err())
        assert log == ["error:boom"]

    def test_factory_is_enabled_toggle(self) -> None:
        state = {"live": True}

        async def _exec(inp, ctx):
            return ToolResult(content="")

        tool = build_tool(
            name="toggle",
            description="d",
            input_schema={"type": "object"},
            execute=_exec,
            is_enabled=lambda: state["live"],
        )
        assert tool.is_enabled() is True
        state["live"] = False
        assert tool.is_enabled() is False

    def test_factory_api_format(self) -> None:
        async def _exec(inp, ctx):
            return ToolResult(content="")

        tool = build_tool(
            name="api",
            description="An API tool.",
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
            execute=_exec,
        )
        api = tool.to_api_format()
        assert api == {
            "name": "api",
            "description": "An API tool.",
            "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
        }
