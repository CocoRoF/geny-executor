"""env(action="forge_tool") — author a new sandboxed tool live this session.

The forged tool is a SandboxExecTool bound to the session's sandbox; we don't
execute it here (that needs a real sandbox) — we assert it is built + registered
correctly and the guards fire.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from geny_executor.core.environment_control import PipelineEnvironment
from geny_executor.tools.built_in.env_tools import EnvTool
from geny_executor.tools.built_in.sandbox_exec_tool import SandboxExecTool


class _Registry:
    def __init__(self) -> None:
        self._d: dict = {}

    def get(self, name):
        return self._d.get(name)

    def register(self, tool):
        self._d[tool.name] = tool

    def unregister(self, name):
        self._d.pop(name, None)


class _SkillRegistry:
    def __init__(self) -> None:
        self._d: dict = {}

    def get(self, sid):
        return self._d.get(sid)

    def register(self, skill):
        self._d[skill.id] = skill

    def unregister(self, sid):
        self._d.pop(sid, None)


def _env(with_sandbox: bool = True) -> tuple[PipelineEnvironment, _Registry]:
    reg = _Registry()
    ctx = SimpleNamespace(sandbox=object()) if with_sandbox else SimpleNamespace(sandbox=None)
    return PipelineEnvironment(registry=reg, tool_context=ctx), reg


def test_forge_tool_registers_a_sandbox_exec_tool() -> None:
    env, reg = _env()
    ok, msg = env.forge_tool(
        name="wordcount",
        description="count words",
        entrypoint="tools/wordcount/main.py",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
    )
    assert ok, msg
    tool = reg.get("wordcount")
    assert isinstance(tool, SandboxExecTool)
    spec = tool.to_dict()
    assert spec["name"] == "wordcount"
    assert spec["entrypoint"] == "tools/wordcount/main.py"
    assert spec["runtime"] == "python3"
    assert tool.input_schema["properties"]["text"]["type"] == "string"


def test_forge_tool_needs_a_sandbox() -> None:
    env, reg = _env(with_sandbox=False)
    ok, msg = env.forge_tool(name="x", entrypoint="x.py")
    assert not ok
    assert "sandbox" in msg.lower()
    assert reg.get("x") is None


def test_forge_tool_requires_name_and_entrypoint() -> None:
    env, _ = _env()
    assert env.forge_tool(name="", entrypoint="x.py")[0] is False
    assert env.forge_tool(name="x", entrypoint="")[0] is False


def test_forge_tool_refuses_to_clobber_active_name() -> None:
    env, reg = _env()
    assert env.forge_tool(name="dup", entrypoint="a.py")[0] is True
    ok, msg = env.forge_tool(name="dup", entrypoint="b.py")
    assert not ok and "already active" in msg
    # original kept
    assert reg.get("dup").to_dict()["entrypoint"] == "a.py"


def test_forge_tool_carries_runtime_and_flags() -> None:
    env, reg = _env()
    ok, _ = env.forge_tool(
        name="slug", entrypoint="tools/slug/main.js", runtime="node",
        timeout_s=30, network_egress=True, read_only=True, workdir="/workspace",
    )
    assert ok
    spec = reg.get("slug").to_dict()
    assert spec["runtime"] == "node" and spec["network_egress"] is True and spec["read_only"] is True
    assert spec["timeout_s"] == 30.0


def test_save_pack_gathers_forged_tools_and_skills() -> None:
    reg = _Registry()
    captured = {}

    async def persist(payload):
        captured.update(payload)
        return {"pack_id": "pk_123"}

    env = PipelineEnvironment(
        registry=reg,
        skill_registry=_SkillRegistry(),
        tool_context=SimpleNamespace(sandbox=SimpleNamespace(workspace_id="W1")),
        pack_persistence=persist,
    )
    env.forge_tool(name="a", entrypoint="tools/a/main.py")
    env.forge_tool(name="b", entrypoint="tools/b/main.js", runtime="node")
    env.create_skill("howto", "how to use a+b", "# body")

    ok, msg = asyncio.run(env.save_pack("mypack", description="two tools"))
    assert ok, msg
    assert "pk_123" in msg
    assert {t["name"] for t in captured["tools"]} == {"a", "b"}
    assert captured["skills"][0]["id"] == "howto"
    assert getattr(captured["sandbox"], "workspace_id") == "W1"


def test_save_pack_needs_a_forged_tool_and_callback() -> None:
    reg = _Registry()
    # no callback
    env = PipelineEnvironment(registry=reg, tool_context=SimpleNamespace(sandbox=object()))
    assert asyncio.run(env.save_pack("p"))[0] is False  # no pack_persistence

    async def persist(_):
        return {"pack_id": "x"}

    env2 = PipelineEnvironment(
        registry=reg, tool_context=SimpleNamespace(sandbox=object()), pack_persistence=persist
    )
    ok, msg = asyncio.run(env2.save_pack("p"))  # no forged tools
    assert not ok and "forge_tool" in msg


def test_save_pack_filters_to_named_tools() -> None:
    reg = _Registry()

    async def persist(payload):
        return {"pack_id": "pk", "got": [t["name"] for t in payload["tools"]]}

    env = PipelineEnvironment(
        registry=reg, tool_context=SimpleNamespace(sandbox=object()), pack_persistence=persist
    )
    env.forge_tool(name="keep", entrypoint="a.py")
    env.forge_tool(name="drop", entrypoint="b.py")
    ok, msg = asyncio.run(env.save_pack("p", tools=["keep"]))
    assert ok and "1 tool(s)" in msg


def test_forge_tool_via_env_dispatcher() -> None:
    env, reg = _env()
    tool = EnvTool()
    res = asyncio.run(
        tool.execute(
            {"action": "forge_tool", "args": {"name": "echo", "entrypoint": "e.py"}},
            SimpleNamespace(environment=env),
        )
    )
    assert not res.is_error, res.content
    assert isinstance(reg.get("echo"), SandboxExecTool)
