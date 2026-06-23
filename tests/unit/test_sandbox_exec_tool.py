"""SandboxExecTool — runs an authored script inside a sandbox (docker exec).

The sandbox is faked: a SandboxHandle whose ``container_name``/``ensure`` are
inert, and we monkeypatch ``sandbox_exec`` to simulate the container without
needing Docker. This exercises the tool↔script JSON contract + error paths.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Tuple

import pytest

from geny_executor.tools.base import ToolContext
from geny_executor.tools.built_in.sandbox_exec_tool import SandboxExecTool


class _FakeSandbox:
    container_name = "gapt-ws-tool-test"

    def __init__(self) -> None:
        self.ensured = 0

    async def ensure(self) -> None:
        self.ensured += 1


def _spec(**over: Any) -> Dict[str, Any]:
    base = {
        "name": "echo_upper",
        "description": "uppercase the 'text' field",
        "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}},
        "runtime": "python3",
        "entrypoint": "tools/echo_upper/main.py",
        "timeout_s": 5.0,
    }
    base.update(over)
    return base


def _patch_exec(monkeypatch, fn) -> List[Tuple[Any, ...]]:
    """Replace sandbox_exec in the tool module; record (argv, cwd, input)."""
    calls: List[Tuple[Any, ...]] = []

    async def _fake(sandbox, argv, *, cwd, input_bytes=None, env=None, timeout_s=120.0, launcher="docker"):
        calls.append((argv, cwd, input_bytes, timeout_s))
        return fn(argv, input_bytes)

    monkeypatch.setattr(
        "geny_executor.tools.built_in.sandbox_exec_tool.sandbox_exec", _fake
    )
    return calls


@pytest.mark.asyncio
async def test_success_passes_json_stdin_and_returns_stdout(monkeypatch) -> None:
    def run(argv, input_bytes):
        # argv = [runtime, entrypoint]; input is JSON on stdin.
        assert argv == ["python3", "tools/echo_upper/main.py"]
        assert b'"hello"' in input_bytes
        return (0, b'{"result": "HELLO"}', b"")

    calls = _patch_exec(monkeypatch, run)
    tool = SandboxExecTool.from_dict(_spec(), sandbox=_FakeSandbox())
    res = await tool.execute({"text": "hello"}, ToolContext())
    assert not res.is_error
    assert res.content == '{"result": "HELLO"}'
    assert calls and calls[0][1] == "/workspace"           # cwd
    assert calls[0][3] == 5.0                                # timeout passed through


@pytest.mark.asyncio
async def test_no_sandbox_is_error() -> None:
    tool = SandboxExecTool.from_dict(_spec())  # no sandbox attached
    res = await tool.execute({"text": "x"}, ToolContext())  # ctx.sandbox unset
    assert res.is_error and "no sandbox" in res.content.lower()


@pytest.mark.asyncio
async def test_uses_context_sandbox_when_tool_has_none(monkeypatch) -> None:
    _patch_exec(monkeypatch, lambda a, i: (0, b'{"ok": true}', b""))
    tool = SandboxExecTool.from_dict(_spec())  # no own sandbox
    res = await tool.execute({}, ToolContext(sandbox=_FakeSandbox()))
    assert not res.is_error and res.content == '{"ok": true}'


@pytest.mark.asyncio
async def test_nonzero_exit_is_error_with_stderr(monkeypatch) -> None:
    _patch_exec(monkeypatch, lambda a, i: (2, b"", b"Traceback: boom"))
    tool = SandboxExecTool.from_dict(_spec(), sandbox=_FakeSandbox())
    res = await tool.execute({}, ToolContext())
    assert res.is_error
    assert "exit 2" in res.content and "boom" in res.content
    assert res.metadata["exit_code"] == 2


@pytest.mark.asyncio
async def test_json_error_payload_is_error(monkeypatch) -> None:
    _patch_exec(monkeypatch, lambda a, i: (0, b'{"error": "bad input"}', b""))
    tool = SandboxExecTool.from_dict(_spec(), sandbox=_FakeSandbox())
    res = await tool.execute({}, ToolContext())
    assert res.is_error and res.content == "bad input"


@pytest.mark.asyncio
async def test_timeout_is_error(monkeypatch) -> None:
    async def _boom(*a, **k):
        raise asyncio.TimeoutError()

    monkeypatch.setattr(
        "geny_executor.tools.built_in.sandbox_exec_tool.sandbox_exec", _boom
    )
    tool = SandboxExecTool.from_dict(_spec(timeout_s=3.0), sandbox=_FakeSandbox())
    res = await tool.execute({}, ToolContext())
    assert res.is_error and "timed out" in res.content and res.metadata["timeout"]


@pytest.mark.asyncio
async def test_plain_text_stdout_passed_through(monkeypatch) -> None:
    _patch_exec(monkeypatch, lambda a, i: (0, b"not json, just text\n", b""))
    tool = SandboxExecTool.from_dict(_spec(), sandbox=_FakeSandbox())
    res = await tool.execute({}, ToolContext())
    assert not res.is_error and res.content == "not json, just text"


def test_spec_roundtrip() -> None:
    spec = _spec(runtime="node", argv=["--flag"], network_egress=True, read_only=True)
    tool = SandboxExecTool.from_dict(spec)
    out = tool.to_dict()
    assert out["runtime"] == "node"
    assert out["argv"] == ["--flag"]
    assert out["network_egress"] is True and out["read_only"] is True
    # round-trips to an equivalent tool
    again = SandboxExecTool.from_dict(out)
    assert again.to_dict() == out
    assert tool.name == "echo_upper"


def test_capabilities_reflect_spec() -> None:
    ro = SandboxExecTool.from_dict(_spec(read_only=True, network_egress=True))
    caps = ro.capabilities({})
    assert caps.read_only and caps.concurrency_safe and caps.network_egress
    rw = SandboxExecTool.from_dict(_spec())
    assert not rw.capabilities({}).concurrency_safe       # fail-closed default


@pytest.mark.asyncio
async def test_exposed_from_public_packages() -> None:
    # built_in package export + container-exec primitives on tools package.
    from geny_executor.tools.built_in import SandboxExecTool as A
    from geny_executor.tools import sandbox_exec, SandboxExecError  # noqa: F401
    from geny_executor.tools.built_in import BUILT_IN_TOOL_CLASSES

    assert A is SandboxExecTool
    # NOT a manifest-activated built-in.
    assert "SandboxExecTool" not in BUILT_IN_TOOL_CLASSES
    assert not any(v is SandboxExecTool for v in BUILT_IN_TOOL_CLASSES.values())
