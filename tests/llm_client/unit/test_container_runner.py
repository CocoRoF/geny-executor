"""Tests for the container sandbox runner (L1 sandbox-execution primitive).

``ContainerCLIRunner`` generalises GAPT's former ``SandboxedCLIProcessRunner``:
it runs the agent CLI inside a sandbox container via ``<launcher> exec``. These
tests are host-independent — they never require ``docker`` or ``claude`` to be
installed (the launcher check uses ``sh``, and the spawn is intercepted).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import pytest

import geny_executor.llm_client._cli_runtime as rt
from geny_executor.llm_client import (
    ClaudeCodeCLIClient,
    ContainerCLIRunner,
    SandboxHandle,
    build_container_cli_client,
)
from geny_executor.llm_client._cli_runtime import CLIBinaryNotFound


class FakeSandbox:
    """Satisfies the :class:`SandboxHandle` Protocol."""

    def __init__(self, name: str) -> None:
        self.container_name = name
        self.ensured = False

    async def ensure(self) -> None:
        self.ensured = True


def test_fake_sandbox_satisfies_protocol() -> None:
    assert isinstance(FakeSandbox("c"), SandboxHandle)


def test_requires_sandbox() -> None:
    with pytest.raises(ValueError):
        ContainerCLIRunner(binary="", sandbox=None, launcher="sh")


def test_bad_launcher_raises() -> None:
    with pytest.raises(CLIBinaryNotFound):
        ContainerCLIRunner(
            binary="",
            sandbox=FakeSandbox("c"),
            launcher="definitely-not-a-launcher-xyz",
        )


def test_host_binary_not_required() -> None:
    # The agent binary lives in the container; an empty/absent host binary
    # must NOT trip the parent's host-binary existence check.
    runner = ContainerCLIRunner(binary="", sandbox=FakeSandbox("c"), launcher="sh")
    assert runner.workdir == "/workspace"
    assert runner.container_binary == "claude"


@pytest.mark.asyncio
async def test_spawn_builds_exec_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class _Proc:
        returncode = 0

    async def fake_exec(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _Proc()

    monkeypatch.setattr(rt.asyncio, "create_subprocess_exec", fake_exec)

    sandbox = FakeSandbox("gapt-ws-abc")
    runner = ContainerCLIRunner(
        binary="",
        sandbox=sandbox,
        launcher="sh",
        env_extras={"ANTHROPIC_API_KEY": "sek"},
    )
    await runner._spawn(["-p", "hi"])

    # ensure() ran before the spawn.
    assert sandbox.ensured is True

    args = list(captured["args"])
    assert args[0] == "sh"  # launcher first
    rest = args[1:]
    assert rest[:4] == ["exec", "-i", "-w", "/workspace"]
    assert "--env" in rest
    assert "ANTHROPIC_API_KEY=sek" in rest
    # container name, then in-container binary, then the agent argv.
    ci = rest.index("gapt-ws-abc")
    assert rest[ci + 1] == "claude"
    assert rest[ci + 2 :] == ["-p", "hi"]
    # The launcher needs host env; the child env is via --env flags.
    assert captured["kwargs"]["cwd"] is None


@pytest.mark.asyncio
async def test_custom_workdir_and_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class _Proc:
        returncode = 0

    async def fake_exec(*args, **kwargs):
        captured["args"] = args
        return _Proc()

    monkeypatch.setattr(rt.asyncio, "create_subprocess_exec", fake_exec)

    runner = ContainerCLIRunner(
        binary="",
        sandbox=FakeSandbox("box"),
        launcher="sh",
        workdir="/srv/app",
        container_binary="codex",
    )
    await runner._spawn(["--version"])
    rest = list(captured["args"])[1:]
    assert rest[:4] == ["exec", "-i", "-w", "/srv/app"]
    ci = rest.index("box")
    assert rest[ci + 1] == "codex"


def test_build_container_cli_client_sets_factory() -> None:
    sandbox = FakeSandbox("gapt-ws-1")
    client = build_container_cli_client(sandbox=sandbox, launcher="sh", api_key="k")
    runner = client._make_runner()
    assert isinstance(runner, ContainerCLIRunner)
    assert runner.sandbox is sandbox
    assert runner.launcher == "sh"


def test_build_rejects_runner_factory() -> None:
    with pytest.raises(TypeError):
        build_container_cli_client(
            sandbox=FakeSandbox("c"),
            launcher="sh",
            runner_factory=lambda **k: None,  # type: ignore[arg-type]
        )


def test_make_runner_without_factory_still_requires_host_binary() -> None:
    # Backward-compat: the default in-process runner still needs a real host
    # binary; only the factory path is exempt.
    client = ClaudeCodeCLIClient(api_key="k", binary_path="/nonexistent/claude-xyz")
    with pytest.raises(CLIBinaryNotFound):
        client._make_runner()
