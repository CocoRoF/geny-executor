"""Built-in fs/shell tools route through the container when ToolContext.sandbox
is set (SDK-path sandboxing). A fake ``docker exec`` simulates a tiny in-container
filesystem so we can assert read/write/edit/bash round-trips — no docker needed.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest

import geny_executor.tools._sandbox as sbmod
from geny_executor.tools.base import ToolContext
from geny_executor.tools.built_in.bash_tool import BashTool
from geny_executor.tools.built_in.edit_tool import EditTool
from geny_executor.tools.built_in.read_tool import ReadTool
from geny_executor.tools.built_in.write_tool import WriteTool


class _FakeSandbox:
    container_name = "gapt-ws-t"

    async def ensure(self) -> None:
        return None


@pytest.fixture
def fake_container_fs(monkeypatch):
    """Patch the docker-exec spawn with an in-memory container filesystem."""
    fs: dict[str, bytes] = {}

    def fake_exec(launcher, *argv, **kwargs):
        a = list(argv)
        ci = a.index("gapt-ws-t")
        cmd = a[ci + 1 :]

        class _Proc:
            returncode = 0

            async def communicate(self, input=None):
                if cmd[0] == "cat":  # cat -- <path>
                    path = cmd[-1]
                    if path in fs:
                        return fs[path], b""
                    self.returncode = 1
                    return b"", b"cat: no such file or directory"
                if cmd[0] == "sh":  # sh -c <script> sh <path>  (write)
                    path = cmd[-1]
                    fs[path] = input or b""
                    return b"", b""
                if cmd[0] == "bash":  # bash -lc <command>
                    return b"hello-from-container\n", b""
                return b"", b""

        return _Proc()

    async def _fake_create(*args, **kwargs):
        return fake_exec(*args, **kwargs)

    monkeypatch.setattr(sbmod.asyncio, "create_subprocess_exec", _fake_create)
    return fs


def _ctx() -> ToolContext:
    return ToolContext(working_dir="/workspace", sandbox=_FakeSandbox())


@pytest.mark.asyncio
async def test_write_then_read_roundtrip(fake_container_fs) -> None:
    ctx = _ctx()
    w = await WriteTool().execute({"file_path": "a/b.txt", "content": "hi there"}, ctx)
    assert not w.is_error
    assert fake_container_fs["/workspace/a/b.txt"] == b"hi there"

    r = await ReadTool().execute({"file_path": "a/b.txt"}, ctx)
    assert not r.is_error
    assert "hi there" in r.content  # line-numbered output


@pytest.mark.asyncio
async def test_read_missing_file(fake_container_fs) -> None:
    r = await ReadTool().execute({"file_path": "nope.txt"}, _ctx())
    assert r.is_error
    assert "not found" in r.content.lower()


@pytest.mark.asyncio
async def test_edit_in_sandbox(fake_container_fs) -> None:
    ctx = _ctx()
    await WriteTool().execute({"file_path": "c.txt", "content": "foo bar foo"}, ctx)
    e = await EditTool().execute(
        {"file_path": "c.txt", "old_string": "bar", "new_string": "BAZ"}, ctx
    )
    assert not e.is_error
    assert fake_container_fs["/workspace/c.txt"] == b"foo BAZ foo"


@pytest.mark.asyncio
async def test_bash_in_sandbox(fake_container_fs) -> None:
    r = await BashTool().execute({"command": "echo hi"}, _ctx())
    assert not r.is_error
    assert "hello-from-container" in r.content
    assert r.metadata.get("sandboxed") is True


@pytest.mark.asyncio
async def test_path_escape_blocked(fake_container_fs) -> None:
    r = await WriteTool().execute(
        {"file_path": "../../etc/passwd", "content": "x"}, _ctx()
    )
    assert r.is_error
    assert "outside" in r.content.lower()


def test_container_path_resolution() -> None:
    assert sbmod.container_path("a/b.txt", "/workspace") == "/workspace/a/b.txt"
    assert sbmod.container_path("/workspace/x", "/workspace") == "/workspace/x"
    with pytest.raises(PermissionError):
        sbmod.container_path("../escape", "/workspace")
