"""Run a tool's fs/shell primitive inside a sandbox container (``docker exec``).

Shared by the built-in fs/shell tools so an SDK-provider agent
(anthropic/openai/google/vllm) gets the same container isolation the
``claude_code_cli`` path already has: when ``ToolContext.sandbox`` is set, the
tool routes its I/O here instead of touching the host filesystem.

The ``sandbox`` is any object with ``container_name: str`` + async
``ensure()`` (the executor's ``SandboxHandle`` Protocol; GAPT's
``WorkspaceSandbox`` and Geny's ``GaptSandboxHandle`` both satisfy it).
"""

from __future__ import annotations

import asyncio
import os
import posixpath
import sys
from typing import Any, Mapping, Optional, Sequence, Tuple


class SandboxExecError(RuntimeError):
    """A ``docker exec`` into the sandbox failed to spawn."""


def container_path(file_path: str, workdir: str) -> str:
    """Resolve ``file_path`` to an absolute path *inside* the container,
    rooted at ``workdir`` (default ``/workspace``), refusing to escape it.

    Relative paths join onto ``workdir``; absolute paths are taken as-is but
    must stay within ``workdir``. Raises ``PermissionError`` on escape."""
    base = workdir or "/workspace"
    base = "/" + base.strip("/") if base != "/" else "/"
    raw = file_path or "."
    joined = raw if posixpath.isabs(raw) else posixpath.join(base, raw)
    norm = posixpath.normpath(joined)
    if norm != base and not norm.startswith(base.rstrip("/") + "/"):
        raise PermissionError(
            f"path {file_path!r} resolves outside the sandbox workdir {base!r}"
        )
    return norm


async def sandbox_exec(
    sandbox: Any,
    argv: Sequence[str],
    *,
    cwd: str = "/workspace",
    input_bytes: Optional[bytes] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout_s: float = 120.0,
    launcher: str = "docker",
) -> Tuple[int, bytes, bytes]:
    """``<launcher> exec -i -w <cwd> --env … <container> <argv>``.

    Returns ``(returncode, stdout, stderr)``. Brings the container live first
    via ``sandbox.ensure()`` (idempotent). Raises :class:`SandboxExecError` if
    the launcher cannot be spawned, :class:`asyncio.TimeoutError` on timeout."""
    try:
        await sandbox.ensure()
    except Exception:  # pragma: no cover - defensive; exec surfaces real errors
        pass

    exec_argv = ["exec", "-i", "-w", cwd]
    for k, v in dict(env or {}).items():
        exec_argv += ["--env", f"{k}={v}"]
    exec_argv += [sandbox.container_name, *list(argv)]

    kwargs: dict[str, Any] = dict(
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ.copy(),  # the launcher needs host PATH/DOCKER_HOST
    )
    if sys.platform != "win32":
        kwargs["start_new_session"] = True
    try:
        proc = await asyncio.create_subprocess_exec(launcher, *exec_argv, **kwargs)
    except OSError as e:
        raise SandboxExecError(f"failed to spawn {launcher!r}: {e}") from e

    try:
        out, err = await asyncio.wait_for(
            proc.communicate(input=input_bytes), timeout=timeout_s
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
            await proc.wait()
        except Exception:
            pass
        raise
    return (proc.returncode if proc.returncode is not None else -1), out, err


async def sb_read_bytes(sandbox: Any, path: str, *, workdir: str) -> bytes:
    """Read a file inside the container. Raises FileNotFoundError / OSError."""
    cpath = container_path(path, workdir)
    rc, out, err = await sandbox_exec(sandbox, ["cat", "--", cpath], cwd=workdir)
    if rc != 0:
        msg = err.decode("utf-8", "replace").strip()
        if "no such file" in msg.lower():
            raise FileNotFoundError(path)
        raise OSError(msg or f"cat exited {rc}")
    return out


async def sb_write_bytes(sandbox: Any, path: str, data: bytes, *, workdir: str) -> int:
    """Create/overwrite a file inside the container (parent dirs created).
    Returns bytes written."""
    cpath = container_path(path, workdir)
    # mkdir -p the parent, then write stdin to the file. Single sh -c so the
    # path is quoted once; data flows over stdin (binary-safe, no escaping).
    script = f'mkdir -p "$(dirname "$1")" && cat > "$1"'
    rc, _out, err = await sandbox_exec(
        sandbox,
        ["sh", "-c", script, "sh", cpath],
        cwd=workdir,
        input_bytes=data,
    )
    if rc != 0:
        raise OSError(err.decode("utf-8", "replace").strip() or f"write exited {rc}")
    return len(data)


async def sb_run(
    sandbox: Any,
    command: str,
    *,
    workdir: str,
    env: Optional[Mapping[str, str]] = None,
    timeout_s: float = 120.0,
) -> Tuple[int, str, str]:
    """Run a shell command inside the container; returns (rc, stdout, stderr)."""
    rc, out, err = await sandbox_exec(
        sandbox,
        ["bash", "-lc", command],
        cwd=workdir,
        env=env,
        timeout_s=timeout_s,
    )
    return rc, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")
