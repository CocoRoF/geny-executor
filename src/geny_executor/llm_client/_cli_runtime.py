"""Async subprocess primitives shared by CLI-backed LLM clients.

This module is the *only* place where ``asyncio.create_subprocess_exec`` is
called inside ``llm_client/``. ``ClaudeCodeCLIClient`` (Phase B) drives its
work through these helpers. (The Phase-C ``CopilotCLIClient`` was removed
in 2.0.6 — see the commit message for the structural-incompatibility
rationale.)

Design rules
------------

* ``shell=False`` always. argv lists only. No string interpolation.
* ``start_new_session=True`` on POSIX so we can ``killpg`` the whole tree on
  cancellation / timeout.
* Environment is scrubbed by default: only the host whitelist + caller-supplied
  ``env_extras`` are visible to the child. Prevents leaking unrelated host env
  (random tokens, profile metadata, etc.) into the CLI process.
* ``stream(...)`` is line-buffered for stream-json. ``run_oneshot(...)`` returns
  the full stdout/stderr blob.
* Timeout is enforced by the wrapper, *not* the OS. We send SIGTERM, wait
  ``kill_grace_s``, then SIGKILL the process group.

Exceptions
----------

* ``CLIBinaryNotFound``: binary path doesn't exist or isn't executable.
* ``CLIAuthFailed``: stderr / exit code indicates auth issues (heuristic;
  callers may upgrade via ``classify_cli_failure``).
* ``CLITimeout``: wall-clock exceeded.
* ``CLIProtocolError``: malformed stream-json / unexpected exit on a streaming
  request.

The CLI clients wrap these into ``APIError`` with the matching
``ErrorCategory.CLI_*`` value.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Iterable,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CLIBinaryNotFound(Exception):
    """Configured CLI binary path does not exist or is not executable."""


class CLIAuthFailed(Exception):
    """CLI subprocess reported an authentication / authorisation failure."""


class CLITimeout(Exception):
    """CLI subprocess exceeded the configured timeout."""


class CLIProtocolError(Exception):
    """CLI subprocess produced malformed or unexpected output."""


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


class CLIResult(NamedTuple):
    """One-shot CLI invocation result."""

    returncode: int
    stdout: bytes
    stderr: bytes
    duration_ms: int


# ---------------------------------------------------------------------------
# Defaults / constants
# ---------------------------------------------------------------------------


#: Environment variables that are always passed through. CLI tools commonly
#: need ``HOME`` (to find their own config), ``PATH`` (for spawning helpers),
#: locale info, and a sensible ``TERM``.
DEFAULT_ENV_WHITELIST: frozenset[str] = frozenset(
    {
        "HOME",
        "PATH",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TERM",
        "TMPDIR",
        "TZ",
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def detect_binary(name: str, override: Optional[str] = None) -> Optional[str]:
    """Resolve a CLI binary path.

    Order:
    1. Explicit ``override`` argument (if executable).
    2. ``shutil.which(name)``.

    Returns ``None`` if neither path exists / is executable.
    """
    if override:
        p = Path(override).expanduser()
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
        return None
    found = shutil.which(name)
    return found


def scrub_env(
    parent: Mapping[str, str],
    *,
    whitelist: Iterable[str] = DEFAULT_ENV_WHITELIST,
    extras: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """Build a child env containing only whitelisted parent vars + extras."""
    allowed = set(whitelist)
    out: dict[str, str] = {k: v for k, v in parent.items() if k in allowed}
    if extras:
        out.update(extras)
    return out


def parse_stream_json_line(line: bytes) -> Optional[dict[str, Any]]:
    """Parse one stream-json line.

    Returns:
    - ``None`` for empty / comment lines (caller skips).
    - ``dict`` for valid JSON.
    - ``{"__malformed__": "<raw>"}`` for unparseable content — caller decides
      whether to raise ``CLIProtocolError`` or log and continue.
    """
    s = line.decode("utf-8", errors="replace").strip()
    if not s:
        return None
    if s.startswith("#"):
        return None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return {"__malformed__": s}
    if isinstance(obj, dict):
        return obj
    return {"__malformed__": s}


# ---------------------------------------------------------------------------
# Process runner
# ---------------------------------------------------------------------------


@dataclass
class CLIProcessRunner:
    """Async wrapper for one CLI invocation.

    A runner is single-use: each ``run_oneshot`` / ``stream`` call spawns
    its own process. Concurrent calls on the same runner are not supported
    (callers must construct one runner per invocation if they need
    concurrency).
    """

    binary: str
    env_whitelist: frozenset[str] = DEFAULT_ENV_WHITELIST
    env_extras: Optional[Mapping[str, str]] = None
    cwd: Optional[str] = None
    timeout_s: float = 300.0
    kill_grace_s: float = 2.0

    def __post_init__(self) -> None:
        if not self.binary:
            raise CLIBinaryNotFound("CLI binary path is empty")
        p = Path(self.binary)
        if not p.exists():
            raise CLIBinaryNotFound(f"binary not found: {self.binary}")
        if not os.access(p, os.X_OK):
            raise CLIBinaryNotFound(f"binary not executable: {self.binary}")

    # ------------------------------------------------------------------ run
    async def run_oneshot(
        self,
        argv: Sequence[str],
        *,
        stdin: Optional[bytes] = None,
    ) -> CLIResult:
        """Spawn, drain stdout/stderr, wait for exit. Single-shot."""
        proc, t0 = await self._spawn(argv)
        try:
            stdout_bytes, stderr_bytes = await self._communicate(proc, stdin)
        except asyncio.TimeoutError as e:
            await self._kill_tree(proc)
            raise CLITimeout(f"CLI {self.binary!r} exceeded {self.timeout_s:.1f}s") from e
        duration_ms = int((time.monotonic() - t0) * 1000)
        rc = proc.returncode if proc.returncode is not None else -1
        return CLIResult(
            returncode=rc, stdout=stdout_bytes, stderr=stderr_bytes, duration_ms=duration_ms
        )

    # --------------------------------------------------------------- stream
    async def stream(
        self,
        argv: Sequence[str],
        *,
        stdin_iter: Optional[AsyncIterator[bytes]] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Spawn and yield stdout *lines* as they arrive.

        Newline-delimited (i.e. one stream-json line per yield). Caller is
        responsible for invoking ``parse_stream_json_line`` on each line.

        Stderr is captured but not yielded. On non-zero exit, stderr is
        attached to the raised exception.

        Consumer disconnect (audit 2026-06-09 §3.7): both reference hosts
        are SSE servers, so a client disconnect closes this generator
        mid-output (``GeneratorExit`` at the ``yield``). The ``finally``
        block below therefore owns the full teardown: cancel + reap the
        stdin-drain side task, cancel the stderr collector, and — when
        the child has not been reaped yet — run the SIGTERM → grace →
        SIGKILL ladder so no real CLI process is left orphaned on the
        box. Normal completion and the CLITimeout path reach the
        ``finally`` with ``proc.returncode`` already set, making the
        kill a no-op (no double-kill).
        """
        proc, t0 = await self._spawn(argv)

        # If caller has stdin_iter, drive it in a side task.
        stdin_task: Optional[asyncio.Task[None]] = None
        if stdin_iter is not None and proc.stdin is not None:
            stdin_task = asyncio.create_task(_drain_stdin(proc.stdin, stdin_iter))
        elif proc.stdin is not None:
            proc.stdin.close()

        # Side-collect stderr.
        stderr_buf: list[bytes] = []
        stderr_task = asyncio.create_task(_collect_stderr(proc.stderr, stderr_buf))

        try:
            async for line in _aiter_lines(proc.stdout, timeout_s=self.timeout_s, start_t=t0):
                yield line
            rc = await proc.wait()
        except CLITimeout:
            await self._kill_tree(proc)
            raise
        finally:
            stderr_task.cancel()
            if stdin_task is not None:
                stdin_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stdin_task
            if proc.returncode is None:
                # Generator exited before the child was reaped — the
                # consumer-disconnect path. Kill the process group.
                await self._kill_tree(proc)

        if rc != 0:
            tail = b"".join(stderr_buf).decode("utf-8", errors="replace")
            raise CLIProtocolError(
                f"CLI {self.binary!r} exited with code {rc}: {tail.strip()[:400]}"
            )

    # ---------------------------------------------------------------- spawn
    async def _spawn(self, argv: Sequence[str]) -> tuple[asyncio.subprocess.Process, float]:
        env = scrub_env(os.environ, whitelist=self.env_whitelist, extras=self.env_extras)
        kwargs: dict[str, Any] = dict(
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=self.cwd,
        )
        if sys.platform != "win32":
            # New process group → killpg-able.
            kwargs["start_new_session"] = True
        full_argv = (self.binary, *argv)
        proc = await asyncio.create_subprocess_exec(*full_argv, **kwargs)
        return proc, time.monotonic()

    # --------------------------------------------------------------- comm
    async def _communicate(
        self,
        proc: asyncio.subprocess.Process,
        stdin: Optional[bytes],
    ) -> tuple[bytes, bytes]:
        return await asyncio.wait_for(
            proc.communicate(input=stdin),
            timeout=self.timeout_s,
        )

    # ------------------------------------------------------------- kill
    async def _kill_tree(self, proc: asyncio.subprocess.Process) -> None:
        """Send SIGTERM, wait grace, then SIGKILL the process group."""
        if proc.returncode is not None:
            return
        try:
            if sys.platform != "win32":
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
            else:  # pragma: no cover — Windows path not exercised
                proc.terminate()
        except Exception:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=self.kill_grace_s)
            return
        except asyncio.TimeoutError:
            pass
        try:
            if sys.platform != "win32":
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            else:  # pragma: no cover
                proc.kill()
        except Exception:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass


# ---------------------------------------------------------------------------
# Container sandbox runner
# ---------------------------------------------------------------------------


@runtime_checkable
class SandboxHandle(Protocol):
    """Minimal handle the :class:`ContainerCLIRunner` needs to target a
    sandbox container.

    Any object exposing a ``container_name`` and an idempotent async
    ``ensure()`` satisfies this — e.g. GAPT's ``WorkspaceSandbox``. The
    executor deliberately knows nothing about *how* the container is created,
    cloned, or mounted (that is the host platform's concern). It only needs
    the running container's name and a way to make sure it is up before the
    first spawn.
    """

    @property
    def container_name(self) -> str: ...

    async def ensure(self) -> None: ...


@dataclass
class ContainerCLIRunner(CLIProcessRunner):
    """``CLIProcessRunner`` that spawns the CLI *inside* a sandbox container.

    Generalises the ``SandboxedCLIProcessRunner`` that previously lived in
    GAPT: only ``_spawn`` differs from the parent — argv becomes

        <launcher> exec -i -w <workdir> --env K=V ... <container> <bin> <argv>

    so the agent only ever sees the container's ``<workdir>`` (a bind mount),
    never the host filesystem. Everything else (timeout ladder,
    SIGTERM→SIGKILL process-group teardown via the host-side ``exec``, stderr
    collection, stream-json line buffering) is inherited unchanged:
    ``start_new_session`` is preserved on POSIX so killing the host-side
    ``exec`` group propagates to the CLI inside the container.

    The host needs the ``launcher`` (``docker`` by default) on PATH; it does
    **not** need the agent binary — that lives in the container image. The
    parent's host-binary existence check is therefore intentionally skipped.
    """

    sandbox: Optional[SandboxHandle] = None
    #: Working directory *inside* the container (the bind-mounted project root).
    workdir: str = "/workspace"
    #: Host launcher that enters the container. ``docker`` by default; any
    #: ``exec``-compatible CLI works (``podman`` etc.).
    launcher: str = "docker"
    #: The agent binary *inside* the container — always on PATH there (the
    #: image installs it). The host-side ``binary`` field is ignored for the
    #: actual spawn (it need not exist on the host).
    container_binary: str = "claude"

    def __post_init__(self) -> None:
        # Deliberately do NOT call super().__post_init__(): the parent
        # validates that ``binary`` exists on the *host*, but for a container
        # runner the agent binary lives in the image. Validate the launcher
        # (and that a sandbox was supplied) instead.
        if self.sandbox is None:
            raise ValueError("ContainerCLIRunner requires sandbox=")
        if not (shutil.which(self.launcher) or Path(self.launcher).exists()):
            raise CLIBinaryNotFound(f"container launcher not found: {self.launcher!r}")

    async def _spawn(
        self, argv: Sequence[str]
    ) -> tuple[asyncio.subprocess.Process, float]:
        sandbox = self.sandbox
        assert sandbox is not None  # guaranteed by __post_init__
        # First spawn after a host restart may hit a stopped container.
        # ensure() is idempotent; a failure here is non-fatal — the exec
        # below surfaces the real error if the container truly isn't up.
        try:
            await sandbox.ensure()
        except Exception:  # pragma: no cover - defensive
            logger.warning(
                "container_cli_runner.ensure_failed container=%s",
                getattr(sandbox, "container_name", "?"),
            )

        exec_argv: list[str] = ["exec", "-i", "-w", self.workdir]
        for k, v in dict(self.env_extras or {}).items():
            exec_argv += ["--env", f"{k}={v}"]
        # Inside the container the agent CLI is on PATH (the image installs
        # it). We deliberately don't forward ``self.binary`` — a host path
        # that need not exist in the container.
        exec_argv += [sandbox.container_name, self.container_binary, *list(argv)]

        kwargs: dict[str, Any] = dict(
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # The launcher needs the *host* env (PATH, DOCKER_HOST, ...). The
            # child's env is what we passed via --env flags above; that is
            # separate and already scoped.
            env=os.environ.copy(),
            cwd=None,
        )
        if sys.platform != "win32":
            kwargs["start_new_session"] = True
        proc = await asyncio.create_subprocess_exec(
            self.launcher, *exec_argv, **kwargs
        )
        return proc, time.monotonic()


# ---------------------------------------------------------------------------
# Internal coroutine helpers
# ---------------------------------------------------------------------------


async def _aiter_lines(
    stream: Optional[asyncio.StreamReader],
    *,
    timeout_s: float,
    start_t: float,
) -> AsyncIterator[bytes]:
    if stream is None:
        return
    while True:
        elapsed = time.monotonic() - start_t
        remaining = timeout_s - elapsed
        if remaining <= 0:
            raise CLITimeout(f"stream timeout after {timeout_s:.1f}s")
        try:
            line = await asyncio.wait_for(stream.readline(), timeout=remaining)
        except asyncio.TimeoutError as e:
            raise CLITimeout(f"stream readline timeout after {timeout_s:.1f}s") from e
        if not line:
            return
        yield line


async def _drain_stdin(
    stdin: asyncio.StreamWriter,
    chunks: AsyncIterator[bytes],
) -> None:
    try:
        async for chunk in chunks:
            if not chunk:
                continue
            stdin.write(chunk)
            await stdin.drain()
    finally:
        try:
            stdin.close()
        except Exception:
            pass


async def _collect_stderr(
    stream: Optional[asyncio.StreamReader],
    sink: list[bytes],
) -> None:
    if stream is None:
        return
    try:
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                return
            sink.append(chunk)
    except asyncio.CancelledError:
        return
    except Exception:
        return


# ---------------------------------------------------------------------------
# Convenience iterators
# ---------------------------------------------------------------------------


async def aiter_bytes(data: Optional[bytes]) -> AsyncIterator[bytes]:
    """Wrap a single bytes blob as an async iterator (for stdin_iter)."""
    if data:
        yield data
