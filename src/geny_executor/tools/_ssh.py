"""Low-level SSH primitives for the built-in SSH tools.

Pure-async, built on ``asyncssh`` (native asyncio — no thread offload, so it
drops straight into a tool's ``async def execute``). Mirrors the shape of
``tools/_sandbox.py``: small helpers that open a connection, run a command (or
transfer a file), and return a ``(rc, stdout, stderr)`` triple, with all output
shaping/truncation left to the caller.

``asyncssh`` is an OPTIONAL dependency (extra ``ssh``). It is imported lazily so
the core install stays lean; callers get :class:`SSHUnavailableError` (a clean
install hint) when it is absent, never an ``ImportError`` at import time.

A "server" here is a plain dict — the credential record resolved by name from
the per-session store (see ``tools/built_in/_ssh_store.py``):

    {"name", "host", "port"?, "user", "password"?, "private_key"?,
     "passphrase"?, "description"?, "strict_host_key"?}

At least one of ``password`` / ``private_key`` must be present. The password
never leaves this module (it is fed to the transport / to ``sudo -S`` on
stdin), so the agent — which only ever passes a server *name* — never handles
it.
"""

from __future__ import annotations

import asyncio
import shlex
import time
from typing import Any, Dict, Optional, Tuple


class SSHUnavailableError(RuntimeError):
    """Raised when the optional ``asyncssh`` dependency is not installed."""


class SSHConfigError(ValueError):
    """Raised when a server record is malformed (missing host/creds, bad key)."""


_INSTALL_HINT = (
    "SSH tools require the `asyncssh` package. Install it via "
    "`pip install geny-executor[ssh]`."
)


def _require_asyncssh():
    try:
        import asyncssh  # type: ignore
    except ImportError as exc:  # pragma: no cover - env dependent
        raise SSHUnavailableError(_INSTALL_HINT) from exc
    return asyncssh


def _connect_kwargs(server: Dict[str, Any], *, connect_timeout: float) -> Tuple[str, Dict[str, Any]]:
    """Build ``(host, asyncssh.connect kwargs)`` from a server record.

    Host-key verification is DISABLED by default (``known_hosts=None``) so a
    user's own freshly-provisioned server "just works" on first contact; set
    ``strict_host_key: true`` on the record to opt back into verification.
    """
    asyncssh = _require_asyncssh()
    host = str(server.get("host") or "").strip()
    if not host:
        raise SSHConfigError("server record has no 'host'")
    try:
        port = int(server.get("port") or 22)
    except (TypeError, ValueError):
        raise SSHConfigError(f"invalid 'port': {server.get('port')!r}")
    username = str(server.get("user") or server.get("username") or "").strip() or None

    kwargs: Dict[str, Any] = {
        "port": port,
        "username": username,
        "connect_timeout": connect_timeout,
    }
    # known_hosts=None disables host-key checking; opt-in strict mode uses the
    # default system known_hosts.
    if not server.get("strict_host_key"):
        kwargs["known_hosts"] = None

    password = server.get("password") or None
    private_key = server.get("private_key") or None
    if private_key:
        passphrase = server.get("passphrase") or None
        try:
            key = asyncssh.import_private_key(private_key, passphrase)
        except Exception as exc:  # noqa: BLE001 — surface a clean config error
            raise SSHConfigError(f"invalid private_key: {exc}") from exc
        kwargs["client_keys"] = [key]
        if password:  # allow key + password (e.g. key + sudo password)
            kwargs["password"] = password
    elif password:
        kwargs["password"] = password
    else:
        raise SSHConfigError("server record has neither 'password' nor 'private_key'")

    return host, kwargs


async def _open(server: Dict[str, Any], *, connect_timeout: float):
    """Open an asyncssh connection for *server* (caller manages the context)."""
    asyncssh = _require_asyncssh()
    host, kwargs = _connect_kwargs(server, connect_timeout=connect_timeout)
    return await asyncssh.connect(host, **kwargs)


def _remote_command(command: str, *, cwd: Optional[str], sudo: bool, password: Optional[str]) -> Tuple[str, Optional[str]]:
    """Compose the remote command string + optional stdin (sudo password)."""
    inner = command
    if cwd:
        inner = f"cd {shlex.quote(cwd)} && {command}"
    if sudo:
        # -S reads the password from stdin; -p '' suppresses the prompt so it
        # can't pollute stdout. Wrap in a login shell so pipes/&&/redirs work.
        remote = f"sudo -S -p '' /bin/sh -c {shlex.quote(inner)}"
        return remote, (password or "") + "\n"
    return inner, None


async def ssh_exec(
    server: Dict[str, Any],
    command: str,
    *,
    timeout: float = 60.0,
    cwd: Optional[str] = None,
    sudo: bool = False,
    connect_timeout: float = 15.0,
) -> Tuple[int, str, str]:
    """Run *command* on *server*; return ``(exit_code, stdout, stderr)``.

    ``sudo=True`` runs it under ``sudo -S`` with the server's stored password
    fed on stdin — the agent never sees the password.
    """
    async with await _open(server, connect_timeout=connect_timeout) as conn:
        remote, run_input = _remote_command(
            command, cwd=cwd, sudo=sudo, password=server.get("password")
        )
        result = await asyncio.wait_for(
            conn.run(remote, input=run_input, check=False), timeout=timeout
        )
        rc = result.exit_status
        # exit_status is None on signal termination; map to 137-style non-zero.
        if rc is None:
            rc = 128 + int(result.exit_signal[1]) if getattr(result, "exit_signal", None) else 1
        return int(rc), str(result.stdout or ""), str(result.stderr or "")


async def ssh_test_connection(
    server: Dict[str, Any], *, connect_timeout: float = 15.0
) -> Dict[str, Any]:
    """Open a connection and run a trivial command to verify reachability +
    auth. Returns ``{success, latency_ms, error?}`` — never raises."""
    start = time.monotonic()
    try:
        async with await _open(server, connect_timeout=connect_timeout) as conn:
            r = await asyncio.wait_for(
                conn.run("echo geny-ssh-ok", check=False), timeout=connect_timeout
            )
            latency_ms = (time.monotonic() - start) * 1000.0
            ok = (r.exit_status == 0) and ("geny-ssh-ok" in str(r.stdout or ""))
            out: Dict[str, Any] = {"success": bool(ok), "latency_ms": round(latency_ms, 1)}
            if not ok:
                out["error"] = "connected but test command failed"
            return out
    except SSHUnavailableError as exc:
        return {"success": False, "error": str(exc)}
    except SSHConfigError as exc:
        return {"success": False, "error": f"config: {exc}"}
    except asyncio.TimeoutError:
        return {"success": False, "error": f"timed out after {connect_timeout:.0f}s"}
    except Exception as exc:  # noqa: BLE001 — auth/network/etc → friendly string
        return {"success": False, "error": _friendly_error(exc)}


async def sftp_put(
    server: Dict[str, Any], local_path: str, remote_path: str, *, connect_timeout: float = 15.0
) -> None:
    async with await _open(server, connect_timeout=connect_timeout) as conn:
        async with conn.start_sftp_client() as sftp:
            await sftp.put(local_path, remote_path)


async def sftp_get(
    server: Dict[str, Any], remote_path: str, local_path: str, *, connect_timeout: float = 15.0
) -> None:
    async with await _open(server, connect_timeout=connect_timeout) as conn:
        async with conn.start_sftp_client() as sftp:
            await sftp.get(remote_path, local_path)


def _friendly_error(exc: Exception) -> str:
    """Map common asyncssh/OS errors to a short, safe message (no secrets)."""
    name = type(exc).__name__
    msg = str(exc).strip() or name
    lowered = msg.lower()
    if "permission denied" in lowered or "authentication" in lowered or name == "PermissionDenied":
        return "authentication failed (check user / password / key)"
    if "name or service not known" in lowered or "nodename nor servname" in lowered:
        return "host not found (check host / DNS)"
    if "connection refused" in lowered:
        return "connection refused (is SSH listening on that host/port?)"
    if "timed out" in lowered or name == "TimeoutError":
        return "connection timed out"
    return f"{name}: {msg}"


__all__ = [
    "SSHUnavailableError",
    "SSHConfigError",
    "ssh_exec",
    "ssh_test_connection",
    "sftp_put",
    "sftp_get",
]
