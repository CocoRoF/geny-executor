"""Built-in SSH tools — run commands and move files on the session's
pre-configured servers.

The host (Geny) records the user's servers per session (see
``_ssh_store.SSHServerStore``). Each tool takes a server by NAME and resolves
the credential internally, so the agent operates servers without ever handling
a password or key. Gated on ``feature:ssh_enabled`` so the family stays hidden
until the host provisions SSH for the session.

Tools:
  * ``SshListServers`` — what servers can I reach? (names + host/user, no secrets)
  * ``SshRun``         — run a shell command on a named server (optional sudo)
  * ``SshUpload``      — copy a session-storage file → server (SFTP)
  * ``SshDownload``    — copy a server file → session storage (SFTP)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from geny_executor.tools.base import Tool, ToolCapabilities, ToolResult
from geny_executor.tools._ssh import (
    SSHConfigError,
    SSHUnavailableError,
    sftp_get,
    sftp_put,
    ssh_exec,
)
from geny_executor.tools.built_in._ssh_store import SSHServerStore

_SSH_FEATURE_KEY = "feature:ssh_enabled"
_MAX_OUTPUT = 100_000  # chars per stream, matching BashTool
_DEFAULT_TIMEOUT = 60.0
_MAX_TIMEOUT = 900.0
_MAX_TRANSFER_BYTES = 100 * 1024 * 1024


def _err(code: str, message: str) -> ToolResult:
    return ToolResult(content={"error": {"code": code, "message": message}}, is_error=True)


def _store(context: Any) -> SSHServerStore:
    return SSHServerStore.from_context(context)


def _resolve_or_error(store: SSHServerStore, name: Optional[str]):
    name = (name or "").strip()
    if not name:
        return None, _err("NO_SERVER", "Provide 'server' (a configured server name).")
    server = store.resolve(name)
    if server is None:
        avail = ", ".join(store.names()) or "(none configured)"
        return None, _err(
            "UNKNOWN_SERVER",
            f"No SSH server named '{name}'. Available: {avail}. "
            "Use SshListServers to see configured servers.",
        )
    return server, None


def _truncate(text: str) -> tuple[str, bool]:
    if len(text) > _MAX_OUTPUT:
        return text[:_MAX_OUTPUT], True
    return text, False


def _storage_root(context: Any) -> Optional[Path]:
    sp = getattr(context, "storage_path", None)
    return Path(sp).resolve() if sp else None


def _guarded_local(context: Any, path: str):
    """Resolve *path* under the session storage root, refusing escapes."""
    root = _storage_root(context)
    if root is None:
        return None, _err("NO_STORAGE", "This session has no storage_path for local files.")
    p = Path(path)
    target = (p if p.is_absolute() else root / p).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None, _err("PATH_ESCAPE", f"Local path escapes the session storage: {path}")
    return target, None


class _SSHToolBase(Tool):
    """Shared feature gate for the SSH family."""

    def required_config_keys(self) -> List[str]:
        # Host gate — hidden until Geny marks the session SSH-provisioned.
        return [_SSH_FEATURE_KEY]


class SshListServersTool(_SSHToolBase):
    """List the servers this session can SSH into (no secrets)."""

    @property
    def name(self) -> str:
        return "SshListServers"

    @property
    def description(self) -> str:
        return (
            "List the SSH servers configured for this session — name, host, "
            "port, user, and description. Passwords/keys are never shown. Use a "
            "server's 'name' with SshRun / SshUpload / SshDownload."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(read_only=True, concurrency_safe=True, idempotent=True)

    async def execute(self, input: Dict[str, Any], context: Any) -> ToolResult:
        servers = _store(context).list_public()
        if not servers:
            return ToolResult(
                content={"servers": []},
                display_text="No SSH servers are configured for this session.",
            )
        lines = [
            f"- {s['name']}: {s['user']}@{s['host']}:{s['port']} "
            f"[{s['auth']}]" + (f" — {s['description']}" if s["description"] else "")
            for s in servers
        ]
        return ToolResult(
            content={"servers": servers},
            display_text="Configured SSH servers:\n" + "\n".join(lines),
        )


class SshRunTool(_SSHToolBase):
    """Run a shell command on a named server over SSH."""

    @property
    def name(self) -> str:
        return "SshRun"

    @property
    def description(self) -> str:
        return (
            "Run a shell command on a configured SSH server and return its "
            "stdout, stderr, and exit code. Pass 'server' (a name from "
            "SshListServers) and 'command'. Optional: 'cwd' (remote working "
            "dir), 'timeout' seconds, and 'sudo' (run under sudo using the "
            "server's stored password — you never handle the password)."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "server": {"type": "string", "description": "Configured server name."},
                "command": {"type": "string", "description": "Shell command to run remotely."},
                "cwd": {"type": "string", "description": "Remote working directory (optional)."},
                "timeout": {
                    "type": "number",
                    "description": f"Max seconds to wait (default {int(_DEFAULT_TIMEOUT)}, max {int(_MAX_TIMEOUT)}).",
                },
                "sudo": {"type": "boolean", "description": "Run under sudo -S (default false)."},
            },
            "required": ["server", "command"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(network_egress=True, interrupt="cancel")

    async def execute(self, input: Dict[str, Any], context: Any) -> ToolResult:
        server, err = _resolve_or_error(_store(context), input.get("server"))
        if err is not None:
            return err
        command = str(input.get("command") or "").strip()
        if not command:
            return _err("NO_COMMAND", "Provide a non-empty 'command'.")
        try:
            timeout = float(input.get("timeout") or _DEFAULT_TIMEOUT)
        except (TypeError, ValueError):
            timeout = _DEFAULT_TIMEOUT
        timeout = max(1.0, min(timeout, _MAX_TIMEOUT))
        cwd = (input.get("cwd") or None)
        sudo = bool(input.get("sudo"))
        if sudo and not server.get("password"):
            return _err(
                "NO_SUDO_PASSWORD",
                f"Server '{server.get('name')}' has no stored password, so sudo "
                "cannot be used (key-only auth). Configure a password to enable sudo.",
            )

        name = server.get("name")
        try:
            rc, out, err_out = await ssh_exec(
                server, command, timeout=timeout, cwd=cwd, sudo=sudo
            )
        except SSHUnavailableError as exc:
            return _err("SSH_UNAVAILABLE", str(exc))
        except SSHConfigError as exc:
            return _err("BAD_SERVER", f"Server '{name}': {exc}")
        except asyncio.TimeoutError:
            return _err("TIMEOUT", f"Command on '{name}' exceeded {timeout:.0f}s.")
        except Exception as exc:  # noqa: BLE001 — auth/network → structured error
            return _err("SSH_ERROR", f"{type(exc).__name__}: {exc}")

        out_s, out_trunc = _truncate(out)
        err_s, err_trunc = _truncate(err_out)
        content = {
            "server": name,
            "exit_code": rc,
            "stdout": out_s,
            "stderr": err_s,
            "truncated": bool(out_trunc or err_trunc),
        }
        header = f"[{name}] exit={rc}" + (" (sudo)" if sudo else "")
        body = out_s if out_s else ""
        if err_s:
            body += ("\n" if body else "") + "stderr:\n" + err_s
        display = header + ("\n" + body if body else "")
        return ToolResult(
            content=content,
            is_error=(rc != 0),
            display_text=display,
            metadata={"exit_code": rc, "server": name},
        )


class SshUploadTool(_SSHToolBase):
    """Upload a session-storage file to a server via SFTP."""

    @property
    def name(self) -> str:
        return "SshUpload"

    @property
    def description(self) -> str:
        return (
            "Upload a file from this session's storage to a configured SSH "
            "server via SFTP. Pass 'server', 'local_path' (relative to session "
            "storage), and 'remote_path' (absolute path on the server)."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "server": {"type": "string", "description": "Configured server name."},
                "local_path": {"type": "string", "description": "File in session storage to upload."},
                "remote_path": {"type": "string", "description": "Destination path on the server."},
            },
            "required": ["server", "local_path", "remote_path"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(network_egress=True)

    async def execute(self, input: Dict[str, Any], context: Any) -> ToolResult:
        server, err = _resolve_or_error(_store(context), input.get("server"))
        if err is not None:
            return err
        local, lerr = _guarded_local(context, str(input.get("local_path") or ""))
        if lerr is not None:
            return lerr
        if not local.is_file():
            return _err("NOT_FOUND", f"Local file not found: {input.get('local_path')}")
        if local.stat().st_size > _MAX_TRANSFER_BYTES:
            return _err("TOO_LARGE", f"File exceeds {_MAX_TRANSFER_BYTES // (1024*1024)} MB transfer cap.")
        remote_path = str(input.get("remote_path") or "").strip()
        if not remote_path:
            return _err("NO_REMOTE_PATH", "Provide 'remote_path'.")
        name = server.get("name")
        try:
            await sftp_put(server, str(local), remote_path)
        except SSHUnavailableError as exc:
            return _err("SSH_UNAVAILABLE", str(exc))
        except SSHConfigError as exc:
            return _err("BAD_SERVER", f"Server '{name}': {exc}")
        except Exception as exc:  # noqa: BLE001
            return _err("SFTP_ERROR", f"{type(exc).__name__}: {exc}")
        return ToolResult(
            content={"server": name, "uploaded": remote_path, "bytes": local.stat().st_size},
            display_text=f"Uploaded {local.name} → {name}:{remote_path}",
        )


class SshDownloadTool(_SSHToolBase):
    """Download a server file into session storage via SFTP."""

    @property
    def name(self) -> str:
        return "SshDownload"

    @property
    def description(self) -> str:
        return (
            "Download a file from a configured SSH server into this session's "
            "storage via SFTP. Pass 'server', 'remote_path' (path on the "
            "server), and 'local_path' (relative to session storage)."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "server": {"type": "string", "description": "Configured server name."},
                "remote_path": {"type": "string", "description": "File path on the server."},
                "local_path": {"type": "string", "description": "Destination in session storage."},
            },
            "required": ["server", "remote_path", "local_path"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(network_egress=True)

    async def execute(self, input: Dict[str, Any], context: Any) -> ToolResult:
        server, err = _resolve_or_error(_store(context), input.get("server"))
        if err is not None:
            return err
        local, lerr = _guarded_local(context, str(input.get("local_path") or ""))
        if lerr is not None:
            return lerr
        remote_path = str(input.get("remote_path") or "").strip()
        if not remote_path:
            return _err("NO_REMOTE_PATH", "Provide 'remote_path'.")
        name = server.get("name")
        try:
            local.parent.mkdir(parents=True, exist_ok=True)
            await sftp_get(server, remote_path, str(local))
        except SSHUnavailableError as exc:
            return _err("SSH_UNAVAILABLE", str(exc))
        except SSHConfigError as exc:
            return _err("BAD_SERVER", f"Server '{name}': {exc}")
        except Exception as exc:  # noqa: BLE001
            return _err("SFTP_ERROR", f"{type(exc).__name__}: {exc}")
        size = local.stat().st_size if local.is_file() else 0
        return ToolResult(
            content={"server": name, "downloaded": str(local), "bytes": size},
            display_text=f"Downloaded {name}:{remote_path} → {input.get('local_path')} ({size} bytes)",
        )


SSH_TOOL_CLASSES: Dict[str, type] = {
    "SshListServers": SshListServersTool,
    "SshRun": SshRunTool,
    "SshUpload": SshUploadTool,
    "SshDownload": SshDownloadTool,
}

__all__ = [
    "SshListServersTool",
    "SshRunTool",
    "SshUploadTool",
    "SshDownloadTool",
    "SSH_TOOL_CLASSES",
]
