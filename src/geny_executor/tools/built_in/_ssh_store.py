"""Per-session SSH server credential store (file-backed).

Each session records its known servers as a JSON file at
``<storage_path>/ssh/servers.json``. The host (Geny) hands the user's
configured servers to the session via ``ToolContext.extras["ssh"]["servers"]``;
this store persists them to that file so the session has a durable, per-session
record — and, when no host injection is present (e.g. the executor used
standalone), it reads whatever is already in the file.

The split that keeps secrets away from the agent:

* :meth:`SSHServerStore.list_public` returns only non-secret metadata (name,
  host, port, user, description, auth kind) — safe to show the model.
* :meth:`SSHServerStore.resolve` returns the FULL record (including password /
  private key), used ONLY inside a tool to open the connection. The agent
  passes a server *name*; it never sees or handles the credential.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

_SECRET_KEYS = ("password", "private_key", "passphrase")


def _server_dir(storage_path: Optional[str]) -> Optional[Path]:
    if not storage_path:
        return None
    return Path(storage_path).resolve() / "ssh"


def _auth_kind(server: Dict[str, Any]) -> str:
    has_pw = bool(server.get("password"))
    has_key = bool(server.get("private_key"))
    if has_pw and has_key:
        return "password+key"
    if has_key:
        return "key"
    if has_pw:
        return "password"
    return "none"


class SSHServerStore:
    """A per-session view of the configured SSH servers, keyed by name."""

    def __init__(self, servers: Optional[List[Dict[str, Any]]] = None) -> None:
        self._by_name: Dict[str, Dict[str, Any]] = {}
        for s in servers or []:
            name = str(s.get("name") or "").strip()
            if not name:
                continue
            self._by_name[name] = dict(s)

    # ── construction ────────────────────────────────────────────────
    @classmethod
    def from_context(cls, context: Any) -> "SSHServerStore":
        """Build the store for a session.

        Source of record is the host-injected list on
        ``context.extras["ssh"]["servers"]``. When present it is (re)written to
        the per-session file so the session keeps a durable record; when absent
        the file is read as the fallback source.
        """
        extras = getattr(context, "extras", None) or {}
        ssh_extra = extras.get("ssh") if isinstance(extras, dict) else None
        injected = (ssh_extra or {}).get("servers") if isinstance(ssh_extra, dict) else None
        injected = injected if isinstance(injected, list) else None

        sdir = _server_dir(getattr(context, "storage_path", None))

        if injected:
            # Persist the host-provided list as the per-session record.
            if sdir is not None:
                cls._write_file(sdir, injected)
            return cls(injected)

        # No host injection — read the file record if one exists.
        if sdir is not None:
            servers = cls._read_file(sdir)
            if servers:
                return cls(servers)
        return cls([])

    # ── lookups ─────────────────────────────────────────────────────
    def names(self) -> List[str]:
        return list(self._by_name.keys())

    def is_empty(self) -> bool:
        return not self._by_name

    def resolve(self, name: str) -> Optional[Dict[str, Any]]:
        """Full record (WITH secrets) for internal connection use only."""
        return self._by_name.get(str(name).strip())

    def list_public(self) -> List[Dict[str, Any]]:
        """Non-secret metadata for every server — safe to show the agent."""
        out: List[Dict[str, Any]] = []
        for name, s in self._by_name.items():
            out.append(
                {
                    "name": name,
                    "host": s.get("host", ""),
                    "port": int(s.get("port") or 22),
                    "user": s.get("user") or s.get("username") or "",
                    "description": s.get("description", "") or "",
                    "auth": _auth_kind(s),
                }
            )
        return out

    # ── file I/O ────────────────────────────────────────────────────
    @staticmethod
    def _read_file(sdir: Path) -> List[Dict[str, Any]]:
        path = sdir / "servers.json"
        try:
            if not path.is_file():
                return []
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if isinstance(data, dict):  # tolerate {"servers": [...]}
            data = data.get("servers")
        return [s for s in data if isinstance(s, dict)] if isinstance(data, list) else []

    @staticmethod
    def _write_file(sdir: Path, servers: List[Dict[str, Any]]) -> None:
        try:
            sdir.mkdir(parents=True, exist_ok=True)
            path = sdir / "servers.json"
            tmp = sdir / "servers.json.tmp"
            tmp.write_text(json.dumps(list(servers), ensure_ascii=False, indent=2), encoding="utf-8")
            # Credentials on disk — keep them owner-only.
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
            os.replace(tmp, path)
        except OSError:
            pass  # best-effort record; the in-memory view still works


__all__ = ["SSHServerStore"]
