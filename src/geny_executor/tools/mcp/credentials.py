"""Credential store abstraction for MCP OAuth (S8.1).

The MCP OAuth flow needs to persist tokens between pipeline runs so a
user only consents once. This module provides a tiny pluggable
:class:`CredentialStore` Protocol with two built-in implementations:

* :class:`MemoryCredentialStore` — process-lifetime dict. Tests and
  ephemeral hosts use this.
* :class:`FileCredentialStore` — atomic JSON-file persistence with
  ``mode=0600``. Plain text on disk; relies on filesystem permissions
  rather than encryption. Production hosts that need encryption
  should plug their own implementation (the protocol is intentionally
  small).

The :class:`OAuthFlow` (S8.2) and the MCP manager (S8.3+) consume
these via the protocol — there is no hard dependency on either
built-in implementation.

Key naming
----------

The MCP layer uses ``mcp:<server_name>`` as the canonical key prefix
for OAuth tokens (see :func:`mcp_credential_key`). Hosts that share
the same store with other subsystems should use disjoint prefixes to
avoid collisions.
"""

from __future__ import annotations

import json
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Protocol, runtime_checkable


_MCP_PREFIX = "mcp:"


def mcp_credential_key(server_name: str) -> str:
    """Canonical credential-store key for an MCP server's OAuth blob."""
    if not server_name:
        raise ValueError("server_name must be non-empty")
    return f"{_MCP_PREFIX}{server_name}"


@runtime_checkable
class CredentialStore(Protocol):
    """Minimal credential persistence contract.

    Implementations should be safe to call from a single async context
    (the manager serialises MCP work). Concurrency across processes is
    *not* required — the file-based store atomically replaces its
    backing file, so concurrent writers may lose updates but cannot
    corrupt the file.
    """

    def get(self, key: str) -> Optional[str]: ...

    def set(self, key: str, value: str) -> None: ...

    def delete(self, key: str) -> bool: ...

    def keys(self) -> List[str]: ...


class _BaseCredentialStore(ABC):
    """Common validation surface so subclasses share input checks."""

    @staticmethod
    def _check_key(key: str) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("credential key must be a non-empty string")

    @staticmethod
    def _check_value(value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("credential value must be a string")

    @abstractmethod
    def get(self, key: str) -> Optional[str]: ...

    @abstractmethod
    def set(self, key: str, value: str) -> None: ...

    @abstractmethod
    def delete(self, key: str) -> bool: ...

    @abstractmethod
    def keys(self) -> List[str]: ...


class MemoryCredentialStore(_BaseCredentialStore):
    """In-memory credential store. Loses data on process exit."""

    def __init__(self) -> None:
        self._data: Dict[str, str] = {}
        self._lock = Lock()

    def get(self, key: str) -> Optional[str]:
        self._check_key(key)
        with self._lock:
            return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        self._check_key(key)
        self._check_value(value)
        with self._lock:
            self._data[key] = value

    def delete(self, key: str) -> bool:
        self._check_key(key)
        with self._lock:
            return self._data.pop(key, None) is not None

    def keys(self) -> List[str]:
        with self._lock:
            return sorted(self._data.keys())


class FileCredentialStore(_BaseCredentialStore):
    """JSON-file credential store with mode-0600 atomic writes.

    The file is created on first :meth:`set` if missing. Reads tolerate
    a missing or empty file (returns ``None``). Writes go to a temp
    file in the same directory and ``os.replace`` into place so a
    crash mid-write cannot truncate the existing store.

    Security note: contents are stored *plaintext*. The 0600 file
    mode keeps other local users out, but a root-equivalent attacker
    can still read the tokens. For stronger guarantees, plug a
    Keychain-backed implementation by satisfying the
    :class:`CredentialStore` protocol.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._lock = Lock()

    @property
    def path(self) -> Path:
        return self._path

    def _read_locked(self) -> Dict[str, str]:
        if not self._path.exists():
            return {}
        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        if not raw.strip():
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"credential store file is corrupt (not valid JSON): {self._path}"
            ) from exc
        if not isinstance(data, dict):
            raise ValueError(f"credential store file must contain a JSON object: {self._path}")
        # Coerce values to str defensively — non-string values would
        # silently break downstream consumers.
        return {str(k): str(v) for k, v in data.items()}

    def _write_locked(self, data: Dict[str, str]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replace: write to a temp file in the same dir, fsync,
        # then os.replace.
        fd, tmp_path = tempfile.mkstemp(
            prefix=".cred-", suffix=".json.tmp", dir=str(self._path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, self._path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def get(self, key: str) -> Optional[str]:
        self._check_key(key)
        with self._lock:
            return self._read_locked().get(key)

    def set(self, key: str, value: str) -> None:
        self._check_key(key)
        self._check_value(value)
        with self._lock:
            data = self._read_locked()
            data[key] = value
            self._write_locked(data)

    def delete(self, key: str) -> bool:
        self._check_key(key)
        with self._lock:
            data = self._read_locked()
            if key not in data:
                return False
            del data[key]
            self._write_locked(data)
            return True

    def keys(self) -> List[str]:
        with self._lock:
            return sorted(self._read_locked().keys())


__all__ = [
    "CredentialStore",
    "MemoryCredentialStore",
    "FileCredentialStore",
    "mcp_credential_key",
]
