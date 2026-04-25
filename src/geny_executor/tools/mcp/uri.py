"""``mcp://`` URI scheme for cross-server resource references (S8.3).

A small grammar that lets callers (Stage 2 retrievers, host code,
:class:`MCPResourceRetriever`) refer to resources without having to
carry around a separate ``server_name`` field.

Grammar
-------

    mcp://<server_name>[/<resource_id>]

* ``server_name`` — the name registered with :class:`MCPManager`.
  Must match the regex ``[A-Za-z0-9_.-]+``. URL-encoded forms are
  rejected to keep the parser unambiguous.
* ``resource_id`` — everything after the first ``/`` following the
  server name. Passed back to the underlying MCP SDK
  (``client_session.read_resource``) verbatim, so any URI scheme the
  server understands works (``file:///foo``, plain paths, etc.).
  May be empty (``mcp://server`` or ``mcp://server/``).

Hosts that want to express server-native URIs containing slashes can
simply include them — there is no extra escaping. The leading
``mcp://server/`` prefix is the boundary; everything after is opaque.
"""

from __future__ import annotations

import re
from typing import Tuple

MCP_URI_SCHEME = "mcp://"

# Server-name validity. We deliberately exclude '/' so the prefix
# delimiter is unambiguous, and exclude colons / encoded chars to keep
# the parser trivially round-trippable.
_SERVER_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class MCPURIError(ValueError):
    """Raised when an ``mcp://`` URI cannot be parsed."""


def is_mcp_uri(value: str) -> bool:
    """Cheap prefix check — does not validate the rest."""
    return isinstance(value, str) and value.startswith(MCP_URI_SCHEME)


def parse_mcp_uri(uri: str) -> Tuple[str, str]:
    """Split an ``mcp://`` URI into ``(server_name, resource_id)``.

    ``resource_id`` is the empty string for bare ``mcp://server`` /
    ``mcp://server/`` URIs. Raises :class:`MCPURIError` for invalid
    inputs (wrong scheme, missing/invalid server name).
    """
    if not isinstance(uri, str):
        raise MCPURIError(f"mcp:// URI must be a string, got {type(uri).__name__}")
    if not uri.startswith(MCP_URI_SCHEME):
        raise MCPURIError(f"not an mcp:// URI: {uri!r}")
    rest = uri[len(MCP_URI_SCHEME) :]
    if not rest:
        raise MCPURIError("mcp:// URI is missing the server name")
    if "/" in rest:
        server_name, resource_id = rest.split("/", 1)
    else:
        server_name, resource_id = rest, ""
    if not server_name:
        raise MCPURIError("mcp:// URI is missing the server name")
    if not _SERVER_NAME_RE.match(server_name):
        raise MCPURIError(
            f"invalid server name {server_name!r} in mcp:// URI "
            f"(allowed chars: letters, digits, '_', '.', '-')"
        )
    return server_name, resource_id


def build_mcp_uri(server_name: str, resource_id: str = "") -> str:
    """Compose an ``mcp://`` URI. Symmetric with :func:`parse_mcp_uri`."""
    if not isinstance(server_name, str):
        raise MCPURIError(f"server_name must be a string, got {type(server_name).__name__}")
    if not _SERVER_NAME_RE.match(server_name):
        raise MCPURIError(
            f"invalid server name {server_name!r} (allowed chars: letters, digits, '_', '.', '-')"
        )
    if not isinstance(resource_id, str):
        raise MCPURIError(f"resource_id must be a string, got {type(resource_id).__name__}")
    if resource_id:
        # Strip a single leading '/' so build/parse round-trips.
        if resource_id.startswith("/"):
            resource_id = resource_id[1:]
        return f"{MCP_URI_SCHEME}{server_name}/{resource_id}"
    return f"{MCP_URI_SCHEME}{server_name}/"


__all__ = [
    "MCPURIError",
    "MCP_URI_SCHEME",
    "build_mcp_uri",
    "is_mcp_uri",
    "parse_mcp_uri",
]
