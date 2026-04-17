"""SSRF Protection — blocks requests to internal/private IP ranges.

Used by HTTP-based ad-hoc tools and MCP server connections to prevent
Server-Side Request Forgery attacks.
"""

from __future__ import annotations

import ipaddress
import socket
import urllib.parse
from typing import List, Optional


# RFC 1918 / RFC 4193 / link-local ranges that should never be
# reachable from outbound HTTP tool calls.
_BLOCKED_NETWORKS: List[ipaddress._BaseNetwork] = [
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("224.0.0.0/4"),      # multicast
    ipaddress.ip_network("255.255.255.255/32"),
    # IPv6
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]

_ALLOWED_SCHEMES = frozenset({"http", "https"})


class SSRFError(Exception):
    """Raised when a URL targets a blocked address."""


def validate_url(url: str, *, extra_blocked: Optional[List[str]] = None) -> str:
    """Validate that *url* does not resolve to a private/internal address.

    Returns the normalised URL on success.
    Raises ``SSRFError`` if the target is blocked.

    Parameters
    ----------
    url:
        The URL to validate.
    extra_blocked:
        Additional CIDR blocks to deny (e.g. ``["100.64.0.0/10"]``).
    """
    parsed = urllib.parse.urlparse(url)

    # 1. Scheme check
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise SSRFError(f"Blocked scheme: {parsed.scheme!r} (allowed: {_ALLOWED_SCHEMES})")

    # 2. Hostname present?
    hostname = parsed.hostname
    if not hostname:
        raise SSRFError("URL has no hostname")

    # 3. Block obvious numeric literals for 127.x, 0x7f, etc.
    #    (covers decimal, hex, and octal IP tricks)
    blocked = list(_BLOCKED_NETWORKS)
    if extra_blocked:
        for cidr in extra_blocked:
            blocked.append(ipaddress.ip_network(cidr, strict=False))

    # 4. DNS resolution — check every resolved address
    try:
        addr_info = socket.getaddrinfo(hostname, parsed.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise SSRFError(f"DNS resolution failed for {hostname!r}: {exc}") from exc

    if not addr_info:
        raise SSRFError(f"No addresses resolved for {hostname!r}")

    for family, _type, _proto, _canonname, sockaddr in addr_info:
        ip = ipaddress.ip_address(sockaddr[0])
        for network in blocked:
            if ip in network:
                raise SSRFError(
                    f"Blocked: {hostname!r} resolves to {ip} "
                    f"which is in {network}"
                )

    return url


def is_safe_url(url: str) -> bool:
    """Non-raising convenience wrapper around :func:`validate_url`."""
    try:
        validate_url(url)
        return True
    except (SSRFError, Exception):
        return False
