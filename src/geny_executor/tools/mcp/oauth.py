"""MCP OAuth 2.0 authorization-code flow (S8.2).

Implements the standard OAuth 2.0 authorization-code flow for MCP
servers that require user consent (e.g. Google Drive).

Flow
----
1. ``OAuthFlow.authorize(server_name, auth_config)`` is called.
2. A cryptographically-random ``state`` token is generated for CSRF
   protection.
3. A local HTTP server is bound to ``127.0.0.1`` on the configured
   port (``0`` → OS picks a free port).
4. The authorization URL (``auth_config.authorize_url`` with
   ``client_id`` / ``redirect_uri`` / ``state`` / ``scope``) is
   handed to the configured ``consent_handler`` so the host can open
   it in a browser.
5. The browser redirects back to ``http://127.0.0.1:<port>/callback``
   with ``?code=...&state=...``.
6. The flow verifies the returned ``state`` matches what was issued
   and exchanges the code at ``auth_config.token_url`` via httpx.
7. The token blob is JSON-encoded and stored in the
   :class:`CredentialStore` under :func:`mcp_credential_key`.

Security
--------
* Callback server binds ``127.0.0.1`` only — never an external
  interface. A different host can be passed but a runtime warning
  in the docs (NOT enforced) signals the risk.
* CSRF state is 32-byte URL-safe random; mismatch → ``ValueError``.
* The token exchange uses ``application/x-www-form-urlencoded`` per
  RFC 6749 §4.1.3. The HTTP client is injectable for tests so the
  test suite never hits real OAuth servers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import socket
import time
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from geny_executor.tools.mcp.credentials import (
    CredentialStore,
    mcp_credential_key,
)

logger = logging.getLogger(__name__)


# ── data classes ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class OAuthAuthConfig:
    """OAuth 2.0 client configuration for one MCP server."""

    client_id: str
    client_secret: str
    authorize_url: str
    token_url: str
    scopes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.client_id:
            raise ValueError("client_id is required")
        if not self.authorize_url:
            raise ValueError("authorize_url is required")
        if not self.token_url:
            raise ValueError("token_url is required")


@dataclass
class OAuthToken:
    """OAuth token response, normalised."""

    access_token: str
    token_type: str = "Bearer"
    refresh_token: Optional[str] = None
    expires_at: Optional[float] = None  # epoch seconds
    scope: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self, *, leeway_seconds: float = 30.0) -> bool:
        if self.expires_at is None:
            return False
        return time.time() >= (self.expires_at - leeway_seconds)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, blob: str) -> "OAuthToken":
        data = json.loads(blob)
        return cls(
            access_token=data["access_token"],
            token_type=data.get("token_type", "Bearer"),
            refresh_token=data.get("refresh_token"),
            expires_at=data.get("expires_at"),
            scope=data.get("scope"),
            raw=data.get("raw") or {},
        )

    @classmethod
    def from_token_response(cls, payload: Dict[str, Any]) -> "OAuthToken":
        access_token = payload.get("access_token")
        if not access_token:
            raise ValueError("token response missing 'access_token'")
        expires_in = payload.get("expires_in")
        expires_at: Optional[float] = None
        if isinstance(expires_in, (int, float)) and expires_in > 0:
            expires_at = time.time() + float(expires_in)
        return cls(
            access_token=str(access_token),
            token_type=str(payload.get("token_type") or "Bearer"),
            refresh_token=payload.get("refresh_token"),
            expires_at=expires_at,
            scope=payload.get("scope"),
            raw=dict(payload),
        )


# ── error type ─────────────────────────────────────────────────────────


class OAuthError(RuntimeError):
    """Raised when the OAuth flow cannot complete."""


# ── helpers ────────────────────────────────────────────────────────────


def _generate_state() -> str:
    """32-byte URL-safe state token (~43 chars)."""
    return secrets.token_urlsafe(32)


def build_authorize_url(
    config: OAuthAuthConfig,
    *,
    redirect_uri: str,
    state: str,
    extra_params: Optional[Dict[str, str]] = None,
) -> str:
    """Compose the authorization URL with the standard query string."""
    params: Dict[str, str] = {
        "response_type": "code",
        "client_id": config.client_id,
        "redirect_uri": redirect_uri,
        "state": state,
    }
    if config.scopes:
        params["scope"] = " ".join(config.scopes)
    if extra_params:
        params.update(extra_params)
    sep = "&" if "?" in config.authorize_url else "?"
    return f"{config.authorize_url}{sep}{urlencode(params)}"


# ── callback HTTP server ───────────────────────────────────────────────


_CALLBACK_PATH = "/callback"
_SUCCESS_BODY = (
    b"<html><body><h1>Authorization complete</h1>"
    b"<p>You can close this tab and return to the application.</p></body></html>"
)
_ERROR_BODY_TEMPLATE = "<html><body><h1>Authorization failed</h1><p>{reason}</p></body></html>"


class _CallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler that captures the OAuth callback."""

    # Per-server slots filled in by _serve_callback
    received: Dict[str, str] = {}  # noqa: RUF012 — per-server class attr swap pattern
    completion: asyncio.Future = None  # type: ignore[assignment]
    loop: asyncio.AbstractEventLoop = None  # type: ignore[assignment]

    def do_GET(self) -> None:  # noqa: N802 — http.server's required name
        url = urlparse(self.path)
        if url.path != _CALLBACK_PATH:
            self.send_response(404)
            self.end_headers()
            return
        params = parse_qs(url.query)
        code = (params.get("code") or [""])[0]
        state = (params.get("state") or [""])[0]
        error = (params.get("error") or [""])[0]

        if error:
            self.send_response(400)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(_ERROR_BODY_TEMPLATE.format(reason=error).encode("utf-8"))
            self._signal({"error": error, "state": state})
            return

        if not code:
            self.send_response(400)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(_ERROR_BODY_TEMPLATE.format(reason="missing 'code'").encode("utf-8"))
            self._signal({"error": "missing code", "state": state})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(_SUCCESS_BODY)
        self._signal({"code": code, "state": state})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002, ARG002
        # Silence the default stderr access-log spam.
        return

    def _signal(self, payload: Dict[str, str]) -> None:
        if self.loop is None or self.completion is None:
            return
        if self.completion.done():
            return
        # Future lives in the asyncio loop — schedule from this thread.
        self.loop.call_soon_threadsafe(self.completion.set_result, payload)


def _bind_server(host: str, port: int) -> HTTPServer:
    """Bind an HTTPServer; ``port=0`` lets the OS pick."""
    handler = type("_PerCallHandler", (_CallbackHandler,), {})
    server = HTTPServer((host, port), handler)
    return server


# ── OAuthFlow ──────────────────────────────────────────────────────────


ConsentHandler = Callable[[str], None]
HttpClient = Callable[[str, Dict[str, str]], Dict[str, Any]]
"""Token-exchange client. Receives (token_url, form_payload), returns parsed JSON."""


async def _default_http_post(token_url: str, form: Dict[str, str]) -> Dict[str, Any]:
    """Default token exchange over httpx."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            token_url,
            data=form,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
    if resp.status_code >= 400:
        raise OAuthError(f"token endpoint returned {resp.status_code}: {resp.text[:200]}")
    try:
        return resp.json()
    except ValueError as exc:  # not JSON
        raise OAuthError(f"token endpoint returned non-JSON: {resp.text[:200]}") from exc


class OAuthFlow:
    """End-to-end OAuth 2.0 authorization-code orchestration.

    Use one instance per credential store. Typical wiring::

        store = FileCredentialStore("~/.geny/mcp_credentials.json")
        flow = OAuthFlow(store, consent_handler=lambda url: print(url))
        token = await flow.authorize("gdrive", auth_config)

    The callback HTTP server is started lazily inside
    :meth:`authorize`; nothing binds a port until an authorization
    is in flight.
    """

    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_TIMEOUT_SECONDS = 300.0  # 5 minutes for the user to consent

    def __init__(
        self,
        credential_store: CredentialStore,
        *,
        callback_host: str = DEFAULT_HOST,
        callback_port: int = 0,
        consent_handler: Optional[ConsentHandler] = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        http_post: Optional[Callable[[str, Dict[str, str]], Any]] = None,
    ) -> None:
        if callback_port < 0 or callback_port > 65_535:
            raise ValueError("callback_port must be in [0, 65535]")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._store = credential_store
        self._host = callback_host
        self._port = callback_port
        self._consent_handler = consent_handler or (lambda url: logger.info("OAuth URL: %s", url))
        self._timeout = float(timeout_seconds)
        self._http_post = http_post or _default_http_post

    @property
    def credential_store(self) -> CredentialStore:
        return self._store

    @property
    def callback_host(self) -> str:
        return self._host

    @property
    def callback_port(self) -> int:
        return self._port

    # ── public API ──

    def load_cached_token(self, server_name: str) -> Optional[OAuthToken]:
        blob = self._store.get(mcp_credential_key(server_name))
        if not blob:
            return None
        try:
            return OAuthToken.from_json(blob)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring corrupt cached OAuth token for %s: %s", server_name, exc)
            return None

    def revoke_cached_token(self, server_name: str) -> bool:
        return self._store.delete(mcp_credential_key(server_name))

    async def authorize(
        self,
        server_name: str,
        auth_config: OAuthAuthConfig,
        *,
        extra_authorize_params: Optional[Dict[str, str]] = None,
    ) -> OAuthToken:
        """Run the full authorization-code flow and persist the token."""
        if not server_name:
            raise ValueError("server_name must be non-empty")

        loop = asyncio.get_running_loop()
        completion: asyncio.Future = loop.create_future()
        state = _generate_state()

        server = _bind_server(self._host, self._port)
        bound_port = server.server_address[1]
        redirect_uri = f"http://{self._host}:{bound_port}{_CALLBACK_PATH}"

        # Wire the per-call class slots.
        handler_cls = server.RequestHandlerClass
        handler_cls.completion = completion  # type: ignore[attr-defined]
        handler_cls.loop = loop  # type: ignore[attr-defined]

        thread = Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.1},
            daemon=True,
            name=f"mcp-oauth-{server_name}",
        )
        thread.start()
        try:
            authorize_url = build_authorize_url(
                auth_config,
                redirect_uri=redirect_uri,
                state=state,
                extra_params=extra_authorize_params,
            )
            self._consent_handler(authorize_url)

            try:
                payload = await asyncio.wait_for(completion, timeout=self._timeout)
            except asyncio.TimeoutError as exc:
                raise OAuthError(
                    f"timed out waiting for OAuth callback after {self._timeout:.0f}s"
                ) from exc

            if payload.get("error"):
                raise OAuthError(f"authorization denied: {payload['error']}")
            if payload.get("state") != state:
                raise OAuthError("OAuth state mismatch — possible CSRF attempt")
            code = payload.get("code")
            if not code:
                raise OAuthError("OAuth callback returned no code")

            token_payload = await self._exchange_code(
                auth_config, code=code, redirect_uri=redirect_uri
            )
            token = OAuthToken.from_token_response(token_payload)
            self._store.set(mcp_credential_key(server_name), token.to_json())
            return token
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)

    async def _exchange_code(
        self,
        auth_config: OAuthAuthConfig,
        *,
        code: str,
        redirect_uri: str,
    ) -> Dict[str, Any]:
        form: Dict[str, str] = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": auth_config.client_id,
        }
        if auth_config.client_secret:
            form["client_secret"] = auth_config.client_secret
        result = self._http_post(auth_config.token_url, form)
        if asyncio.iscoroutine(result):
            result = await result
        if not isinstance(result, dict):
            raise OAuthError(f"http_post returned {type(result).__name__}; expected dict")
        return result


def find_free_port(host: str = "127.0.0.1") -> int:
    """Helper for callers that want to pre-pick a port (e.g. for firewall rules)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


__all__ = [
    "OAuthAuthConfig",
    "OAuthError",
    "OAuthFlow",
    "OAuthToken",
    "build_authorize_url",
    "find_free_port",
]
