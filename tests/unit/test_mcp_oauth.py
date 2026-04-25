"""Unit tests for the MCP OAuth flow (S8.2)."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from geny_executor.tools.mcp import (
    FileCredentialStore,
    MemoryCredentialStore,
    OAuthAuthConfig,
    OAuthError,
    OAuthFlow,
    OAuthToken,
    build_authorize_url,
    find_free_port,
    mcp_credential_key,
)


# ── helpers ───────────────────────────────────────────────────────────────


def _config(**overrides) -> OAuthAuthConfig:
    base = dict(
        client_id="cid",
        client_secret="csec",
        authorize_url="https://provider/authorize",
        token_url="https://provider/token",
        scopes=["a", "b"],
    )
    base.update(overrides)
    return OAuthAuthConfig(**base)


class _RecordingHttpPost:
    """Test double for the token-exchange HTTP client."""

    def __init__(self, response: Dict[str, Any] | Exception) -> None:
        self.response = response
        self.calls: List[Tuple[str, Dict[str, str]]] = []

    async def __call__(self, url: str, form: Dict[str, str]) -> Dict[str, Any]:
        self.calls.append((url, dict(form)))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


async def _hit_callback(
    redirect_uri: str,
    *,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Open the redirect URI to simulate the browser callback."""
    params: Dict[str, str] = {}
    if code is not None:
        params["code"] = code
    if state is not None:
        params["state"] = state
    if error is not None:
        params["error"] = error
    async with httpx.AsyncClient(timeout=5.0) as client:
        # Build URL — the redirect_uri already ends in /callback.
        await client.get(redirect_uri, params=params)


def _start_authorize(
    flow: OAuthFlow,
    server_name: str,
    config: OAuthAuthConfig,
    *,
    extra_authorize_params: Optional[Dict[str, str]] = None,
) -> Tuple[asyncio.Task, asyncio.Future]:
    """Kick off authorize() and capture the URL via consent_handler."""
    captured: asyncio.Future = asyncio.get_event_loop().create_future()

    def consent(url: str) -> None:
        if not captured.done():
            captured.set_result(url)

    flow._consent_handler = consent  # type: ignore[attr-defined]
    task = asyncio.create_task(
        flow.authorize(server_name, config, extra_authorize_params=extra_authorize_params)
    )
    return task, captured


# ── OAuthAuthConfig validation ──────────────────────────────────────────


class TestAuthConfigValidation:
    def test_required_fields(self):
        with pytest.raises(ValueError, match="client_id"):
            OAuthAuthConfig(client_id="", client_secret="x", authorize_url="u", token_url="t")
        with pytest.raises(ValueError, match="authorize_url"):
            OAuthAuthConfig(client_id="x", client_secret="x", authorize_url="", token_url="t")
        with pytest.raises(ValueError, match="token_url"):
            OAuthAuthConfig(client_id="x", client_secret="x", authorize_url="u", token_url="")


# ── build_authorize_url ────────────────────────────────────────────────


class TestBuildAuthorizeUrl:
    def test_basic_params(self):
        cfg = _config(authorize_url="https://x/auth")
        url = build_authorize_url(
            cfg, redirect_uri="http://127.0.0.1:9876/callback", state="STATE"
        )
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        assert params["response_type"] == ["code"]
        assert params["client_id"] == ["cid"]
        assert params["redirect_uri"] == ["http://127.0.0.1:9876/callback"]
        assert params["state"] == ["STATE"]
        assert params["scope"] == ["a b"]

    def test_no_scopes_omits_scope(self):
        cfg = _config(scopes=[])
        url = build_authorize_url(
            cfg, redirect_uri="http://127.0.0.1:9876/callback", state="S"
        )
        params = parse_qs(urlparse(url).query)
        assert "scope" not in params

    def test_appends_to_existing_query(self):
        cfg = _config(authorize_url="https://x/auth?prompt=consent")
        url = build_authorize_url(
            cfg, redirect_uri="http://127.0.0.1:9876/callback", state="S"
        )
        # Second separator should be & not ?
        assert url.startswith("https://x/auth?prompt=consent&")
        params = parse_qs(urlparse(url).query)
        assert params["prompt"] == ["consent"]

    def test_extra_params_merged(self):
        cfg = _config()
        url = build_authorize_url(
            cfg,
            redirect_uri="http://127.0.0.1:9876/callback",
            state="S",
            extra_params={"access_type": "offline"},
        )
        params = parse_qs(urlparse(url).query)
        assert params["access_type"] == ["offline"]


# ── OAuthToken ─────────────────────────────────────────────────────────


class TestOAuthToken:
    def test_from_response_minimal(self):
        token = OAuthToken.from_token_response({"access_token": "xx"})
        assert token.access_token == "xx"
        assert token.token_type == "Bearer"
        assert token.refresh_token is None
        assert token.expires_at is None

    def test_from_response_full(self):
        token = OAuthToken.from_token_response(
            {
                "access_token": "tk",
                "token_type": "MAC",
                "refresh_token": "rt",
                "expires_in": 3600,
                "scope": "a b",
                "extra": "ignored-but-kept-in-raw",
            }
        )
        assert token.access_token == "tk"
        assert token.token_type == "MAC"
        assert token.refresh_token == "rt"
        assert token.expires_at is not None
        # within the test second
        assert abs(token.expires_at - (time.time() + 3600)) < 5
        assert token.scope == "a b"
        assert token.raw["extra"] == "ignored-but-kept-in-raw"

    def test_missing_access_token(self):
        with pytest.raises(ValueError, match="access_token"):
            OAuthToken.from_token_response({"token_type": "Bearer"})

    def test_round_trip_json(self):
        original = OAuthToken.from_token_response(
            {"access_token": "tk", "expires_in": 100, "refresh_token": "rt"}
        )
        rehydrated = OAuthToken.from_json(original.to_json())
        assert rehydrated.access_token == "tk"
        assert rehydrated.refresh_token == "rt"
        assert rehydrated.expires_at == original.expires_at

    def test_is_expired_no_expiry(self):
        token = OAuthToken(access_token="x")
        assert token.is_expired() is False

    def test_is_expired_past(self):
        token = OAuthToken(access_token="x", expires_at=time.time() - 100)
        assert token.is_expired() is True

    def test_is_expired_within_leeway(self):
        token = OAuthToken(access_token="x", expires_at=time.time() + 10)
        # default leeway 30s → counts as expired
        assert token.is_expired() is True
        # zero leeway → not expired yet
        assert token.is_expired(leeway_seconds=0) is False


# ── OAuthFlow constructor ──────────────────────────────────────────────


class TestOAuthFlowConstructor:
    def test_default_host_and_port(self):
        flow = OAuthFlow(MemoryCredentialStore())
        assert flow.callback_host == "127.0.0.1"
        assert flow.callback_port == 0

    def test_invalid_port_rejected(self):
        store = MemoryCredentialStore()
        with pytest.raises(ValueError):
            OAuthFlow(store, callback_port=-1)
        with pytest.raises(ValueError):
            OAuthFlow(store, callback_port=70_000)

    def test_invalid_timeout_rejected(self):
        with pytest.raises(ValueError):
            OAuthFlow(MemoryCredentialStore(), timeout_seconds=0)


# ── load / revoke cached token ─────────────────────────────────────────


class TestCachedToken:
    def test_load_returns_none_when_missing(self):
        flow = OAuthFlow(MemoryCredentialStore())
        assert flow.load_cached_token("server") is None

    def test_load_returns_token_when_present(self):
        store = MemoryCredentialStore()
        token = OAuthToken(access_token="cached")
        store.set(mcp_credential_key("srv"), token.to_json())
        flow = OAuthFlow(store)
        loaded = flow.load_cached_token("srv")
        assert loaded is not None and loaded.access_token == "cached"

    def test_load_ignores_corrupt_cache(self):
        store = MemoryCredentialStore()
        store.set(mcp_credential_key("srv"), "{not json")
        flow = OAuthFlow(store)
        assert flow.load_cached_token("srv") is None

    def test_revoke(self):
        store = MemoryCredentialStore()
        store.set(mcp_credential_key("srv"), OAuthToken(access_token="x").to_json())
        flow = OAuthFlow(store)
        assert flow.revoke_cached_token("srv") is True
        assert flow.load_cached_token("srv") is None
        assert flow.revoke_cached_token("srv") is False


# ── authorize() — end-to-end with real local server ────────────────────


class TestAuthorizeFlow:
    @pytest.mark.asyncio
    async def test_happy_path_persists_token(self, tmp_path):
        store = FileCredentialStore(tmp_path / "creds.json")
        http_post = _RecordingHttpPost(
            {"access_token": "AT", "refresh_token": "RT", "expires_in": 3600}
        )
        flow = OAuthFlow(
            store,
            timeout_seconds=5.0,
            http_post=http_post,
        )
        task, captured_url_fut = _start_authorize(flow, "srv", _config())

        # Wait for the URL the flow built.
        url = await asyncio.wait_for(captured_url_fut, timeout=2.0)
        params = parse_qs(urlparse(url).query)
        state = params["state"][0]
        redirect_uri = params["redirect_uri"][0]

        # Simulate the browser hitting the callback.
        await _hit_callback(redirect_uri, code="THE_CODE", state=state)

        token = await asyncio.wait_for(task, timeout=3.0)
        assert token.access_token == "AT"
        assert token.refresh_token == "RT"

        # Token persisted to the file store.
        cached = flow.load_cached_token("srv")
        assert cached is not None and cached.access_token == "AT"

        # Token-exchange call captured the right form.
        assert len(http_post.calls) == 1
        url_called, form = http_post.calls[0]
        assert url_called == "https://provider/token"
        assert form["grant_type"] == "authorization_code"
        assert form["code"] == "THE_CODE"
        assert form["client_id"] == "cid"
        assert form["client_secret"] == "csec"
        assert form["redirect_uri"] == redirect_uri

    @pytest.mark.asyncio
    async def test_state_mismatch_raises(self):
        flow = OAuthFlow(
            MemoryCredentialStore(),
            timeout_seconds=5.0,
            http_post=_RecordingHttpPost({"access_token": "AT"}),
        )
        task, captured = _start_authorize(flow, "srv", _config())
        url = await asyncio.wait_for(captured, timeout=2.0)
        redirect_uri = parse_qs(urlparse(url).query)["redirect_uri"][0]

        # Send wrong state.
        await _hit_callback(redirect_uri, code="X", state="WRONG")
        with pytest.raises(OAuthError, match="state mismatch"):
            await asyncio.wait_for(task, timeout=3.0)

    @pytest.mark.asyncio
    async def test_provider_error_response(self):
        flow = OAuthFlow(
            MemoryCredentialStore(),
            timeout_seconds=5.0,
            http_post=_RecordingHttpPost({"access_token": "AT"}),
        )
        task, captured = _start_authorize(flow, "srv", _config())
        url = await asyncio.wait_for(captured, timeout=2.0)
        redirect_uri = parse_qs(urlparse(url).query)["redirect_uri"][0]

        await _hit_callback(redirect_uri, error="access_denied")
        with pytest.raises(OAuthError, match="access_denied"):
            await asyncio.wait_for(task, timeout=3.0)

    @pytest.mark.asyncio
    async def test_token_exchange_failure_raises(self):
        flow = OAuthFlow(
            MemoryCredentialStore(),
            timeout_seconds=5.0,
            http_post=_RecordingHttpPost(OAuthError("server returned 500")),
        )
        task, captured = _start_authorize(flow, "srv", _config())
        url = await asyncio.wait_for(captured, timeout=2.0)
        params = parse_qs(urlparse(url).query)
        redirect_uri = params["redirect_uri"][0]
        state = params["state"][0]

        await _hit_callback(redirect_uri, code="X", state=state)
        with pytest.raises(OAuthError, match="500"):
            await asyncio.wait_for(task, timeout=3.0)

    @pytest.mark.asyncio
    async def test_callback_timeout(self):
        flow = OAuthFlow(
            MemoryCredentialStore(),
            timeout_seconds=0.5,
            http_post=_RecordingHttpPost({"access_token": "AT"}),
        )
        task, _ = _start_authorize(flow, "srv", _config())
        with pytest.raises(OAuthError, match="timed out"):
            await asyncio.wait_for(task, timeout=3.0)

    @pytest.mark.asyncio
    async def test_authorize_rejects_blank_server_name(self):
        flow = OAuthFlow(MemoryCredentialStore())
        with pytest.raises(ValueError, match="server_name"):
            await flow.authorize("", _config())

    @pytest.mark.asyncio
    async def test_extra_authorize_params_propagate(self):
        flow = OAuthFlow(
            MemoryCredentialStore(),
            timeout_seconds=5.0,
            http_post=_RecordingHttpPost({"access_token": "AT"}),
        )
        task, captured = _start_authorize(
            flow, "srv", _config(), extra_authorize_params={"prompt": "consent"}
        )
        url = await asyncio.wait_for(captured, timeout=2.0)
        params = parse_qs(urlparse(url).query)
        assert params["prompt"] == ["consent"]
        # Cancel the task — we don't need the full flow here.
        task.cancel()
        with pytest.raises((asyncio.CancelledError, OAuthError)):
            await task


# ── http_post return-shape validation ──────────────────────────────────


class TestHttpPostReturnShape:
    @pytest.mark.asyncio
    async def test_sync_callable_supported(self):
        """A non-coroutine http_post returning a dict should also work."""

        def sync_post(url: str, form: Dict[str, str]) -> Dict[str, Any]:
            return {"access_token": "SYNC"}

        flow = OAuthFlow(
            MemoryCredentialStore(),
            timeout_seconds=5.0,
            http_post=sync_post,
        )
        task, captured = _start_authorize(flow, "srv", _config())
        url = await asyncio.wait_for(captured, timeout=2.0)
        params = parse_qs(urlparse(url).query)
        await _hit_callback(
            params["redirect_uri"][0], code="X", state=params["state"][0]
        )
        token = await asyncio.wait_for(task, timeout=3.0)
        assert token.access_token == "SYNC"

    @pytest.mark.asyncio
    async def test_non_dict_return_raises(self):
        async def bad_post(url, form):
            return "not a dict"

        flow = OAuthFlow(
            MemoryCredentialStore(),
            timeout_seconds=5.0,
            http_post=bad_post,
        )
        task, captured = _start_authorize(flow, "srv", _config())
        url = await asyncio.wait_for(captured, timeout=2.0)
        params = parse_qs(urlparse(url).query)
        await _hit_callback(
            params["redirect_uri"][0], code="X", state=params["state"][0]
        )
        with pytest.raises(OAuthError, match="expected dict"):
            await asyncio.wait_for(task, timeout=3.0)


# ── helpers ─────────────────────────────────────────────────────────────


class TestFindFreePort:
    def test_returns_int_in_range(self):
        port = find_free_port()
        assert isinstance(port, int)
        assert 1024 <= port <= 65_535
