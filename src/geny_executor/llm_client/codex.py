"""ChatGPT (Codex) subscription as a native tool-calling backend.

Signing in to Codex with a ChatGPT plan yields OAuth tokens for
`https://chatgpt.com/backend-api/codex`, which speaks the OpenAI Responses
API. The wire itself lives in `responses_wire` — an API key and a ChatGPT
subscription send byte-identical requests to it. What is left here is the
only part that differs: who the caller is.

Tokens live in the host's secret store (per account). The host refreshes
them before a turn; this client refreshes too when the backend says 401
mid-session, and reports the rotated tokens back through `notify` so the
host persists them — a refresh token is single-use, so losing the new one
would log the account out.
"""
from __future__ import annotations

import base64
import json
import time
from typing import Any, Dict, Optional

import httpx

from geny_executor.core.errors import APIError, ErrorCategory
from geny_executor.llm_client._failover import Notify
from geny_executor.llm_client.responses_wire import (
    ResponsesClient,
    to_input,
    to_tools,
)

DEFAULT_BASE_URL = "https://chatgpt.com/backend-api/codex"
TOKEN_URL = "https://auth.openai.com/oauth/token"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
ORIGINATOR = "geny_executor"
USER_AGENT = "geny-executor (+https://github.com/CocoRoF/geny-executor)"

__all__ = ["CodexResponsesClient", "jwt_claims", "account_id_from", "expires_at",
           "to_input", "to_tools", "DEFAULT_BASE_URL", "TOKEN_URL", "CLIENT_ID"]


def jwt_claims(token: str) -> dict[str, Any]:
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload.encode()))
    except Exception:
        return {}


def account_id_from(tokens: dict[str, Any]) -> str:
    if tokens.get("account_id"):
        return str(tokens["account_id"])
    for key in ("id_token", "access_token"):
        claims = jwt_claims(str(tokens.get(key) or ""))
        auth = claims.get("https://api.openai.com/auth") or {}
        if isinstance(auth, dict) and auth.get("chatgpt_account_id"):
            return str(auth["chatgpt_account_id"])
    return ""


def expires_at(token: str) -> float:
    exp = jwt_claims(token).get("exp")
    return float(exp) if isinstance(exp, (int, float)) else 0.0


class CodexResponsesClient(ResponsesClient):
    """The Responses wire, driven by a ChatGPT subscription's OAuth tokens."""

    provider = "geny_codex"
    base_url_default = DEFAULT_BASE_URL
    label = "Codex"

    def __init__(
        self,
        api_key: str = "",
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        *,
        account_id: str = "",
        account_label: str = "",
        tokens: Optional[dict[str, Any]] = None,
        effort: Optional[str] = None,
        timeout_s: float = 600.0,
        notify: Optional[Notify] = None,
        session_id: Optional[str] = None,
        event_sink: Any = None,
        transport: Any = None,
        **_ignored: Any,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, default_headers=default_headers,
                         account_id=account_id, account_label=account_label, effort=effort,
                         timeout_s=timeout_s, notify=notify, session_id=session_id,
                         event_sink=event_sink, transport=transport)
        self._tokens = dict(tokens or {})
        if api_key and not self._tokens.get("access_token"):
            self._tokens["access_token"] = api_key

    # ── who we are ───────────────────────────────────────────────────

    def _auth_headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._tokens.get('access_token', '')}",
            "originator": ORIGINATOR,
            "User-Agent": USER_AGENT,
            "session_id": self._session_id,
        }
        account = account_id_from(self._tokens)
        if account:
            headers["ChatGPT-Account-Id"] = account
        return headers

    async def _prepare(self) -> None:
        if not self._tokens.get("access_token"):
            raise APIError("This Codex account has no access token — sign in again.",
                           category=ErrorCategory.AUTH)
        exp = expires_at(str(self._tokens.get("access_token")))
        if exp and exp - time.time() < 120:
            await self._renew()

    async def _renew(self) -> bool:
        refresh = self._tokens.get("refresh_token")
        if not refresh:
            return False
        try:
            async with self._client() as client:
                resp = await client.post(
                    TOKEN_URL,
                    data={"grant_type": "refresh_token", "refresh_token": refresh, "client_id": CLIENT_ID},
                    headers={"Accept": "application/json", "User-Agent": USER_AGENT},
                )
        except httpx.HTTPError:
            return False
        if resp.status_code != 200:
            return False
        data = resp.json()
        if not data.get("access_token"):
            return False
        self._tokens["access_token"] = data["access_token"]
        # The refresh token is single-use: the rotated one must reach the host
        # or this account is signed out at the next turn.
        if data.get("refresh_token"):
            self._tokens["refresh_token"] = data["refresh_token"]
        if data.get("id_token"):
            self._tokens["id_token"] = data["id_token"]
        self._tokens["last_refresh"] = time.time()
        self._notify_host({"kind": "tokens", "tokens": self._tokens})
        return True

    # Kept for hosts and tests that reached for the old name.
    _refresh = _renew
