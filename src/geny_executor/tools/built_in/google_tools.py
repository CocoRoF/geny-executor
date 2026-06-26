"""Google Workspace tools — native Gmail / Calendar / Drive / Tasks.

A bundle of first-party tools that let an agent operate a connected
Google account directly through the Google REST APIs. They sit beside
``WebFetch`` / ``WebSearch`` in the built-in set and follow the same
conventions: async ``execute``, ``httpx.AsyncClient`` for I/O, and a
clean LLM-friendly string in ``ToolResult.content`` (``is_error=True``
on failure).

OAuth credentials are injected by the host through
``ctx.extras["google"]``::

    {
        "access_token":  "...",   # required
        "refresh_token": "...",   # optional but needed for auto-refresh
        "client_id":     "...",   # optional, for refresh
        "client_secret": "...",   # optional, for refresh
    }

If that bag is missing (or carries no ``access_token``) the tool returns
a connected-account error instead of raising. On an HTTP 401 the shared
``_GoogleClient`` transparently refreshes the access token once (when a
``refresh_token`` + ``client_id`` + ``client_secret`` are present) and
retries the original request a single time.

Gating: every tool advertises ``required_config_keys() ->
["feature:google_connected"]`` so the host can hide the whole bundle
until the user has linked their Google account. This method is not part
of the ``Tool`` ABC yet — the host reads it via ``getattr`` — so it is
declared on each tool defensively; if the base later grows the method
these simply override it.

Dependencies: ``httpx`` (already a hard dependency, used by WebFetch) +
the standard library only. No new packages are introduced.
"""

from __future__ import annotations

import base64
import logging
from email.message import EmailMessage
from typing import Any, Dict, List, Optional

import httpx

from geny_executor.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult

logger = logging.getLogger(__name__)

# ── Endpoints ────────────────────────────────────────────────────────
_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GMAIL = "https://gmail.googleapis.com/gmail/v1"
_CALENDAR = "https://www.googleapis.com/calendar/v3"
_DRIVE = "https://www.googleapis.com/drive/v3"
_TASKS = "https://tasks.googleapis.com/tasks/v1"

_DEFAULT_TIMEOUT = 30.0
_BODY_TRUNCATE = 10_000  # cap on Drive/Gmail body text returned to the LLM
_GOOGLE_FEATURE_KEY = "feature:google_connected"


class GoogleNotConnectedError(Exception):
    """Raised when no usable Google OAuth credentials are available."""


# ─────────────────────────────────────────────────────────────────────
# Shared OAuth REST client
# ─────────────────────────────────────────────────────────────────────


class _GoogleClient:
    """Thin authenticated wrapper over the Google REST APIs.

    Constructed from the ``ctx.extras["google"]`` credential bag. Sends
    ``Authorization: Bearer <access_token>`` on every request and, on a
    401, refreshes the access token once (if a refresh token + client
    credentials are present) before retrying the original request.
    """

    def __init__(self, creds: Optional[Dict[str, Any]], timeout: float = _DEFAULT_TIMEOUT):
        if not isinstance(creds, dict):
            raise GoogleNotConnectedError(
                "Google account is not connected (no credentials supplied)."
            )
        self._access_token: str = str(creds.get("access_token") or "").strip()
        if not self._access_token:
            raise GoogleNotConnectedError(
                "Google account is not connected (no access token)."
            )
        self._refresh_token: Optional[str] = creds.get("refresh_token") or None
        self._client_id: Optional[str] = creds.get("client_id") or None
        self._client_secret: Optional[str] = creds.get("client_secret") or None
        self._timeout = timeout

    @classmethod
    def from_context(cls, context: ToolContext) -> "_GoogleClient":
        """Build a client from ``context.extras['google']``."""
        extras = getattr(context, "extras", None) or {}
        return cls(extras.get("google"))

    @property
    def _auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._access_token}"}

    async def _refresh(self) -> bool:
        """Refresh the access token in place. Returns True on success."""
        if not (self._refresh_token and self._client_id and self._client_secret):
            return False
        payload = {
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "refresh_token": self._refresh_token,
            "grant_type": "refresh_token",
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(_TOKEN_URL, data=payload)
        except httpx.HTTPError as exc:
            logger.warning("Google token refresh request failed: %s", exc)
            return False
        if resp.status_code >= 400:
            logger.warning(
                "Google token refresh rejected: HTTP %s %s",
                resp.status_code,
                resp.text[:300],
            )
            return False
        token = (resp.json() or {}).get("access_token")
        if not token:
            return False
        self._access_token = str(token)
        return True

    async def request(self, method: str, url: str, **kw: Any) -> httpx.Response:
        """Perform an authenticated request, refreshing once on 401.

        ``**kw`` is forwarded to ``httpx.AsyncClient.request`` (params,
        json, data, headers, ...). Caller-supplied headers are merged
        over the Bearer auth header.
        """
        headers = {**self._auth_headers, **(kw.pop("headers", None) or {})}
        timeout = kw.pop("timeout", self._timeout)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.request(method, url, headers=headers, **kw)
            if resp.status_code == 401 and await self._refresh():
                headers["Authorization"] = f"Bearer {self._access_token}"
                resp = await client.request(method, url, headers=headers, **kw)
            return resp


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _http_error(resp: httpx.Response, what: str) -> str:
    """Build a compact one-line message from a failed Google response."""
    detail = ""
    try:
        body = resp.json()
        err = body.get("error") if isinstance(body, dict) else None
        if isinstance(err, dict):
            detail = str(err.get("message") or "")
        elif isinstance(err, str):
            detail = err
    except Exception:
        detail = (resp.text or "")[:200]
    suffix = f": {detail}" if detail else ""
    return f"Google API error ({what}): HTTP {resp.status_code}{suffix}"


def _header(payload: Dict[str, Any], name: str) -> str:
    """Return a header value from a Gmail message payload (case-insensitive)."""
    for h in (payload or {}).get("headers", []) or []:
        if str(h.get("name", "")).lower() == name.lower():
            return str(h.get("value", ""))
    return ""


def _decode_b64url(data: str) -> str:
    """Decode a Gmail base64url body part to text (best effort)."""
    if not data:
        return ""
    try:
        raw = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_plain_body(payload: Dict[str, Any]) -> str:
    """Walk a Gmail MIME tree and return the best plain-text body.

    Prefers ``text/plain``; falls back to ``text/html`` (returned as raw
    HTML if no plain part exists) and finally to a single-part body.
    """
    if not isinstance(payload, dict):
        return ""

    plain: List[str] = []
    html_parts: List[str] = []

    def _walk(part: Dict[str, Any]) -> None:
        mime = str(part.get("mimeType", ""))
        body = part.get("body") or {}
        data = body.get("data")
        if mime == "text/plain" and data:
            plain.append(_decode_b64url(data))
        elif mime == "text/html" and data:
            html_parts.append(_decode_b64url(data))
        for sub in part.get("parts", []) or []:
            _walk(sub)

    _walk(payload)
    if plain:
        return "\n".join(p for p in plain if p).strip()
    if html_parts:
        return "\n".join(h for h in html_parts if h).strip()
    # Single-part message with the body directly on the top payload.
    return _decode_b64url((payload.get("body") or {}).get("data", "")).strip()


def _truncate(text: str, limit: int = _BODY_TRUNCATE) -> str:
    if len(text) > limit:
        return text[:limit] + f"\n\n[... truncated at {limit} chars]"
    return text


# ─────────────────────────────────────────────────────────────────────
# Base class for the Google tool family
# ─────────────────────────────────────────────────────────────────────


class _GoogleTool(Tool):
    """Common scaffolding for Google Workspace tools.

    Subclasses implement ``name`` / ``description`` / ``input_schema``
    and an async ``_run(self, input, client)`` that talks to Google. The
    base ``execute`` resolves the client from ``context.extras`` and
    funnels every failure into ``ToolResult(is_error=True)`` so nothing
    ever propagates out of ``execute``.
    """

    def required_config_keys(self) -> List[str]:
        """Host gate — hide the tool until a Google account is linked.

        Not part of the ``Tool`` ABC yet; the host reads it via
        ``getattr``. Declared here so the whole family is gated.
        """
        return [_GOOGLE_FEATURE_KEY]

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        # Read-only by default; mutating tools override.
        return ToolCapabilities(
            concurrency_safe=True,
            read_only=True,
            idempotent=False,
            network_egress=True,
        )

    async def _run(self, input: Dict[str, Any], client: _GoogleClient) -> ToolResult:
        raise NotImplementedError

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            client = _GoogleClient.from_context(context)
        except GoogleNotConnectedError as exc:
            return ToolResult(content=str(exc), is_error=True)
        try:
            return await self._run(input or {}, client)
        except GoogleNotConnectedError as exc:
            return ToolResult(content=str(exc), is_error=True)
        except httpx.TimeoutException:
            return ToolResult(content=f"{self.name}: request to Google timed out.", is_error=True)
        except httpx.HTTPError as exc:
            return ToolResult(content=f"{self.name}: network error talking to Google: {exc}", is_error=True)
        except Exception as exc:  # noqa: BLE001 — never let execute raise
            logger.exception("%s failed", self.name)
            return ToolResult(content=f"{self.name} failed: {exc}", is_error=True)


# ─────────────────────────────────────────────────────────────────────
# Gmail
# ─────────────────────────────────────────────────────────────────────


class GmailSearchTool(_GoogleTool):
    """Search the user's Gmail mailbox with a Gmail query string."""

    @property
    def name(self) -> str:
        return "gmail_search"

    @property
    def description(self) -> str:
        return (
            "Search the connected Gmail account. 'q' uses Gmail search "
            "syntax (e.g. 'from:alice is:unread newer_than:7d'). Returns a "
            "list of matching messages with id, from, subject, date, and a "
            "snippet. Use gmail_read with a message id for the full body."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "q": {
                    "type": "string",
                    "description": "Gmail search query (Gmail search syntax).",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of messages to return. Default 10.",
                    "exclusiveMinimum": 0,
                },
            },
            "required": ["q"],
        }

    async def _run(self, input: Dict[str, Any], client: _GoogleClient) -> ToolResult:
        q = (input.get("q") or "").strip()
        if not q:
            return ToolResult(content="gmail_search: 'q' must not be empty.", is_error=True)
        max_results = max(1, min(50, int(input.get("max_results", 10))))

        resp = await client.request(
            "GET",
            f"{_GMAIL}/users/me/messages",
            params={"q": q, "maxResults": max_results},
        )
        if resp.status_code >= 400:
            return ToolResult(content=_http_error(resp, "gmail_search"), is_error=True)

        messages = (resp.json() or {}).get("messages", []) or []
        if not messages:
            return ToolResult(content=f"No Gmail messages match {q!r}.")

        lines: List[str] = [f"Gmail results for {q!r} ({len(messages)} of max {max_results}):", ""]
        for m in messages:
            mid = m.get("id")
            meta = await client.request(
                "GET",
                f"{_GMAIL}/users/me/messages/{mid}",
                params={
                    "format": "metadata",
                    "metadataHeaders": ["From", "Subject", "Date"],
                },
            )
            if meta.status_code >= 400:
                lines.append(f"- {mid}: (could not load headers)")
                continue
            data = meta.json() or {}
            payload = data.get("payload") or {}
            lines.append(
                f"- id: {mid}\n"
                f"  from: {_header(payload, 'From') or '(unknown)'}\n"
                f"  subject: {_header(payload, 'Subject') or '(no subject)'}\n"
                f"  date: {_header(payload, 'Date') or '(unknown)'}\n"
                f"  snippet: {(data.get('snippet') or '').strip()}"
            )
        return ToolResult(content="\n".join(lines), metadata={"count": len(messages)})


class GmailReadTool(_GoogleTool):
    """Read a single Gmail message's headers and plain-text body."""

    @property
    def name(self) -> str:
        return "gmail_read"

    @property
    def description(self) -> str:
        return (
            "Read a single Gmail message by its id (from gmail_search). "
            "Returns From / To / Subject / Date and the decoded plain-text "
            "body."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message_id": {
                    "type": "string",
                    "description": "Gmail message id to read.",
                },
            },
            "required": ["message_id"],
        }

    async def _run(self, input: Dict[str, Any], client: _GoogleClient) -> ToolResult:
        mid = (input.get("message_id") or "").strip()
        if not mid:
            return ToolResult(content="gmail_read: 'message_id' is required.", is_error=True)

        resp = await client.request(
            "GET",
            f"{_GMAIL}/users/me/messages/{mid}",
            params={"format": "full"},
        )
        if resp.status_code >= 400:
            return ToolResult(content=_http_error(resp, "gmail_read"), is_error=True)

        data = resp.json() or {}
        payload = data.get("payload") or {}
        body = _truncate(_extract_plain_body(payload)) or "(empty body)"
        head = (
            f"From: {_header(payload, 'From') or '(unknown)'}\n"
            f"To: {_header(payload, 'To') or '(unknown)'}\n"
            f"Subject: {_header(payload, 'Subject') or '(no subject)'}\n"
            f"Date: {_header(payload, 'Date') or '(unknown)'}"
        )
        return ToolResult(content=f"{head}\n\n{body}", metadata={"message_id": mid})


class GmailSendTool(_GoogleTool):
    """Send a plain-text email from the connected Gmail account."""

    @property
    def name(self) -> str:
        return "gmail_send"

    @property
    def description(self) -> str:
        return (
            "Send a plain-text email from the connected Gmail account. "
            "Provide 'to', 'subject', and 'body'. This sends a real email "
            "immediately — confirm the recipient and content first."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address."},
                "subject": {"type": "string", "description": "Email subject line."},
                "body": {"type": "string", "description": "Plain-text email body."},
            },
            "required": ["to", "subject", "body"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(
            concurrency_safe=False,
            read_only=False,
            idempotent=False,
            network_egress=True,
        )

    async def _run(self, input: Dict[str, Any], client: _GoogleClient) -> ToolResult:
        to = (input.get("to") or "").strip()
        subject = input.get("subject") or ""
        body = input.get("body") or ""
        if not to:
            return ToolResult(content="gmail_send: 'to' is required.", is_error=True)

        msg = EmailMessage()
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(body)
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")

        resp = await client.request(
            "POST",
            f"{_GMAIL}/users/me/messages/send",
            json={"raw": raw},
        )
        if resp.status_code >= 400:
            return ToolResult(content=_http_error(resp, "gmail_send"), is_error=True)

        data = resp.json() or {}
        return ToolResult(
            content=f"Email sent to {to} (subject: {subject!r}). Message id: {data.get('id', '?')}.",
            metadata={"message_id": data.get("id")},
        )


# ─────────────────────────────────────────────────────────────────────
# Calendar
# ─────────────────────────────────────────────────────────────────────


def _event_when(event: Dict[str, Any], key: str) -> str:
    """Render a Calendar event 'start'/'end' (dateTime or all-day date)."""
    when = event.get(key) or {}
    return str(when.get("dateTime") or when.get("date") or "")


class CalendarListEventsTool(_GoogleTool):
    """List upcoming events from a Google Calendar."""

    @property
    def name(self) -> str:
        return "calendar_list_events"

    @property
    def description(self) -> str:
        return (
            "List events from a Google Calendar, ordered by start time. "
            "Optionally bound the window with RFC3339 'time_min' / "
            "'time_max'. Returns summary, start, end, and location per "
            "event. Defaults to the user's primary calendar."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "time_min": {
                    "type": "string",
                    "description": "Lower bound (RFC3339, e.g. 2026-06-26T00:00:00Z). Optional.",
                },
                "time_max": {
                    "type": "string",
                    "description": "Upper bound (RFC3339). Optional.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of events. Default 10.",
                    "exclusiveMinimum": 0,
                },
                "calendar_id": {
                    "type": "string",
                    "description": "Calendar id. Default 'primary'.",
                },
            },
            "required": [],
        }

    async def _run(self, input: Dict[str, Any], client: _GoogleClient) -> ToolResult:
        calendar_id = (input.get("calendar_id") or "primary").strip()
        max_results = max(1, min(100, int(input.get("max_results", 10))))
        params: Dict[str, Any] = {
            "singleEvents": "true",
            "orderBy": "startTime",
            "maxResults": max_results,
        }
        if input.get("time_min"):
            params["timeMin"] = input["time_min"]
        if input.get("time_max"):
            params["timeMax"] = input["time_max"]

        resp = await client.request(
            "GET",
            f"{_CALENDAR}/calendars/{calendar_id}/events",
            params=params,
        )
        if resp.status_code >= 400:
            return ToolResult(content=_http_error(resp, "calendar_list_events"), is_error=True)

        events = (resp.json() or {}).get("items", []) or []
        if not events:
            return ToolResult(content=f"No events found on calendar {calendar_id!r}.")

        lines = [f"Events on {calendar_id!r} ({len(events)}):", ""]
        for ev in events:
            line = (
                f"- {ev.get('summary') or '(no title)'}\n"
                f"  start: {_event_when(ev, 'start') or '(unknown)'}\n"
                f"  end: {_event_when(ev, 'end') or '(unknown)'}"
            )
            if ev.get("location"):
                line += f"\n  location: {ev['location']}"
            if ev.get("id"):
                line += f"\n  id: {ev['id']}"
            lines.append(line)
        return ToolResult(content="\n".join(lines), metadata={"count": len(events)})


class CalendarCreateEventTool(_GoogleTool):
    """Create an event on a Google Calendar."""

    @property
    def name(self) -> str:
        return "calendar_create_event"

    @property
    def description(self) -> str:
        return (
            "Create an event on a Google Calendar. 'start' and 'end' accept "
            "either an RFC3339 datetime (e.g. 2026-06-26T15:00:00-07:00, a "
            "timed event) or a plain date (e.g. 2026-06-26, an all-day "
            "event). Optionally set description and location."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Event title."},
                "start": {
                    "type": "string",
                    "description": "Start: RFC3339 datetime or YYYY-MM-DD date.",
                },
                "end": {
                    "type": "string",
                    "description": "End: RFC3339 datetime or YYYY-MM-DD date.",
                },
                "description": {"type": "string", "description": "Event description. Optional."},
                "location": {"type": "string", "description": "Event location. Optional."},
                "calendar_id": {
                    "type": "string",
                    "description": "Calendar id. Default 'primary'.",
                },
            },
            "required": ["summary", "start", "end"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(
            concurrency_safe=False,
            read_only=False,
            idempotent=False,
            network_egress=True,
        )

    @staticmethod
    def _when(value: str) -> Dict[str, str]:
        """Map a start/end string to a Calendar EventDateTime object.

        A bare 'YYYY-MM-DD' is treated as an all-day 'date'; anything
        else is treated as an RFC3339 'dateTime'.
        """
        v = (value or "").strip()
        if len(v) == 10 and v.count("-") == 2 and "T" not in v:
            return {"date": v}
        return {"dateTime": v}

    async def _run(self, input: Dict[str, Any], client: _GoogleClient) -> ToolResult:
        summary = (input.get("summary") or "").strip()
        start = (input.get("start") or "").strip()
        end = (input.get("end") or "").strip()
        if not (summary and start and end):
            return ToolResult(
                content="calendar_create_event: 'summary', 'start', and 'end' are required.",
                is_error=True,
            )
        calendar_id = (input.get("calendar_id") or "primary").strip()

        event: Dict[str, Any] = {
            "summary": summary,
            "start": self._when(start),
            "end": self._when(end),
        }
        if input.get("description"):
            event["description"] = input["description"]
        if input.get("location"):
            event["location"] = input["location"]

        resp = await client.request(
            "POST",
            f"{_CALENDAR}/calendars/{calendar_id}/events",
            json=event,
        )
        if resp.status_code >= 400:
            return ToolResult(content=_http_error(resp, "calendar_create_event"), is_error=True)

        data = resp.json() or {}
        link = data.get("htmlLink") or ""
        tail = f"\nLink: {link}" if link else ""
        return ToolResult(
            content=f"Event {summary!r} created on {calendar_id!r} (id: {data.get('id', '?')}).{tail}",
            metadata={"event_id": data.get("id"), "html_link": link},
        )


# ─────────────────────────────────────────────────────────────────────
# Drive
# ─────────────────────────────────────────────────────────────────────


class DriveSearchTool(_GoogleTool):
    """Search files in the user's Google Drive."""

    @property
    def name(self) -> str:
        return "drive_search"

    @property
    def description(self) -> str:
        return (
            "Search Google Drive files. 'q' uses Drive query syntax (e.g. "
            "\"name contains 'budget'\" or \"mimeType='application/pdf'\"). "
            "Returns name, type, modified time, and a web link per file. "
            "Use drive_read with a file id to read its contents."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "q": {
                    "type": "string",
                    "description": "Drive search query (Drive query syntax).",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of files. Default 10.",
                    "exclusiveMinimum": 0,
                },
            },
            "required": ["q"],
        }

    async def _run(self, input: Dict[str, Any], client: _GoogleClient) -> ToolResult:
        q = (input.get("q") or "").strip()
        if not q:
            return ToolResult(content="drive_search: 'q' must not be empty.", is_error=True)
        max_results = max(1, min(100, int(input.get("max_results", 10))))

        resp = await client.request(
            "GET",
            f"{_DRIVE}/files",
            params={
                "q": q,
                "pageSize": max_results,
                "fields": "files(id,name,mimeType,modifiedTime,webViewLink)",
            },
        )
        if resp.status_code >= 400:
            return ToolResult(content=_http_error(resp, "drive_search"), is_error=True)

        files = (resp.json() or {}).get("files", []) or []
        if not files:
            return ToolResult(content=f"No Drive files match {q!r}.")

        lines = [f"Drive results for {q!r} ({len(files)}):", ""]
        for f in files:
            lines.append(
                f"- {f.get('name') or '(unnamed)'}\n"
                f"  id: {f.get('id', '?')}\n"
                f"  type: {f.get('mimeType', '?')}\n"
                f"  modified: {f.get('modifiedTime', '?')}\n"
                f"  link: {f.get('webViewLink', '(none)')}"
            )
        return ToolResult(content="\n".join(lines), metadata={"count": len(files)})


class DriveReadTool(_GoogleTool):
    """Read the text contents of a Google Drive file."""

    @property
    def name(self) -> str:
        return "drive_read"

    @property
    def description(self) -> str:
        return (
            "Read a Google Drive file's text by id (from drive_search). "
            "Google Docs are exported to plain text; other text files are "
            "downloaded directly. Output is truncated to roughly 10k chars."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_id": {"type": "string", "description": "Drive file id to read."},
            },
            "required": ["file_id"],
        }

    async def _run(self, input: Dict[str, Any], client: _GoogleClient) -> ToolResult:
        file_id = (input.get("file_id") or "").strip()
        if not file_id:
            return ToolResult(content="drive_read: 'file_id' is required.", is_error=True)

        meta = await client.request(
            "GET",
            f"{_DRIVE}/files/{file_id}",
            params={"fields": "id,name,mimeType"},
        )
        if meta.status_code >= 400:
            return ToolResult(content=_http_error(meta, "drive_read"), is_error=True)

        info = meta.json() or {}
        name = info.get("name") or file_id
        mime = str(info.get("mimeType") or "")

        if mime.startswith("application/vnd.google-apps."):
            # Native Google editor file — must be exported. Only Docs /
            # Slides / scripts export cleanly to text/plain.
            if mime not in (
                "application/vnd.google-apps.document",
                "application/vnd.google-apps.presentation",
                "application/vnd.google-apps.script",
            ):
                return ToolResult(
                    content=(
                        f"drive_read: {name!r} is a {mime} file, which cannot be "
                        f"exported to plain text."
                    ),
                    is_error=True,
                )
            resp = await client.request(
                "GET",
                f"{_DRIVE}/files/{file_id}/export",
                params={"mimeType": "text/plain"},
            )
        else:
            resp = await client.request(
                "GET",
                f"{_DRIVE}/files/{file_id}",
                params={"alt": "media"},
            )

        if resp.status_code >= 400:
            return ToolResult(content=_http_error(resp, "drive_read"), is_error=True)

        text = _truncate(resp.text or "")
        return ToolResult(
            content=f"File: {name} ({mime})\n\n{text or '(empty)'}",
            metadata={"file_id": file_id, "mime_type": mime},
        )


# ─────────────────────────────────────────────────────────────────────
# Tasks
# ─────────────────────────────────────────────────────────────────────


async def _resolve_tasklist(client: _GoogleClient, tasklist: Optional[str]) -> Optional[str]:
    """Return a task-list id. ``None``/empty → the user's default list."""
    tl = (tasklist or "").strip()
    if tl:
        return tl
    resp = await client.request("GET", f"{_TASKS}/users/@me/lists", params={"maxResults": 1})
    if resp.status_code >= 400:
        return None
    items = (resp.json() or {}).get("items", []) or []
    return items[0].get("id") if items else None


class TasksListTool(_GoogleTool):
    """List tasks from a Google Tasks list."""

    @property
    def name(self) -> str:
        return "tasks_list"

    @property
    def description(self) -> str:
        return (
            "List tasks from a Google Tasks list (defaults to the user's "
            "default list). Returns title, status, and due date per task."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tasklist": {
                    "type": "string",
                    "description": "Task list id. Defaults to the user's default list.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of tasks. Default 20.",
                    "exclusiveMinimum": 0,
                },
            },
            "required": [],
        }

    async def _run(self, input: Dict[str, Any], client: _GoogleClient) -> ToolResult:
        max_results = max(1, min(100, int(input.get("max_results", 20))))
        list_id = await _resolve_tasklist(client, input.get("tasklist"))
        if not list_id:
            return ToolResult(content="tasks_list: no task list available.", is_error=True)

        resp = await client.request(
            "GET",
            f"{_TASKS}/lists/{list_id}/tasks",
            params={"maxResults": max_results, "showCompleted": "true"},
        )
        if resp.status_code >= 400:
            return ToolResult(content=_http_error(resp, "tasks_list"), is_error=True)

        tasks = (resp.json() or {}).get("items", []) or []
        if not tasks:
            return ToolResult(content=f"No tasks in list {list_id!r}.")

        lines = [f"Tasks in {list_id!r} ({len(tasks)}):", ""]
        for t in tasks:
            line = (
                f"- {t.get('title') or '(untitled)'}\n"
                f"  status: {t.get('status', 'unknown')}"
            )
            if t.get("due"):
                line += f"\n  due: {t['due']}"
            if t.get("id"):
                line += f"\n  id: {t['id']}"
            lines.append(line)
        return ToolResult(
            content="\n".join(lines),
            metadata={"count": len(tasks), "tasklist": list_id},
        )


class TasksAddTool(_GoogleTool):
    """Add a task to a Google Tasks list."""

    @property
    def name(self) -> str:
        return "tasks_add"

    @property
    def description(self) -> str:
        return (
            "Add a task to a Google Tasks list (defaults to the user's "
            "default list). Optionally set notes and an RFC3339 due date."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Task title."},
                "notes": {"type": "string", "description": "Task notes. Optional."},
                "due": {
                    "type": "string",
                    "description": "Due date (RFC3339, e.g. 2026-06-30T00:00:00Z). Optional.",
                },
                "tasklist": {
                    "type": "string",
                    "description": "Task list id. Defaults to the user's default list.",
                },
            },
            "required": ["title"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(
            concurrency_safe=False,
            read_only=False,
            idempotent=False,
            network_egress=True,
        )

    async def _run(self, input: Dict[str, Any], client: _GoogleClient) -> ToolResult:
        title = (input.get("title") or "").strip()
        if not title:
            return ToolResult(content="tasks_add: 'title' is required.", is_error=True)
        list_id = await _resolve_tasklist(client, input.get("tasklist"))
        if not list_id:
            return ToolResult(content="tasks_add: no task list available.", is_error=True)

        task: Dict[str, Any] = {"title": title}
        if input.get("notes"):
            task["notes"] = input["notes"]
        if input.get("due"):
            task["due"] = input["due"]

        resp = await client.request("POST", f"{_TASKS}/lists/{list_id}/tasks", json=task)
        if resp.status_code >= 400:
            return ToolResult(content=_http_error(resp, "tasks_add"), is_error=True)

        data = resp.json() or {}
        return ToolResult(
            content=f"Task {title!r} added to list {list_id!r} (id: {data.get('id', '?')}).",
            metadata={"task_id": data.get("id"), "tasklist": list_id},
        )


# ─────────────────────────────────────────────────────────────────────
# Registration exports
# ─────────────────────────────────────────────────────────────────────

GOOGLE_TOOL_CLASSES: Dict[str, type] = {
    "gmail_search": GmailSearchTool,
    "gmail_read": GmailReadTool,
    "gmail_send": GmailSendTool,
    "calendar_list_events": CalendarListEventsTool,
    "calendar_create_event": CalendarCreateEventTool,
    "drive_search": DriveSearchTool,
    "drive_read": DriveReadTool,
    "tasks_list": TasksListTool,
    "tasks_add": TasksAddTool,
}

GOOGLE_TOOLS: list = [cls() for cls in GOOGLE_TOOL_CLASSES.values()]

__all__ = [
    "GoogleNotConnectedError",
    "GmailSearchTool",
    "GmailReadTool",
    "GmailSendTool",
    "CalendarListEventsTool",
    "CalendarCreateEventTool",
    "DriveSearchTool",
    "DriveReadTool",
    "TasksListTool",
    "TasksAddTool",
    "GOOGLE_TOOLS",
    "GOOGLE_TOOL_CLASSES",
]
