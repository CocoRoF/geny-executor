"""Atlassian tools — Jira + Confluence, credential-gated built-ins.

An optional executor tool family (like the Google Workspace one): the host
injects credentials into ``ctx.extras["atlassian"]``::

    {
        "base_url": "https://acme.atlassian.net",   # site root
        "email": "user@acme.com",                    # Cloud: Basic email:token
        "api_token": "...",                          # Cloud API token / DC PAT
        "confluence_base_url": "",                   # optional Server/DC split
    }

Auth: ``email`` present → Basic ``email:api_token`` (Atlassian Cloud);
otherwise ``Bearer api_token`` (Server / Data Center personal access token).

Gating: every tool advertises ``required_config_keys() ->
["feature:atlassian_connected"]`` so the host hides the whole family until a
valid config exists (progressive disclosure — an unconfigured tool never
reaches the model).

API surface: Jira ``/rest/api/2`` (string bodies — works on Cloud AND
Server/DC, no ADF juggling) and Confluence ``/wiki/rest/api`` (Cloud) or
``{confluence_base_url}/rest/api`` (Server/DC).
"""

from __future__ import annotations

import base64
import html as _html
import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from geny_executor.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult

logger = logging.getLogger(__name__)

_ATLASSIAN_FEATURE_KEY = "feature:atlassian_connected"
_DEFAULT_TIMEOUT = 30.0
_BODY_TRUNCATE = 20_000  # cap on page/description text returned to the LLM


class AtlassianNotConnectedError(Exception):
    """Raised when no usable Atlassian credentials are available."""


class _ApiError(RuntimeError):
    """A non-auth Atlassian API error, carrying the HTTP status."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def _strip_html(markup: str) -> str:
    """Confluence storage XHTML → readable plain text (best-effort)."""
    text = re.sub(r"<(br|/p|/h[1-6]|/li|/tr)[^>]*>", "\n", markup or "")
    text = re.sub(r"<li[^>]*>", "- ", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = _html.unescape(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


class _AtlassianClient:
    """Thin authenticated wrapper over the Jira / Confluence REST APIs."""

    def __init__(
        self,
        base_url: str,
        api_token: str,
        *,
        email: str = "",
        confluence_base_url: str = "",
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._confluence_base = (confluence_base_url or "").rstrip("/")
        if email:
            raw = f"{email}:{api_token}".encode()
            self._auth_header = "Basic " + base64.b64encode(raw).decode()
        else:
            self._auth_header = f"Bearer {api_token}"
        self._transport = transport

    @classmethod
    def from_context(cls, context: ToolContext) -> "_AtlassianClient":
        extras = getattr(context, "extras", None) or {}
        bag = extras.get("atlassian")
        if not isinstance(bag, dict):
            raise AtlassianNotConnectedError(
                "Atlassian is not connected — configure the site URL and API "
                "token in the host settings (Settings → Tool → Atlassian)."
            )
        base_url = str(bag.get("base_url") or "").strip()
        api_token = str(bag.get("api_token") or "").strip()
        if not base_url or not api_token:
            raise AtlassianNotConnectedError(
                "Atlassian credentials are incomplete (base_url + api_token "
                "required)."
            )
        return cls(
            base_url,
            api_token,
            email=str(bag.get("email") or "").strip(),
            confluence_base_url=str(bag.get("confluence_base_url") or "").strip(),
        )

    # ── HTTP ───────────────────────────────────────────────────────────
    async def _request(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        headers = {
            "Authorization": self._auth_header,
            "Accept": "application/json",
        }
        kwargs: Dict[str, Any] = {"timeout": _DEFAULT_TIMEOUT}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        async with httpx.AsyncClient(**kwargs) as client:
            resp = await client.request(
                method, url, params=params, json=json_body, headers=headers
            )
        if resp.status_code in (401, 403):
            raise AtlassianNotConnectedError(
                f"Atlassian rejected the credentials (HTTP {resp.status_code}) "
                "— check the API token / permissions."
            )
        if resp.status_code >= 400:
            detail = resp.text[:400]
            raise _ApiError(resp.status_code, f"Atlassian API {resp.status_code}: {detail}")
        if resp.status_code == 204 or not resp.content:
            return {}
        return resp.json()

    async def jira(self, method: str, path: str, **kw: Any) -> Any:
        return await self._request(method, f"{self._base}/rest/api/2{path}", **kw)

    async def jira3(self, method: str, path: str, **kw: Any) -> Any:
        return await self._request(method, f"{self._base}/rest/api/3{path}", **kw)

    async def confluence(self, method: str, path: str, **kw: Any) -> Any:
        base = self._confluence_base or f"{self._base}/wiki"
        return await self._request(method, f"{base}/rest/api{path}", **kw)


# ─────────────────────────────────────────────────────────────────────
# Base class for the Atlassian tool family
# ─────────────────────────────────────────────────────────────────────


class _AtlassianTool(Tool):
    """Common scaffolding: client from extras, every failure → ToolResult."""

    def required_config_keys(self) -> List[str]:
        return [_ATLASSIAN_FEATURE_KEY]

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        # Read-only by default; mutating tools override.
        return ToolCapabilities(
            concurrency_safe=True,
            read_only=True,
            idempotent=False,
            network_egress=True,
        )

    async def _run(self, input: Dict[str, Any], client: _AtlassianClient) -> ToolResult:
        raise NotImplementedError

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            client = _AtlassianClient.from_context(context)
        except AtlassianNotConnectedError as exc:
            return ToolResult(content=str(exc), is_error=True)
        return await self._run_wrapped(input, client)

    async def _run_wrapped(
        self, input: Dict[str, Any], client: _AtlassianClient
    ) -> ToolResult:
        """``_run`` with the family's error funneling (tests inject a client)."""
        try:
            return await self._run(input or {}, client)
        except AtlassianNotConnectedError as exc:
            return ToolResult(content=str(exc), is_error=True)
        except httpx.TimeoutException:
            return ToolResult(
                content=f"{self.name}: request to Atlassian timed out.", is_error=True
            )
        except httpx.HTTPError as exc:
            return ToolResult(
                content=f"{self.name}: network error talking to Atlassian: {exc}",
                is_error=True,
            )
        except Exception as exc:  # noqa: BLE001 — never let execute raise
            logger.exception("%s failed", self.name)
            return ToolResult(content=f"{self.name} failed: {exc}", is_error=True)


def _issue_row(issue: Dict[str, Any]) -> Dict[str, Any]:
    f = issue.get("fields") or {}
    return {
        "key": issue.get("key"),
        "summary": f.get("summary"),
        "status": (f.get("status") or {}).get("name"),
        "type": (f.get("issuetype") or {}).get("name"),
        "priority": (f.get("priority") or {}).get("name"),
        "assignee": (f.get("assignee") or {}).get("displayName"),
        "updated": f.get("updated"),
    }


# ─────────────────────────────────────────────────────────────────────
# Jira
# ─────────────────────────────────────────────────────────────────────


class JiraSearchTool(_AtlassianTool):
    @property
    def name(self) -> str:
        return "jira_search"

    @property
    def description(self) -> str:
        return (
            "Search Jira issues with JQL (e.g. 'project = ABC AND status != "
            "Done ORDER BY updated DESC'). Returns key, summary, status, "
            "type, priority, assignee, updated. Use jira_issue for full "
            "detail."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "jql": {"type": "string", "description": "JQL query."},
                "max_results": {
                    "type": "integer",
                    "description": "Max issues to return (default 20, cap 100).",
                },
            },
            "required": ["jql"],
        }

    async def _run(self, input: Dict[str, Any], client: _AtlassianClient) -> ToolResult:
        limit = max(1, min(100, int(input.get("max_results") or 20)))
        payload = {
            "jql": str(input.get("jql") or ""),
            "maxResults": limit,
            "fields": [
                "summary", "status", "issuetype", "priority",
                "assignee", "updated",
            ],
        }
        # Cloud removed ``/rest/api/2/search`` in 2026 (CHANGE-2046) in favour
        # of ``/rest/api/3/search/jql``; Server/DC has no v3. Try the Cloud
        # endpoint first and fall back on "no such endpoint" statuses only —
        # a 400 (bad JQL) is the same on both and must surface as-is.
        try:
            data = await client.jira3("POST", "/search/jql", json_body=payload)
        except _ApiError as exc:
            if exc.status not in (404, 405, 410):
                raise
            data = await client.jira("POST", "/search", json_body=payload)
        rows = [_issue_row(i) for i in data.get("issues", [])]
        total = data.get("total")  # absent on the v3 endpoint
        out: Dict[str, Any] = {"issues": rows}
        if total is not None:
            out["total"] = total
        if data.get("isLast") is False:
            out["more"] = True
        return ToolResult(
            content=json.dumps(out, ensure_ascii=False, indent=1),
            metadata={"total": total, "returned": len(rows)},
        )


class JiraIssueTool(_AtlassianTool):
    @property
    def name(self) -> str:
        return "jira_issue"

    @property
    def description(self) -> str:
        return (
            "Read one Jira issue in full: fields, description, labels, "
            "links, and the latest comments. `key` is the issue key "
            "(e.g. ABC-123)."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Issue key, e.g. ABC-123."},
            },
            "required": ["key"],
        }

    async def _run(self, input: Dict[str, Any], client: _AtlassianClient) -> ToolResult:
        key = str(input.get("key") or "").strip()
        issue = await client.jira("GET", f"/issue/{key}")
        f = issue.get("fields") or {}
        comments = ((f.get("comment") or {}).get("comments") or [])[-5:]
        out = {
            **_issue_row(issue),
            "reporter": (f.get("reporter") or {}).get("displayName"),
            "labels": f.get("labels"),
            "created": f.get("created"),
            "description": str(f.get("description") or "")[:_BODY_TRUNCATE],
            "comments": [
                {
                    "author": (c.get("author") or {}).get("displayName"),
                    "created": c.get("created"),
                    "body": str(c.get("body") or "")[:2000],
                }
                for c in comments
            ],
        }
        return ToolResult(
            content=json.dumps(out, ensure_ascii=False, indent=1),
            metadata={"key": key},
        )


class JiraCreateTool(_AtlassianTool):
    @property
    def name(self) -> str:
        return "jira_create"

    @property
    def description(self) -> str:
        return (
            "Create a Jira issue. Required: project (key, e.g. ABC), "
            "issue_type (e.g. Task/Bug/Story), summary. Optional: "
            "description (plain text / Jira wiki markup), fields (raw "
            "extra fields dict, e.g. labels, priority)."
        )

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(
            concurrency_safe=False, read_only=False, network_egress=True
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "project": {"type": "string", "description": "Project key (ABC)."},
                "issue_type": {"type": "string", "description": "Task / Bug / Story…"},
                "summary": {"type": "string"},
                "description": {"type": "string"},
                "fields": {
                    "type": "object",
                    "description": "Extra raw fields merged into the create body.",
                },
            },
            "required": ["project", "issue_type", "summary"],
        }

    async def _run(self, input: Dict[str, Any], client: _AtlassianClient) -> ToolResult:
        fields: Dict[str, Any] = {
            "project": {"key": str(input.get("project") or "").strip()},
            "issuetype": {"name": str(input.get("issue_type") or "").strip()},
            "summary": str(input.get("summary") or "").strip(),
        }
        if input.get("description"):
            fields["description"] = str(input["description"])
        extra = input.get("fields")
        if isinstance(extra, dict):
            fields.update(extra)
        data = await client.jira("POST", "/issue", json_body={"fields": fields})
        return ToolResult(
            content=json.dumps(
                {"created": data.get("key"), "id": data.get("id")},
                ensure_ascii=False,
            ),
            metadata={"key": data.get("key")},
        )


class JiraUpdateTool(_AtlassianTool):
    @property
    def name(self) -> str:
        return "jira_update"

    @property
    def description(self) -> str:
        return (
            "Update fields on a Jira issue: summary, description, labels, "
            "or any raw fields dict. To move status use jira_transition; "
            "to comment use jira_comment."
        )

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(
            concurrency_safe=False, read_only=False, network_egress=True
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Issue key, e.g. ABC-123."},
                "summary": {"type": "string"},
                "description": {"type": "string"},
                "labels": {"type": "array", "items": {"type": "string"}},
                "fields": {
                    "type": "object",
                    "description": "Extra raw fields merged into the update body.",
                },
            },
            "required": ["key"],
        }

    async def _run(self, input: Dict[str, Any], client: _AtlassianClient) -> ToolResult:
        key = str(input.get("key") or "").strip()
        fields: Dict[str, Any] = {}
        for plain in ("summary", "description"):
            if input.get(plain) is not None:
                fields[plain] = str(input[plain])
        if input.get("labels") is not None:
            fields["labels"] = [str(x) for x in (input.get("labels") or [])]
        extra = input.get("fields")
        if isinstance(extra, dict):
            fields.update(extra)
        if not fields:
            return ToolResult(content="nothing to update — pass at least one field", is_error=True)
        await client.jira("PUT", f"/issue/{key}", json_body={"fields": fields})
        return ToolResult(
            content=json.dumps({"updated": key, "fields": sorted(fields)}, ensure_ascii=False),
            metadata={"key": key},
        )


class JiraCommentTool(_AtlassianTool):
    @property
    def name(self) -> str:
        return "jira_comment"

    @property
    def description(self) -> str:
        return "Add a comment to a Jira issue (plain text / wiki markup)."

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(
            concurrency_safe=False, read_only=False, network_egress=True
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Issue key, e.g. ABC-123."},
                "body": {"type": "string", "description": "Comment text."},
            },
            "required": ["key", "body"],
        }

    async def _run(self, input: Dict[str, Any], client: _AtlassianClient) -> ToolResult:
        key = str(input.get("key") or "").strip()
        data = await client.jira(
            "POST", f"/issue/{key}/comment",
            json_body={"body": str(input.get("body") or "")},
        )
        return ToolResult(
            content=json.dumps({"commented": key, "id": data.get("id")}, ensure_ascii=False),
            metadata={"key": key},
        )


class JiraTransitionTool(_AtlassianTool):
    @property
    def name(self) -> str:
        return "jira_transition"

    @property
    def description(self) -> str:
        return (
            "Move a Jira issue through its workflow. Without `to`: list the "
            "available transitions (id + name). With `to` (transition name "
            "or id, e.g. 'Done'): apply it."
        )

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        read_only = not (input or {}).get("to")
        return ToolCapabilities(
            concurrency_safe=read_only, read_only=read_only, network_egress=True
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Issue key, e.g. ABC-123."},
                "to": {
                    "type": "string",
                    "description": "Transition name or id. Omit to list options.",
                },
            },
            "required": ["key"],
        }

    async def _run(self, input: Dict[str, Any], client: _AtlassianClient) -> ToolResult:
        key = str(input.get("key") or "").strip()
        data = await client.jira("GET", f"/issue/{key}/transitions")
        transitions = [
            {"id": t.get("id"), "name": t.get("name"),
             "to": ((t.get("to") or {}).get("name"))}
            for t in data.get("transitions", [])
        ]
        to = str(input.get("to") or "").strip()
        if not to:
            return ToolResult(
                content=json.dumps({"key": key, "transitions": transitions}, ensure_ascii=False, indent=1),
                metadata={"key": key},
            )
        match = next(
            (t for t in transitions
             if t["id"] == to or str(t["name"]).lower() == to.lower()),
            None,
        )
        if match is None:
            return ToolResult(
                content=(
                    f"no transition {to!r} on {key} — available: "
                    + ", ".join(f"{t['name']} (id {t['id']})" for t in transitions)
                ),
                is_error=True,
            )
        await client.jira(
            "POST", f"/issue/{key}/transitions",
            json_body={"transition": {"id": match["id"]}},
        )
        return ToolResult(
            content=json.dumps(
                {"transitioned": key, "via": match["name"], "to": match["to"]},
                ensure_ascii=False,
            ),
            metadata={"key": key},
        )


# ─────────────────────────────────────────────────────────────────────
# Confluence
# ─────────────────────────────────────────────────────────────────────


class ConfluenceSearchTool(_AtlassianTool):
    @property
    def name(self) -> str:
        return "confluence_search"

    @property
    def description(self) -> str:
        return (
            "Search Confluence pages. Pass `cql` (Confluence Query "
            "Language) OR plain `text` (optionally with `space` key). "
            "Returns page id, title, space, url. Use confluence_page for "
            "the content."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "cql": {"type": "string", "description": "Raw CQL (advanced)."},
                "text": {"type": "string", "description": "Plain-text search."},
                "space": {"type": "string", "description": "Space key filter."},
                "limit": {"type": "integer", "description": "Max results (default 15)."},
            },
        }

    async def _run(self, input: Dict[str, Any], client: _AtlassianClient) -> ToolResult:
        cql = str(input.get("cql") or "").strip()
        if not cql:
            text = str(input.get("text") or "").strip()
            if not text:
                return ToolResult(content="pass `cql` or `text`", is_error=True)
            quoted = text.replace('"', '\\"')
            cql = f'type = page AND text ~ "{quoted}"'
            if input.get("space"):
                cql += f' AND space = "{str(input["space"]).strip()}"'
        limit = max(1, min(50, int(input.get("limit") or 15)))
        data = await client.confluence(
            "GET", "/content/search", params={"cql": cql, "limit": limit}
        )
        rows = [
            {
                "id": r.get("id"),
                "title": r.get("title"),
                "space": ((r.get("space") or {}).get("key")),
                "url": ((r.get("_links") or {}).get("webui")),
            }
            for r in data.get("results", [])
        ]
        return ToolResult(
            content=json.dumps({"results": rows}, ensure_ascii=False, indent=1),
            metadata={"returned": len(rows)},
        )


class ConfluencePageTool(_AtlassianTool):
    @property
    def name(self) -> str:
        return "confluence_page"

    @property
    def description(self) -> str:
        return (
            "Read one Confluence page by id: title, space, version, and the "
            "content as readable text (raw=true for the storage XHTML — "
            "needed before confluence_write updates)."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "page_id": {"type": "string", "description": "Numeric page id."},
                "raw": {
                    "type": "boolean",
                    "description": "Return storage XHTML instead of plain text.",
                },
            },
            "required": ["page_id"],
        }

    async def _run(self, input: Dict[str, Any], client: _AtlassianClient) -> ToolResult:
        pid = str(input.get("page_id") or "").strip()
        data = await client.confluence(
            "GET", f"/content/{pid}",
            params={"expand": "body.storage,version,space"},
        )
        storage = (((data.get("body") or {}).get("storage") or {}).get("value")) or ""
        body = storage if input.get("raw") else _strip_html(storage)
        out = {
            "id": data.get("id"),
            "title": data.get("title"),
            "space": ((data.get("space") or {}).get("key")),
            "version": ((data.get("version") or {}).get("number")),
            "body": body[:_BODY_TRUNCATE],
        }
        return ToolResult(
            content=json.dumps(out, ensure_ascii=False, indent=1),
            metadata={"id": pid, "version": out["version"]},
        )


class ConfluenceWriteTool(_AtlassianTool):
    @property
    def name(self) -> str:
        return "confluence_write"

    @property
    def description(self) -> str:
        return (
            "Create or update a Confluence page. CREATE: space + title + "
            "body. UPDATE: page_id + body (+ title to rename) — the current "
            "version is fetched and bumped automatically. `body` is "
            "Confluence storage format (simple HTML: <h1>, <p>, <ul><li>, "
            "<table>, <strong>…)."
        )

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(
            concurrency_safe=False, read_only=False, network_egress=True
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "page_id": {"type": "string", "description": "UPDATE: existing page id."},
                "space": {"type": "string", "description": "CREATE: space key."},
                "title": {"type": "string", "description": "Page title."},
                "body": {"type": "string", "description": "Storage-format HTML body."},
                "parent_id": {"type": "string", "description": "CREATE: optional parent page."},
            },
            "required": ["body"],
        }

    async def _run(self, input: Dict[str, Any], client: _AtlassianClient) -> ToolResult:
        body = str(input.get("body") or "")
        page_id = str(input.get("page_id") or "").strip()
        if page_id:
            current = await client.confluence(
                "GET", f"/content/{page_id}", params={"expand": "version"}
            )
            version = int(((current.get("version") or {}).get("number")) or 0) + 1
            payload = {
                "id": page_id,
                "type": "page",
                "title": str(input.get("title") or current.get("title") or ""),
                "version": {"number": version},
                "body": {"storage": {"value": body, "representation": "storage"}},
            }
            data = await client.confluence(
                "PUT", f"/content/{page_id}", json_body=payload
            )
            return ToolResult(
                content=json.dumps(
                    {"updated": data.get("id"), "version": version}, ensure_ascii=False
                ),
                metadata={"id": data.get("id")},
            )
        space = str(input.get("space") or "").strip()
        title = str(input.get("title") or "").strip()
        if not space or not title:
            return ToolResult(
                content="CREATE needs `space` + `title` (or pass `page_id` to update)",
                is_error=True,
            )
        payload = {
            "type": "page",
            "title": title,
            "space": {"key": space},
            "body": {"storage": {"value": body, "representation": "storage"}},
        }
        if input.get("parent_id"):
            payload["ancestors"] = [{"id": str(input["parent_id"])}]
        data = await client.confluence("POST", "/content", json_body=payload)
        return ToolResult(
            content=json.dumps(
                {"created": data.get("id"), "title": title,
                 "url": ((data.get("_links") or {}).get("webui"))},
                ensure_ascii=False,
            ),
            metadata={"id": data.get("id")},
        )


#: Registry-name → class map, splatted into ``BUILT_IN_TOOL_CLASSES``.
ATLASSIAN_TOOL_CLASSES: Dict[str, type] = {
    "jira_search": JiraSearchTool,
    "jira_issue": JiraIssueTool,
    "jira_create": JiraCreateTool,
    "jira_update": JiraUpdateTool,
    "jira_comment": JiraCommentTool,
    "jira_transition": JiraTransitionTool,
    "confluence_search": ConfluenceSearchTool,
    "confluence_page": ConfluencePageTool,
    "confluence_write": ConfluenceWriteTool,
}
