"""Atlassian built-in tools (Jira + Confluence).

Covers the family contract (feature gate via required_config_keys, extras →
client, every failure funneled into ToolResult), the auth header modes
(Cloud Basic email:token vs Server/DC Bearer PAT), and each tool's request
shape + response mapping against an httpx.MockTransport. Live end-to-end is
exercised at deploy time against a real site.
"""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from geny_executor.tools.base import ToolContext
from geny_executor.tools.built_in import BUILT_IN_TOOL_CLASSES, BUILT_IN_TOOL_FEATURES
from geny_executor.tools.built_in.atlassian_tools import (
    ATLASSIAN_TOOL_CLASSES,
    AtlassianNotConnectedError,
    ConfluencePageTool,
    ConfluenceSearchTool,
    ConfluenceWriteTool,
    JiraCommentTool,
    JiraCreateTool,
    JiraIssueTool,
    JiraSearchTool,
    JiraTransitionTool,
    JiraUpdateTool,
    _AtlassianClient,
    _strip_html,
)

_BASE = "https://acme.atlassian.net"


def _ctx(extras=None):
    return ToolContext(session_id="s1", storage_path="/tmp/x", extras=extras or {})


def _client(handler, **kw):
    return _AtlassianClient(
        _BASE, "tok", email="me@acme.com",
        transport=httpx.MockTransport(handler), **kw,
    )


def _json_response(payload, status=200):
    return httpx.Response(status, json=payload)


# ── registry / gating contract ───────────────────────────────────────

def test_family_registered_and_feature_grouped():
    for name, cls in ATLASSIAN_TOOL_CLASSES.items():
        assert BUILT_IN_TOOL_CLASSES[name] is cls
    assert BUILT_IN_TOOL_FEATURES["atlassian"] == list(ATLASSIAN_TOOL_CLASSES.keys())


def test_every_tool_declares_the_feature_gate():
    for cls in ATLASSIAN_TOOL_CLASSES.values():
        assert cls().required_config_keys() == ["feature:atlassian_connected"]


def test_tool_names_match_registry_keys():
    for name, cls in ATLASSIAN_TOOL_CLASSES.items():
        assert cls().name == name


def test_read_write_capability_split():
    ro = {"jira_search", "jira_issue", "confluence_search", "confluence_page"}
    for name, cls in ATLASSIAN_TOOL_CLASSES.items():
        if name == "jira_transition":
            continue  # dual-mode, asserted below
        assert cls().capabilities({}).read_only is (name in ro), name
    t = JiraTransitionTool()
    assert t.capabilities({"key": "A-1"}).read_only is True
    assert t.capabilities({"key": "A-1", "to": "Done"}).read_only is False


# ── client construction / auth ───────────────────────────────────────

def test_from_context_requires_credentials():
    with pytest.raises(AtlassianNotConnectedError):
        _AtlassianClient.from_context(_ctx())
    with pytest.raises(AtlassianNotConnectedError):
        _AtlassianClient.from_context(_ctx({"atlassian": {"base_url": _BASE}}))
    c = _AtlassianClient.from_context(
        _ctx({"atlassian": {"base_url": _BASE + "/", "api_token": "t", "email": "e@x"}})
    )
    assert c._base == _BASE  # trailing slash normalized


def test_basic_auth_header_for_cloud():
    c = _AtlassianClient(_BASE, "tok", email="me@acme.com")
    expected = "Basic " + base64.b64encode(b"me@acme.com:tok").decode()
    assert c._auth_header == expected


def test_bearer_auth_for_server_pat():
    c = _AtlassianClient(_BASE, "pat-123")
    assert c._auth_header == "Bearer pat-123"


@pytest.mark.asyncio
async def test_execute_without_config_is_clean_error():
    res = await JiraSearchTool().execute({"jql": "x"}, _ctx())
    assert res.is_error and "not connected" in res.content


@pytest.mark.asyncio
async def test_401_maps_to_credential_error():
    async def handler(request):
        return httpx.Response(401, text="nope")
    res = await JiraIssueTool()._run_wrapped({"key": "A-1"}, _client(handler))
    assert res.is_error and "credentials" in res.content


@pytest.mark.asyncio
async def test_api_error_surfaces_status_and_body():
    async def handler(request):
        return httpx.Response(400, text='{"errorMessages":["bad jql"]}')
    res = await JiraSearchTool()._run_wrapped({"jql": "x"}, _client(handler))
    assert res.is_error and "400" in res.content and "bad jql" in res.content


# ── Jira tools ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_jira_search_uses_cloud_v3_endpoint():
    seen = {}

    async def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        return _json_response({
            "isLast": False,
            "issues": [{
                "key": "ABC-1",
                "fields": {
                    "summary": "s", "updated": "2026-07-16",
                    "status": {"name": "In Progress"},
                    "issuetype": {"name": "Task"},
                    "priority": {"name": "High"},
                    "assignee": {"displayName": "HR"},
                },
            }],
        })

    res = await JiraSearchTool()._run_wrapped(
        {"jql": "project = ABC", "max_results": 5}, _client(handler)
    )
    assert not res.is_error
    # Cloud removed /rest/api/2/search (CHANGE-2046) — v3 search/jql first.
    assert seen["url"] == f"{_BASE}/rest/api/3/search/jql"
    assert seen["auth"].startswith("Basic ")
    assert seen["body"]["jql"] == "project = ABC"
    assert seen["body"]["maxResults"] == 5
    out = json.loads(res.content)
    assert out["more"] is True  # isLast=False surfaced
    assert out["issues"][0] == {
        "key": "ABC-1", "summary": "s", "status": "In Progress", "type": "Task",
        "priority": "High", "assignee": "HR", "updated": "2026-07-16",
    }


@pytest.mark.asyncio
async def test_jira_search_falls_back_to_v2_on_server_dc():
    urls = []

    async def handler(request):
        urls.append(str(request.url))
        if "/rest/api/3/" in str(request.url):
            return httpx.Response(404, text="no v3 here")
        return _json_response({"total": 0, "issues": []})

    res = await JiraSearchTool()._run_wrapped({"jql": "x"}, _client(handler))
    assert not res.is_error
    assert urls == [f"{_BASE}/rest/api/3/search/jql", f"{_BASE}/rest/api/2/search"]
    assert json.loads(res.content)["total"] == 0


@pytest.mark.asyncio
async def test_jira_search_bad_jql_does_not_retry_v2():
    urls = []

    async def handler(request):
        urls.append(str(request.url))
        return httpx.Response(400, text='{"errorMessages":["bad jql"]}')

    res = await JiraSearchTool()._run_wrapped({"jql": "x"}, _client(handler))
    assert res.is_error and "bad jql" in res.content
    assert len(urls) == 1  # a 400 is the same on both APIs — no second call


@pytest.mark.asyncio
async def test_jira_issue_includes_description_and_last_comments():
    async def handler(request):
        assert str(request.url).endswith("/rest/api/2/issue/ABC-2")
        return _json_response({
            "key": "ABC-2",
            "fields": {
                "summary": "s", "description": "long text",
                "status": {"name": "Done"}, "labels": ["x"],
                "reporter": {"displayName": "R"},
                "comment": {"comments": [
                    {"author": {"displayName": f"a{i}"}, "body": f"c{i}"}
                    for i in range(8)
                ]},
            },
        })

    res = await JiraIssueTool()._run_wrapped({"key": "ABC-2"}, _client(handler))
    out = json.loads(res.content)
    assert out["description"] == "long text"
    assert [c["body"] for c in out["comments"]] == ["c3", "c4", "c5", "c6", "c7"]


@pytest.mark.asyncio
async def test_jira_create_body_shape_and_extra_fields():
    seen = {}

    async def handler(request):
        seen["body"] = json.loads(request.content)
        return _json_response({"key": "ABC-9", "id": "10009"}, status=201)

    res = await JiraCreateTool()._run_wrapped(
        {
            "project": "ABC", "issue_type": "Bug", "summary": "boom",
            "description": "d", "fields": {"labels": ["p0"]},
        },
        _client(handler),
    )
    assert json.loads(res.content)["created"] == "ABC-9"
    assert seen["body"]["fields"] == {
        "project": {"key": "ABC"}, "issuetype": {"name": "Bug"},
        "summary": "boom", "description": "d", "labels": ["p0"],
    }


@pytest.mark.asyncio
async def test_jira_update_requires_some_field_and_merges():
    res = await JiraUpdateTool()._run_wrapped({"key": "A-1"}, _client(None))
    assert res.is_error and "nothing to update" in res.content

    seen = {}

    async def handler(request):
        seen["method"] = request.method
        seen["body"] = json.loads(request.content)
        return httpx.Response(204)

    res = await JiraUpdateTool()._run_wrapped(
        {"key": "A-1", "summary": "new", "labels": ["l1"], "fields": {"priority": {"name": "Low"}}},
        _client(handler),
    )
    assert not res.is_error
    assert seen["method"] == "PUT"
    assert seen["body"]["fields"] == {
        "summary": "new", "labels": ["l1"], "priority": {"name": "Low"},
    }


@pytest.mark.asyncio
async def test_jira_comment():
    async def handler(request):
        assert str(request.url).endswith("/issue/A-1/comment")
        assert json.loads(request.content) == {"body": "hello"}
        return _json_response({"id": "77"}, status=201)

    res = await JiraCommentTool()._run_wrapped({"key": "A-1", "body": "hello"}, _client(handler))
    assert json.loads(res.content) == {"commented": "A-1", "id": "77"}


@pytest.mark.asyncio
async def test_jira_transition_list_apply_and_unknown():
    calls = []

    async def handler(request):
        calls.append((request.method, str(request.url)))
        if request.method == "GET":
            return _json_response({"transitions": [
                {"id": "31", "name": "Done", "to": {"name": "Done"}},
                {"id": "21", "name": "In Progress", "to": {"name": "In Progress"}},
            ]})
        assert json.loads(request.content) == {"transition": {"id": "31"}}
        return httpx.Response(204)

    tool = JiraTransitionTool()
    res = await tool._run_wrapped({"key": "A-1"}, _client(handler))
    assert [t["name"] for t in json.loads(res.content)["transitions"]] == ["Done", "In Progress"]

    res = await tool._run_wrapped({"key": "A-1", "to": "done"}, _client(handler))
    assert json.loads(res.content)["transitioned"] == "A-1"

    res = await tool._run_wrapped({"key": "A-1", "to": "Nope"}, _client(handler))
    assert res.is_error and "Done (id 31)" in res.content


# ── Confluence tools ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_confluence_search_builds_cql_and_uses_wiki_base():
    seen = {}

    async def handler(request):
        seen["url"] = str(request.url.copy_with(query=None))
        seen["cql"] = request.url.params["cql"]
        return _json_response({"results": [{
            "id": "123", "title": "T", "space": {"key": "ENG"},
            "_links": {"webui": "/spaces/ENG/pages/123"},
        }]})

    res = await ConfluenceSearchTool()._run_wrapped(
        {"text": 'release "notes"', "space": "ENG"}, _client(handler)
    )
    assert seen["url"] == f"{_BASE}/wiki/rest/api/content/search"
    assert seen["cql"] == 'type = page AND text ~ "release \\"notes\\"" AND space = "ENG"'
    assert json.loads(res.content)["results"][0]["id"] == "123"


@pytest.mark.asyncio
async def test_confluence_search_needs_cql_or_text():
    res = await ConfluenceSearchTool()._run_wrapped({}, _client(None))
    assert res.is_error


@pytest.mark.asyncio
async def test_confluence_base_url_override_for_server_dc():
    async def handler(request):
        assert str(request.url).startswith("https://conf.corp.local/rest/api/")
        return _json_response({"results": []})

    c = _client(handler, confluence_base_url="https://conf.corp.local")
    res = await ConfluenceSearchTool()._run_wrapped({"cql": "type = page"}, c)
    assert not res.is_error


@pytest.mark.asyncio
async def test_confluence_page_text_vs_raw():
    storage = "<h1>Title</h1><p>Hello <strong>world</strong></p><ul><li>a</li></ul>"

    async def handler(request):
        assert request.url.params["expand"] == "body.storage,version,space"
        return _json_response({
            "id": "42", "title": "Page", "space": {"key": "ENG"},
            "version": {"number": 7},
            "body": {"storage": {"value": storage}},
        })

    tool = ConfluencePageTool()
    out = json.loads((await tool._run_wrapped({"page_id": "42"}, _client(handler))).content)
    assert out["version"] == 7
    assert "<p>" not in out["body"] and "Hello world" in out["body"] and "- a" in out["body"]

    out = json.loads((await tool._run_wrapped({"page_id": "42", "raw": True}, _client(handler))).content)
    assert out["body"] == storage


@pytest.mark.asyncio
async def test_confluence_write_update_bumps_version():
    calls = []

    async def handler(request):
        calls.append(request.method)
        if request.method == "GET":
            return _json_response({"id": "42", "title": "Old", "version": {"number": 7}})
        body = json.loads(request.content)
        assert body["version"] == {"number": 8}
        assert body["title"] == "Old"  # kept when not renaming
        assert body["body"]["storage"]["representation"] == "storage"
        return _json_response({"id": "42"})

    res = await ConfluenceWriteTool()._run_wrapped(
        {"page_id": "42", "body": "<p>new</p>"}, _client(handler)
    )
    assert json.loads(res.content) == {"updated": "42", "version": 8}
    assert calls == ["GET", "PUT"]


@pytest.mark.asyncio
async def test_confluence_write_create_requires_space_and_title():
    res = await ConfluenceWriteTool()._run_wrapped({"body": "<p>x</p>"}, _client(None))
    assert res.is_error and "space" in res.content

    seen = {}

    async def handler(request):
        seen["body"] = json.loads(request.content)
        return _json_response({"id": "9", "_links": {"webui": "/x"}}, status=201)

    res = await ConfluenceWriteTool()._run_wrapped(
        {"space": "ENG", "title": "New", "body": "<p>x</p>", "parent_id": "5"},
        _client(handler),
    )
    assert json.loads(res.content)["created"] == "9"
    assert seen["body"]["space"] == {"key": "ENG"}
    assert seen["body"]["ancestors"] == [{"id": "5"}]


# ── helpers ──────────────────────────────────────────────────────────

def test_strip_html():
    assert _strip_html("<p>a&amp;b</p><ul><li>x</li></ul>") == "a&b\n- x"
    assert _strip_html("") == ""
