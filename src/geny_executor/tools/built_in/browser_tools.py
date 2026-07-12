"""Browser tools — AI-native web exploration backed by the an-web engine.

2.43.0 — replaces host-side Playwright tool stacks (e.g. Geny's old
``browser_*`` family) with `an-web &lt;https://pypi.org/project/an-web/&gt;`_:
a pip-installable headless engine (httpx + Lexbor parser + embedded V8
via mini-racer). No Chromium download, no ``playwright install``, and
the page is exposed to the model as a *semantic tree* — roles, names
and stable ``[ref=nN]`` handles — instead of pixels.

Install: ``pip install 'geny-executor[browser]'`` (requires Python
>= 3.12 and a glibc platform; Alpine/musl is not supported by the V8
wheel). The tools import an-web lazily and return a friendly install
hint as a ToolResult error when it is missing, so catalogs that ship
them unconditionally still degrade gracefully.

Session model
-------------

One an-web ``Session`` ("browser tab") per pipeline session, created
lazily on first use and keyed by ``ToolContext.session_id`` — cookies,
localStorage and history persist across tool calls exactly like the
old Playwright singleton, but *per agent session* instead of process-
global, so concurrent agents do not stomp each other's tabs. Engines
are kept per event loop (an-web is asyncio-bound, not thread-safe).
Idle sessions are reaped after ``_SESSION_IDLE_TTL`` seconds;
``BrowserClose`` frees the tab (V8 heap + connection pool) eagerly.

All tools are ``concurrency_safe=False`` — calls within one agent
session share one tab and must serialize.

Element targeting (the ``target`` parameter)
--------------------------------------------

* ``"n42"`` — a ``[ref=...]`` node handle from a snapshot (exact).
* ``"text=Sign in"`` — visible-text match.
* any other string — CSS selector (``#login``, ``.price``, ``a[href]``).
* a dict is passed through verbatim (an-web semantic locator, e.g.
  ``{"by": "role", "role": "button", "text": "Login"}``).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from geny_executor.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult

_INSTALL_HINT = (
    "The an-web engine is not installed. Install it with: "
    "pip install 'geny-executor[browser]' (requires Python >= 3.12, "
    "glibc platform). Host operators: add 'an-web>=0.9.1' to the "
    "deployment image."
)

_REF_PATTERN = re.compile(r"^n\d+$")
_TEXT_PREFIX = "text="

# Snapshot rendering budgets — a full semantic tree on a large page is
# unbounded; the model reads a capped YAML view and drills down with
# BrowserExtract when it needs more.
_SNAPSHOT_NODE_BUDGET = 400
_AFTER_ACTION_NODE_BUDGET = 150
_SESSION_IDLE_TTL = 900.0  # seconds


def _load_an_web():
    """Import an-web lazily. Raises RuntimeError with an install hint."""
    try:
        from an_web import ANWebEngine  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(_INSTALL_HINT) from exc
    return ANWebEngine


class _BrowserRuntime:
    """Per-process registry: one an-web engine per event loop, one
    session per (loop, session_id). an-web engines are asyncio-bound —
    sharing one across loops (threads) is undefined behaviour, so the
    registry keys everything by the running loop's id."""

    def __init__(self) -> None:
        self._engines: Dict[int, Any] = {}
        # (loop_id, session_id) -> [session, last_used_monotonic]
        self._sessions: Dict[Tuple[int, str], List[Any]] = {}

    @staticmethod
    def _loop_id() -> int:
        return id(asyncio.get_running_loop())

    async def _engine(self) -> Any:
        ANWebEngine = _load_an_web()
        lid = self._loop_id()
        engine = self._engines.get(lid)
        if engine is None:
            engine = ANWebEngine()
            self._engines[lid] = engine
        return engine

    async def get_session(self, session_id: str) -> Any:
        """Return (creating if needed) the tab bound to *session_id*."""
        await self._reap_idle()
        lid = self._loop_id()
        key = (lid, session_id or "default")
        entry = self._sessions.get(key)
        if entry is not None:
            entry[1] = time.monotonic()
            return entry[0]
        engine = await self._engine()
        session = await engine.create_session()
        self._sessions[key] = [session, time.monotonic()]
        return session

    def peek_session(self, session_id: str) -> Optional[Any]:
        try:
            lid = self._loop_id()
        except RuntimeError:
            return None
        entry = self._sessions.get((lid, session_id or "default"))
        return entry[0] if entry else None

    async def close_session(self, session_id: str) -> bool:
        lid = self._loop_id()
        entry = self._sessions.pop((lid, session_id or "default"), None)
        if entry is None:
            return False
        try:
            await entry[0].close()
        except Exception:  # noqa: BLE001 — closing must never raise upward
            pass
        return True

    async def _reap_idle(self) -> None:
        now = time.monotonic()
        expired = [k for k, v in self._sessions.items() if now - v[1] > _SESSION_IDLE_TTL]
        for key in expired:
            entry = self._sessions.pop(key, None)
            if entry is not None:
                try:
                    await entry[0].close()
                except Exception:  # noqa: BLE001
                    pass


_runtime = _BrowserRuntime()


# ── Target / rendering helpers ───────────────────────────────


def _parse_target(target: Any) -> Any:
    """Map the string conventions documented in the module docstring
    onto an-web locators; dicts pass through verbatim."""
    if isinstance(target, dict):
        return target
    if not isinstance(target, str):
        return target
    t = target.strip()
    if _REF_PATTERN.match(t):
        return {"by": "node_id", "node_id": t}
    if t.startswith(_TEXT_PREFIX):
        return {"by": "text", "text": t[len(_TEXT_PREFIX):]}
    return t


def _render_node(node: Any, lines: List[str], depth: int, budget: List[int]) -> None:
    """YAML-ish semantic-tree renderer (an-web MCP reference layout):
    one line per meaningful node, interactive nodes carry [ref=nN]."""
    if budget[0] <= 0:
        return
    role = getattr(node, "role", None) or getattr(node, "tag", None) or ""
    name = (getattr(node, "name", None) or "").strip()
    if len(name) > 80:
        name = name[:77] + "..."
    interactive = bool(getattr(node, "is_interactive", False))

    skip_self = role in ("generic", "none", "presentation", "") and not name and not interactive
    if not skip_self:
        parts = [f"- {role}"]
        if name:
            parts.append(f' "{name}"')
        value = getattr(node, "value", None)
        if value:
            parts.append(f" [value={str(value)[:40]!r}]")
        if interactive:
            parts.append(f" [ref={getattr(node, 'node_id', '?')}]")
        lines.append("  " * depth + "".join(parts))
        budget[0] -= 1
        depth += 1

    for child in getattr(node, "children", None) or []:
        _render_node(child, lines, depth, budget)


async def _snapshot_text(
    session: Any,
    header: str = "",
    node_budget: int = _SNAPSHOT_NODE_BUDGET,
) -> str:
    snap = await session.snapshot()
    lines: List[str] = []
    budget = [node_budget]
    tree_root = getattr(snap, "semantic_tree", None)
    if tree_root is not None:
        _render_node(tree_root, lines, 0, budget)
    tree = "\n".join(lines) if lines else "(empty page)"
    if budget[0] <= 0:
        tree += "\n... (truncated at node budget; use BrowserExtract for details)"

    out: List[str] = []
    if header:
        out.append(header)
    out.append("### Page")
    out.append(f"- URL: {session.current_url}")
    out.append(f"- Title: {getattr(snap, 'title', '')}")
    out.append(f"- Page type: {getattr(snap, 'page_type', '')}")
    out.append("\n### Snapshot")
    out.append("```yaml")
    out.append(tree)
    out.append("```")
    out.append(
        "\nInteract via BrowserAct using a [ref=...] handle, text=..., or a CSS selector."
    )
    return "\n".join(out)


def _semantic_to_text(node: Any, parts: List[str]) -> None:
    """Plain reading text from the semantic tree (StaticText leaves,
    newline per structural node) — used by WebFetch's render_js mode."""
    role = getattr(node, "role", None) or ""
    name = getattr(node, "name", None) or ""
    if role == "StaticText" and name:
        parts.append(name)
        parts.append(" ")
        return
    children = getattr(node, "children", None) or []
    for child in children:
        _semantic_to_text(child, parts)
    if role in ("paragraph", "heading", "listitem", "row", "region", "article", "list"):
        parts.append("\n")


async def fetch_rendered_text(
    url: str,
    *,
    timeout: Optional[float] = None,
) -> Tuple[str, str, str]:
    """One-shot JS-rendered fetch for WebFetch's ``render_js`` mode.

    Uses an ephemeral an-web session (no cookies/history shared with the
    per-session tab — WebFetch stays stateless) and returns
    ``(final_url, title, reading_text)``. Raises RuntimeError with an
    install hint when an-web is unavailable, or with the navigate error.
    """
    engine = await _runtime._engine()
    session = await engine.create_session()
    try:
        kwargs: Dict[str, Any] = {}
        if timeout is not None:
            kwargs["timeout"] = timeout
        result = await session.navigate(url, **kwargs)
        if result.get("status") != "ok":
            raise RuntimeError(
                str(result.get("error") or f"navigate failed ({result.get('status')})")
            )
        snap = await session.snapshot()
        parts: List[str] = []
        tree = getattr(snap, "semantic_tree", None)
        if tree is not None:
            _semantic_to_text(tree, parts)
        text = re.sub(r"\n{3,}", "\n\n", "".join(parts)).strip()
        return session.current_url or url, getattr(snap, "title", "") or "", text
    finally:
        try:
            await session.close()
        except Exception:  # noqa: BLE001
            pass


def _effects_summary(result: Dict[str, Any]) -> str:
    status = result.get("status", "?")
    if status != "ok":
        return f"### Error\nstatus: {status}\n{result.get('error', 'unknown error')}"
    effects = result.get("effects") or {}
    scalars = {
        k: v
        for k, v in effects.items()
        if not isinstance(v, (list, dict)) and k not in ("body",)
    }
    lines = [f"status: {status}"]
    for k, v in scalars.items():
        lines.append(f"{k}: {v}")
    return "\n".join(lines)


class _BrowserToolBase(Tool):
    """Shared plumbing: resolve the per-session tab, translate an-web
    unavailability / engine errors into ToolResult errors."""

    async def _session(self, context: ToolContext) -> Any:
        return await _runtime.get_session(context.session_id)

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            return await self._run(input, context)
        except RuntimeError as exc:
            return ToolResult(content=str(exc), is_error=True)
        except Exception as exc:  # noqa: BLE001 — engine faults become tool errors
            return ToolResult(
                content=f"{type(exc).__name__}: {exc}",
                is_error=True,
                metadata={"error_type": type(exc).__name__},
            )

    async def _run(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        raise NotImplementedError


class BrowserNavigateTool(_BrowserToolBase):
    """Open a URL in the session's tab (executes page JavaScript) and
    return the semantic snapshot."""

    @property
    def name(self) -> str:
        return "BrowserNavigate"

    @property
    def description(self) -> str:
        return (
            "Open a URL in this session's browser tab. Runs the page's "
            "JavaScript (embedded V8 — handles SPAs/React), keeps cookies and "
            "history across calls, and returns a semantic snapshot of the "
            "rendered page with [ref=...] handles for interactive elements. "
            "Use BrowserAct to click/type, BrowserExtract to pull data."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "http(s) URL to open."},
                "timeout": {
                    "type": "number",
                    "description": (
                        "Settle budget in seconds (default 15). Lower it (e.g. 3) "
                        "for script-heavy sites when a partial render is fine."
                    ),
                    "exclusiveMinimum": 0,
                },
                "max_nodes": {
                    "type": "integer",
                    "description": f"Snapshot node budget (default {_SNAPSHOT_NODE_BUDGET}).",
                    "exclusiveMinimum": 0,
                },
            },
            "required": ["url"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(
            concurrency_safe=False,
            read_only=True,
            network_egress=True,
            max_result_chars=50_000,
        )

    async def _run(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        url = (input.get("url") or "").strip()
        if not url:
            return ToolResult(content="url must not be empty", is_error=True)
        if "://" not in url:
            url = f"https://{url}"
        if not url.startswith(("http://", "https://")):
            return ToolResult(
                content=f"Unsupported URL scheme: {url!r} (http/https only)",
                is_error=True,
            )
        # SSRF guard (audit S5): block navigation to private / loopback /
        # link-local / cloud-metadata addresses.
        from geny_executor.security import SSRFError, validate_url as _ssrf_validate

        try:
            _ssrf_validate(url)
        except SSRFError as exc:
            return ToolResult(content=f"blocked (SSRF guard): {exc}", is_error=True)
        session = await self._session(context)
        kwargs: Dict[str, Any] = {}
        if input.get("timeout") is not None:
            kwargs["timeout"] = float(input["timeout"])
        result = await session.navigate(url, **kwargs)
        status = result.get("status")
        if status != "ok":
            return ToolResult(content=_effects_summary(result), is_error=True)
        budget = int(input.get("max_nodes") or _SNAPSHOT_NODE_BUDGET)
        text = await _snapshot_text(
            session, header=_effects_summary(result), node_budget=budget
        )
        return ToolResult(
            content=text,
            metadata={"url": session.current_url, "effects": result.get("effects", {})},
        )


class BrowserSnapshotTool(_BrowserToolBase):
    """Re-read the current page as a semantic snapshot."""

    @property
    def name(self) -> str:
        return "BrowserSnapshot"

    @property
    def description(self) -> str:
        return (
            "Return the semantic snapshot of the CURRENT page in this "
            "session's browser tab (roles, names, [ref=...] handles). Use "
            "after BrowserAct when you need to re-read the page state."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "max_nodes": {
                    "type": "integer",
                    "description": f"Snapshot node budget (default {_SNAPSHOT_NODE_BUDGET}).",
                    "exclusiveMinimum": 0,
                },
            },
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(
            concurrency_safe=False,
            read_only=True,
            idempotent=True,
            max_result_chars=50_000,
        )

    async def _run(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        session = _runtime.peek_session(context.session_id)
        if session is None:
            return ToolResult(
                content="No open page in this session — call BrowserNavigate first.",
                is_error=True,
            )
        budget = int(input.get("max_nodes") or _SNAPSHOT_NODE_BUDGET)
        text = await _snapshot_text(session, node_budget=budget)
        return ToolResult(content=text, metadata={"url": session.current_url})


class BrowserActTool(_BrowserToolBase):
    """Interact with the current page: click / type / select / clear /
    submit / scroll / wait_for."""

    _ACTIONS = ("click", "type", "select", "clear", "submit", "scroll", "wait_for")

    @property
    def name(self) -> str:
        return "BrowserAct"

    @property
    def description(self) -> str:
        return (
            "Interact with the current page in this session's browser tab. "
            "Actions: click, type (set text; append=true to append), select "
            "(dropdown), clear, submit (form), scroll, wait_for (condition: "
            "network_idle | dom_stable | selector | element_visible). Target "
            "elements by snapshot ref ('n42'), visible text ('text=Sign in'), "
            "or CSS selector ('#login'). Clicking a link navigates and "
            "returns the new page's snapshot."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": list(self._ACTIONS),
                    "description": "Interaction to perform.",
                },
                "target": {
                    "anyOf": [{"type": "string"}, {"type": "object"}],
                    "description": (
                        "Element: 'n42' (snapshot ref), 'text=...' (visible text), "
                        "CSS selector, or an an-web locator object. Required for "
                        "click/type/select/clear/submit; optional for scroll."
                    ),
                },
                "text": {"type": "string", "description": "Text for action=type."},
                "append": {
                    "type": "boolean",
                    "description": "type: append instead of replace (default false).",
                },
                "value": {"type": "string", "description": "Option value for action=select."},
                "by_text": {
                    "type": "boolean",
                    "description": "select: match option by visible text (default false).",
                },
                "delta_y": {
                    "type": "integer",
                    "description": "scroll: vertical pixels (default 300).",
                },
                "delta_x": {"type": "integer", "description": "scroll: horizontal pixels."},
                "condition": {
                    "type": "string",
                    "enum": ["network_idle", "dom_stable", "selector", "element_visible"],
                    "description": "wait_for: condition to await.",
                },
                "selector": {
                    "type": "string",
                    "description": "wait_for: CSS selector for selector/element_visible.",
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "wait_for: timeout in ms (default 5000).",
                    "exclusiveMinimum": 0,
                },
            },
            "required": ["action"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(
            concurrency_safe=False,
            network_egress=True,
            max_result_chars=40_000,
        )

    async def _run(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        action = (input.get("action") or "").strip()
        if action not in self._ACTIONS:
            return ToolResult(
                content=f"Unknown action {action!r} — expected one of {self._ACTIONS}",
                is_error=True,
            )
        session = _runtime.peek_session(context.session_id)
        if session is None:
            return ToolResult(
                content="No open page in this session — call BrowserNavigate first.",
                is_error=True,
            )

        call: Dict[str, Any] = {"tool": action}
        if input.get("target") is not None:
            call["target"] = _parse_target(input["target"])
        for key in ("text", "append", "value", "by_text", "delta_x", "delta_y",
                    "condition", "selector", "timeout_ms"):
            if input.get(key) is not None:
                call[key] = input[key]

        result = await session.act(call)
        summary = _effects_summary(result)
        is_error = result.get("status") != "ok"
        effects = result.get("effects") or {}
        # A click that navigated → show the new page right away so the
        # model does not need a follow-up BrowserSnapshot call.
        if not is_error and effects.get("navigation"):
            summary = await _snapshot_text(
                session, header=summary, node_budget=_AFTER_ACTION_NODE_BUDGET
            )
        return ToolResult(
            content=summary,
            is_error=is_error,
            metadata={"action": action, "effects": effects, "url": session.current_url},
        )


class BrowserExtractTool(_BrowserToolBase):
    """Pull structured data out of the current page by CSS query."""

    @property
    def name(self) -> str:
        return "BrowserExtract"

    @property
    def description(self) -> str:
        return (
            "Extract data from the current page by CSS selector. Modes: "
            "'css' (visible text per match), 'structured' (text + attributes), "
            "'json' (parse JSON islands), 'html' (raw HTML per match). Use "
            "after BrowserNavigate/BrowserAct when the snapshot alone is not "
            "detailed enough (tables, lists, prices, article bodies)."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "CSS selector to match."},
                "mode": {
                    "type": "string",
                    "enum": ["css", "structured", "json", "html"],
                    "description": "Extraction mode (default 'css').",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max matches to return (default 100).",
                    "exclusiveMinimum": 0,
                },
            },
            "required": ["query"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(
            concurrency_safe=False,
            read_only=True,
            idempotent=True,
            max_result_chars=50_000,
        )

    async def _run(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        session = _runtime.peek_session(context.session_id)
        if session is None:
            return ToolResult(
                content="No open page in this session — call BrowserNavigate first.",
                is_error=True,
            )
        call: Dict[str, Any] = {"tool": "extract", "query": input.get("query") or ""}
        if input.get("mode"):
            call["mode"] = input["mode"]
        if input.get("limit") is not None:
            call["limit"] = int(input["limit"])
        result = await session.act(call)
        if result.get("status") != "ok":
            return ToolResult(content=_effects_summary(result), is_error=True)
        effects = result.get("effects") or {}
        payload = effects.get("results", effects)
        return ToolResult(
            content=json.dumps(payload, ensure_ascii=False, indent=1, default=str),
            metadata={"url": session.current_url, "count": len(payload) if isinstance(payload, list) else None},
        )


class BrowserEvalTool(_BrowserToolBase):
    """Run JavaScript in the current page's V8 context."""

    @property
    def name(self) -> str:
        return "BrowserEval"

    @property
    def description(self) -> str:
        return (
            "Evaluate a JavaScript expression in the current page's V8 "
            "context and return the result (JSON-serialized). The page's own "
            "scripts have already run; DOM and window are available."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "script": {"type": "string", "description": "JavaScript to evaluate."},
            },
            "required": ["script"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(concurrency_safe=False, max_result_chars=30_000)

    async def _run(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        session = _runtime.peek_session(context.session_id)
        if session is None:
            return ToolResult(
                content="No open page in this session — call BrowserNavigate first.",
                is_error=True,
            )
        script = input.get("script") or ""
        if not script.strip():
            return ToolResult(content="script must not be empty", is_error=True)
        result = await session.act({"tool": "eval_js", "script": script})
        if result.get("status") != "ok":
            return ToolResult(content=_effects_summary(result), is_error=True)
        effects = result.get("effects") or {}
        value = effects.get("result", effects)
        return ToolResult(content=json.dumps(value, ensure_ascii=False, default=str))


class BrowserBackTool(_BrowserToolBase):
    """Go back one entry in this session's history."""

    @property
    def name(self) -> str:
        return "BrowserBack"

    @property
    def description(self) -> str:
        return "Go back to the previous page in this session's browser history."

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(
            concurrency_safe=False,
            read_only=True,
            network_egress=True,
            max_result_chars=50_000,
        )

    async def _run(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        session = _runtime.peek_session(context.session_id)
        if session is None:
            return ToolResult(
                content="No open page in this session — call BrowserNavigate first.",
                is_error=True,
            )
        result = await session.back()
        if result.get("status") != "ok":
            return ToolResult(content=_effects_summary(result), is_error=True)
        text = await _snapshot_text(session, header=_effects_summary(result))
        return ToolResult(content=text, metadata={"url": session.current_url})


class BrowserCloseTool(_BrowserToolBase):
    """Close this session's tab and free its resources."""

    @property
    def name(self) -> str:
        return "BrowserClose"

    @property
    def description(self) -> str:
        return (
            "Close this session's browser tab, releasing its JS runtime and "
            "connections. Cookies/history are discarded. A later "
            "BrowserNavigate starts a fresh tab."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(concurrency_safe=False, idempotent=True)

    async def _run(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        closed = await _runtime.close_session(context.session_id)
        return ToolResult(
            content="Browser tab closed." if closed else "No open browser tab.",
            metadata={"closed": closed},
        )


BROWSER_TOOL_CLASSES: Dict[str, type] = {
    "BrowserNavigate": BrowserNavigateTool,
    "BrowserSnapshot": BrowserSnapshotTool,
    "BrowserAct": BrowserActTool,
    "BrowserExtract": BrowserExtractTool,
    "BrowserEval": BrowserEvalTool,
    "BrowserBack": BrowserBackTool,
    "BrowserClose": BrowserCloseTool,
}
