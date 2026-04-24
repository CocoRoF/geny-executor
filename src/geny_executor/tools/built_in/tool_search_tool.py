"""ToolSearch — discovery helper for the tools available this turn.

Cycle 20260424 executor uplift — Phase 3 Week 6.

When a pipeline has dozens of tools wired up (executor built-ins + MCP
servers + skill bundles + host-specific custom tools) the LLM often
needs help finding the right one. ``ToolSearch`` is a lightweight
introspection tool — pass it a query, get back a ranked list of
matching tool names + their one-line descriptions.

Source of truth:

1. If ``ToolContext.state_view`` is set (the usual Stage 10 path), we
   read ``state_view.tools`` — the live list of API-format tool
   descriptors the LLM already sees. This includes MCP tools, custom
   host tools, everything.
2. Otherwise (e.g. a test harness running a bare executor without a
   Stage), we fall back to the built-in catalogue
   (``BUILT_IN_TOOL_CLASSES``) so the tool still produces useful
   output in isolation.

Ranking is simple: a hit on the tool ``name`` beats a hit on the
``description``; exact name match beats substring; case-insensitive
throughout. No fuzzy matching yet — plain text is enough for the LLM's
use case here.

See ``executor_uplift/06_design_tool_system.md`` §7 and
``executor_uplift/12_detailed_plan.md`` §3 (Week 6 Meta).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from geny_executor.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult

_DEFAULT_LIMIT = 20
_HARD_LIMIT = 100


def _fallback_descriptors() -> List[Dict[str, Any]]:
    """Build descriptors from the built-in catalogue.

    Only used when ``state_view`` is absent. Mirrors the shape of the
    Anthropic API tool descriptor so rank logic can treat both sources
    the same.
    """
    # Local import to avoid the package-init circular dependency.
    from geny_executor.tools.built_in import BUILT_IN_TOOL_CLASSES

    out: List[Dict[str, Any]] = []
    for name, cls in BUILT_IN_TOOL_CLASSES.items():
        try:
            instance = cls()
            out.append(
                {
                    "name": name,
                    "description": instance.description,
                    "input_schema": instance.input_schema,
                }
            )
        except Exception:
            out.append({"name": name, "description": ""})
    return out


def _rank(descriptor: Dict[str, Any], query: str) -> int:
    """Score a descriptor against ``query``. Higher = better match.

    0 means "no match" (filtered out). The heuristic:

        +100 — exact name match (case-insensitive)
        +50  — query appears in name
        +10  — query appears in description
        +1   — query appears in input_schema description / property keys

    Multi-word queries: every space-separated token must land somewhere.
    """
    name = str(descriptor.get("name", ""))
    description = str(descriptor.get("description", ""))
    schema = descriptor.get("input_schema", {})

    name_lower = name.lower()
    desc_lower = description.lower()
    q = query.lower().strip()
    if not q:
        return 0

    tokens = [t for t in q.split() if t]
    if not tokens:
        return 0

    # Flattened string of input_schema text for cheap substring checks.
    schema_text = ""
    if isinstance(schema, dict):
        props = schema.get("properties")
        if isinstance(props, dict):
            schema_text = " ".join(str(k) for k in props.keys())
            for v in props.values():
                if isinstance(v, dict):
                    schema_text += " " + str(v.get("description", ""))
        schema_text += " " + str(schema.get("description", ""))
        schema_text = schema_text.lower()

    total = 0
    for token in tokens:
        token_score = 0
        if token == name_lower:
            token_score = max(token_score, 100)
        elif token in name_lower:
            token_score = max(token_score, 50)
        if token in desc_lower:
            token_score = max(token_score, 10)
        if token in schema_text:
            token_score = max(token_score, 1)
        if token_score == 0:
            return 0  # one missing token → not a hit
        total += token_score
    return total


class ToolSearchTool(Tool):
    """Find tools matching a keyword query.

    Usage: the LLM calls ``ToolSearch({"query": "grep files"})`` and
    gets back something like::

        Matching 2 tools for 'grep files':
        1. Grep — Search file contents with ripgrep regex syntax.
        2. Glob — Match file paths with shell globbing.

    The host decides which tools to include by deciding what to put on
    ``state.tools`` each turn — this tool just surfaces that list.
    """

    @property
    def name(self) -> str:
        return "ToolSearch"

    @property
    def description(self) -> str:
        return (
            "Find tools available this turn by keyword. Returns ranked "
            "matches with name + description. Useful when unsure which "
            "tool to call for a task."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Keyword query. Multi-word queries require every "
                        "token to match somewhere (name / description / "
                        "input schema)."
                    ),
                    "minLength": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        f"Maximum number of matches to return. Default "
                        f"{_DEFAULT_LIMIT}, hard cap {_HARD_LIMIT}."
                    ),
                    "exclusiveMinimum": 0,
                },
            },
            "required": ["query"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(
            concurrency_safe=True,
            read_only=True,
            idempotent=True,
        )

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        query = (input.get("query") or "").strip()
        if not query:
            return ToolResult(content="query must not be empty", is_error=True)

        limit = int(input.get("limit", _DEFAULT_LIMIT))
        limit = max(1, min(_HARD_LIMIT, limit))

        descriptors = self._collect_descriptors(context)
        ranked: List[Tuple[int, Dict[str, Any]]] = []
        for desc in descriptors:
            score = _rank(desc, query)
            if score > 0:
                ranked.append((score, desc))

        ranked.sort(key=lambda pair: (-pair[0], str(pair[1].get("name", ""))))
        top = ranked[:limit]

        if not top:
            return ToolResult(
                content=f"No matching tools for {query!r} (searched {len(descriptors)} tools).",
                metadata={
                    "query": query,
                    "results_count": 0,
                    "searched": len(descriptors),
                },
            )

        lines = [f"Matching {len(top)} tool(s) for {query!r}:"]
        for i, (score, desc) in enumerate(top, 1):
            name = desc.get("name", "?")
            d = str(desc.get("description", "")).strip()
            # Keep descriptions on one line for easy scanning.
            one_liner = d.splitlines()[0] if d else "(no description)"
            lines.append(f"{i}. {name} — {one_liner}")

        return ToolResult(
            content="\n".join(lines),
            metadata={
                "query": query,
                "results_count": len(top),
                "results": [
                    {"name": d.get("name"), "score": s, "description": d.get("description")}
                    for s, d in top
                ],
                "searched": len(descriptors),
            },
        )

    def _collect_descriptors(self, context: ToolContext) -> List[Dict[str, Any]]:
        """Prefer live pipeline tools (via ``state_view``), fall back to built-ins.

        The ``state_view`` path is what Stage 10 configures at runtime —
        it reflects MCP servers, custom tools, skills, everything. The
        fallback is just a graceful default for test harnesses that run
        executors without a Stage wrapper.
        """
        view: Optional[Any] = getattr(context, "state_view", None)
        if view is not None:
            tools = getattr(view, "tools", None)
            if isinstance(tools, list) and tools:
                return [d for d in tools if isinstance(d, dict)]
        return _fallback_descriptors()
