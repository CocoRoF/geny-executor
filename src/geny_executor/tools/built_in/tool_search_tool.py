"""ToolSearch — deferred-tool discovery + activation.

Cycle 20260424 executor uplift — Phase 3 Week 6.
2.42.0 — upgraded from a list-introspection helper to the discovery
half of the core/deferred tool contract.

The pipeline no longer ships every tool schema to the LLM upfront.
*Core* tools (framework built-ins by default, plus anything the
manifest's ``tools.core_overrides`` promotes) are always in the
request payload; everything else — host/external tools, provider
bundles, MCP adapters — is registered but *deferred*: dispatchable,
yet invisible to the model until discovered here.

``ToolSearch`` is that discovery path. Pass it a keyword query and it:

1. Searches the FULL catalogue via ``ToolContext.tool_registry``
   (Stage 10 binds the live :class:`ToolRegistry`), deferred tools
   included.
2. Ranks matches (name > description > schema, see :func:`_rank`).
3. **Activates** every deferred match — ``registry.activate(name)``
   bumps the registry version, so Stage 3 rebuilds ``state.tools`` on
   the next loop iteration and the activated schemas reach the model
   on its very next step within the same turn.

Fallback sources when no registry handle is bound (bare harnesses,
older hosts):

1. ``ToolContext.state_view.tools`` — the live API-format descriptors
   the LLM already sees (pre-2.42 behaviour; search only, nothing to
   activate).
2. The built-in catalogue (``BUILT_IN_TOOL_CLASSES``) so the tool
   still produces useful output in isolation.

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

_DEFAULT_LIMIT = 10
_HARD_LIMIT = 100


def _fallback_descriptors() -> List[Dict[str, Any]]:
    """Build descriptors from the built-in catalogue.

    Only used when neither ``tool_registry`` nor ``state_view`` is
    available. Mirrors the shape of the Anthropic API tool descriptor
    so rank logic can treat every source the same.
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
    """Discover tools from the full catalogue — and activate deferred ones.

    Usage: the LLM calls ``ToolSearch({"query": "spreadsheet"})`` and
    gets back something like::

        Matching 2 tool(s) for 'spreadsheet':
        1. xlsx_edit — Edit XLSX workbooks in place. [activated]
        2. Read — Read a file from the filesystem. [available]
        Activated tool schemas become available on your next step.

    Matches already exposed to the model are tagged ``[available]``;
    deferred matches are tagged ``[activated]`` — their full schemas
    enter the tool list on the next loop iteration.
    """

    @property
    def name(self) -> str:
        return "ToolSearch"

    @property
    def description(self) -> str:
        return (
            "Search the full tool catalog by keyword — it contains more "
            "tools than your current tool list. Matching tools that are "
            "not yet in your tool list are activated automatically and "
            "become callable on your next step. Call with NO query to "
            "BROWSE the whole hidden catalog as a compact list (nothing "
            "is activated). Use this whenever no visible tool fits the "
            "task."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Keyword query. An exact tool name is the most "
                        "precise query; multi-word queries prefer results "
                        "matching every token, falling back to any-token "
                        "matches. OMIT (or pass '*') to browse the full "
                        "hidden catalog without activating anything."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        f"Maximum number of matches to return (and thus to "
                        f"activate). Default {_DEFAULT_LIMIT}, hard cap "
                        f"{_HARD_LIMIT}. Keep it small to avoid bloating "
                        f"your tool list."
                    ),
                    "exclusiveMinimum": 0,
                },
            },
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(
            concurrency_safe=True,
            read_only=True,
            idempotent=True,
        )

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        query = (input.get("query") or "").strip()

        registry = getattr(context, "tool_registry", None)
        descriptors = self._collect_descriptors(context, registry)

        # Browse mode — no query (or an explicit wildcard): return the
        # hidden catalog as a compact, grouped list. Read-only: nothing is
        # activated, so browsing costs no tool-list bloat. This is the
        # "what exists beyond my list?" answer the model can't get from
        # keyword search (you can't search for what you don't know exists).
        if not query or query in ("*", "list", "all"):
            return self._browse(descriptors, registry)

        limit = int(input.get("limit", _DEFAULT_LIMIT))
        limit = max(1, min(_HARD_LIMIT, limit))

        ranked: List[Tuple[int, Dict[str, Any]]] = []
        for desc in descriptors:
            score = _rank(desc, query)
            if score > 0:
                ranked.append((score, desc))

        fuzzy = False
        if not ranked and len(query.split()) > 1:
            # AND matching found nothing — fall back to any-token (OR)
            # matching so a single off token doesn't zero the search.
            for desc in descriptors:
                score = max(_rank(desc, tok) for tok in query.split())
                if score > 0:
                    ranked.append((score, desc))
            fuzzy = bool(ranked)

        ranked.sort(key=lambda pair: (-pair[0], str(pair[1].get("name", ""))))
        top = ranked[:limit]

        if not top:
            return ToolResult(
                content=(
                    f"No matching tools for {query!r} (searched "
                    f"{len(descriptors)} tools). Call ToolSearch with no "
                    "query to browse the full catalog."
                ),
                metadata={
                    "query": query,
                    "results_count": 0,
                    "searched": len(descriptors),
                },
            )

        # Promote deferred matches: activate() bumps the registry
        # version, Stage 3 rebuilds state.tools next iteration, and the
        # schemas reach the model on its next step within this turn.
        activated: List[str] = []
        if registry is not None:
            for _, desc in top:
                name = str(desc.get("name", ""))
                if desc.get("_deferred") and self._activate(registry, name):
                    activated.append(name)

        header = f"Matching {len(top)} tool(s) for {query!r}"
        if fuzzy:
            header += " (fuzzy: matched any keyword)"
        lines = [header + ":"]
        for i, (score, desc) in enumerate(top, 1):
            name = desc.get("name", "?")
            d = str(desc.get("description", "")).strip()
            # Keep descriptions on one line for easy scanning.
            one_liner = d.splitlines()[0] if d else "(no description)"
            tag = ""
            if registry is not None:
                tag = " [activated]" if name in activated else " [available]"
            lines.append(f"{i}. {name} — {one_liner}{tag}")
        if activated:
            lines.append(
                "Activated tool schemas become available on your next step — "
                "call them then."
            )

        return ToolResult(
            content="\n".join(lines),
            metadata={
                "query": query,
                "results_count": len(top),
                "results": [
                    {"name": d.get("name"), "score": s, "description": d.get("description")}
                    for s, d in top
                ],
                "activated": activated,
                "searched": len(descriptors),
                "fuzzy": fuzzy,
            },
        )

    @staticmethod
    def _one_liner(desc: Dict[str, Any], max_chars: int = 80) -> str:
        d = str(desc.get("description", "")).strip()
        line = d.splitlines()[0] if d else "(no description)"
        return line[: max_chars - 1] + "…" if len(line) > max_chars else line

    def _browse(
        self, descriptors: List[Dict[str, Any]], registry: Optional[Any]
    ) -> ToolResult:
        """Compact grouped catalog of the tools NOT in the model's list."""
        hidden = [d for d in descriptors if d.get("_deferred")]
        if not hidden:
            return ToolResult(
                content=(
                    "Your tool list already contains every registered tool — "
                    "nothing hidden to browse."
                ),
                metadata={"mode": "browse", "hidden_count": 0},
            )
        # Group by name prefix (text before the first '_') so families
        # read as one scannable block.
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for d in sorted(hidden, key=lambda x: str(x.get("name", ""))):
            fam = str(d.get("name", "")).split("_", 1)[0]
            groups.setdefault(fam, []).append(d)
        lines = [
            f"{len(hidden)} hidden tool(s) — not in your list until activated. "
            "Activate with ToolSearch(\"<keyword or exact name>\"); the schema "
            "arrives on your next step:"
        ]
        for fam in sorted(groups):
            for d in groups[fam]:
                lines.append(f"- {d.get('name')} — {self._one_liner(d)}")
        return ToolResult(
            content="\n".join(lines),
            metadata={
                "mode": "browse",
                "hidden_count": len(hidden),
                "hidden": [str(d.get("name")) for d in hidden],
            },
        )

    @staticmethod
    def _activate(registry: Any, name: str) -> bool:
        """Best-effort ``registry.activate(name)`` — never breaks the search."""
        activate = getattr(registry, "activate", None)
        if not callable(activate):
            return False
        try:
            return bool(activate(name))
        except Exception:
            return False

    def _collect_descriptors(
        self, context: ToolContext, registry: Optional[Any]
    ) -> List[Dict[str, Any]]:
        """Prefer the live registry (full catalogue), then ``state_view``, then built-ins.

        The registry path is what Stage 10 configures at runtime — it
        covers MCP tools, custom host tools, skills, everything,
        *including deferred tools whose schemas the model has not seen*.
        Each descriptor is annotated with a private ``_deferred`` flag
        so ``execute`` knows which matches need activation. The
        ``state_view`` path is the pre-2.42 behaviour (exposed tools
        only); the built-in catalogue is a graceful default for test
        harnesses running executors without a Stage wrapper.
        """
        if registry is not None:
            try:
                out: List[Dict[str, Any]] = []
                for tool in registry.list_all():
                    desc = tool.to_api_format()
                    if isinstance(desc, dict):
                        is_exposed = getattr(registry, "is_exposed", None)
                        desc["_deferred"] = (
                            not bool(is_exposed(tool.name)) if callable(is_exposed) else False
                        )
                        out.append(desc)
                if out:
                    return out
            except Exception:
                pass  # fall through to the view/built-in sources
        view: Optional[Any] = getattr(context, "state_view", None)
        if view is not None:
            tools = getattr(view, "tools", None)
            if isinstance(tools, list) and tools:
                return [d for d in tools if isinstance(d, dict)]
        return _fallback_descriptors()
