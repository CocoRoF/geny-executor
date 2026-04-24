"""Tool base class and types.

Cycle 20260424 executor uplift — Phase 1 Week 1:
The ``Tool`` ABC has been extended to carry richer runtime metadata
(concurrency safety, destructiveness, permission matchers, lifecycle
hooks, render hints). All additions are **additive with defaults** —
existing Tool subclasses continue to work without modification.

See ``Geny/executor_uplift/06_design_tool_system.md`` + ``12_detailed_plan.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────
# Capability + Permission primitives (new)
# ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolCapabilities:
    """Runtime traits describing how a tool should be orchestrated.

    Stage 10 (Tool) inspects these to decide between parallel and
    serial batches, enforce budgets, and forward structured metadata
    to downstream stages (Tool Review, Guard, Permission matrix).

    Attributes:
        concurrency_safe: Tool may run in parallel with other
            concurrency-safe tools in the same turn. Default ``False``
            (fail-closed — serialize until an author explicitly opts in).
        read_only: Tool has no observable side effects on filesystem,
            network, or process state. Implies ``concurrency_safe``
            semantically but is tracked independently for clarity.
        destructive: Tool may irrecoverably delete or overwrite data.
            Guard / Permission / HITL stages escalate on this flag.
        idempotent: Repeated invocation with the same input yields the
            same result (safe to retry).
        network_egress: Tool performs outbound network I/O. Useful for
            egress audits + air-gapped deployments.
        interrupt: Behaviour on cancellation — ``"cancel"`` stops work
            immediately, ``"block"`` refuses to be interrupted (e.g. a
            write that must complete atomically).
        max_result_chars: If the tool's ``display_text`` exceeds this,
            Stage 10 persists the full result to disk and returns the
            path instead. ``0`` disables the limit (infinite — use for
            Read-style tools whose output is already bounded).
    """

    concurrency_safe: bool = False
    read_only: bool = False
    destructive: bool = False
    idempotent: bool = False
    network_egress: bool = False
    interrupt: str = "block"
    max_result_chars: int = 100_000


@dataclass(frozen=True)
class PermissionDecision:
    """Outcome of a permission check for a specific tool invocation.

    Attributes:
        behavior: One of ``"allow"``, ``"deny"``, ``"ask"``. ``"ask"``
            defers the decision to the HITL stage (human approval).
        updated_input: If set, the tool is invoked with this payload
            instead of the original (permission layer can sanitize).
        reason: Human-readable explanation for logs and UI.
    """

    behavior: str
    updated_input: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None


# ─────────────────────────────────────────────────────────────────
# Tool execution context
# ─────────────────────────────────────────────────────────────────


@dataclass
class ToolContext:
    """Context passed to tool execution.

    Cycle 20260424 additions (all optional, defaults preserve old
    behaviour):
        ``permission_mode`` — ``"default" | "plan" | "auto" | "bypass"``.
        ``state_view`` — Read-only handle to ``PipelineState`` (tools
            may introspect but MUST NOT mutate; use ``ToolResult.
            state_mutations`` to propose writes).
        ``event_emit`` — Callback to emit structured events
            (``(event_type, payload)``) during long-running tools.
        ``parent_tool_use_id`` — ID of the LLM tool_use block that
            triggered this call; useful for audit linking.
        ``extras`` — Free-form bag for host-specific data (e.g. Geny
            injects creature_role, mutation_buffer handles).

    Attributes:
        session_id: Unique session identifier.
        working_dir: Working directory for file operations. Tools should
            resolve relative paths against this directory.
        storage_path: Session-specific storage directory (e.g. for logs,
            session state files). May differ from working_dir.
        env_vars: Environment variables to inject when spawning
            subprocesses (e.g. GITHUB_TOKEN, ANTHROPIC_API_KEY).
        allowed_paths: If set, tools MUST restrict file system access to
            these directories. An empty list means no restriction.
        metadata: Arbitrary key-value metadata forwarded to tools.
    """

    session_id: str = ""
    working_dir: str = ""
    storage_path: Optional[str] = None
    env_vars: Optional[Dict[str, str]] = None
    allowed_paths: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    stage_order: int = 0
    stage_name: str = ""
    # Cycle 20260424 additions (optional; default behaviour unchanged)
    permission_mode: str = "default"
    state_view: Optional[Any] = None
    event_emit: Optional[Callable[[str, Dict[str, Any]], None]] = None
    parent_tool_use_id: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────
# Tool result
# ─────────────────────────────────────────────────────────────────


@dataclass
class ToolResult:
    """Result of a tool execution.

    Cycle 20260424 additions (all optional):
        ``display_text`` — Compact representation suitable for the LLM's
            ``tool_result`` block. When absent, ``content`` is serialized
            via ``to_api_format``. Kept separate so tools can return a
            rich ``content`` payload for host consumers while sending a
            summary to the model.
        ``persist_full`` — Path where the full result was persisted if
            it exceeded ``ToolCapabilities.max_result_chars``. Stage 10
            populates this automatically; tools rarely set it directly.
        ``state_mutations`` — Dict of ``state.shared`` updates proposed
            by the tool. Stage 10 applies them after successful
            permission + review checks.
        ``artifacts`` — Files / objects the tool produced outside of
            ``content`` (paths, IDs, resource URIs).
        ``new_messages`` — Messages to inject into the conversation
            history (e.g. Skill tool returns prompt blocks for the next
            iteration).
        ``mcp_meta`` — Pass-through metadata from MCP server responses
            (``_meta``, ``structuredContent``).
    """

    content: Any = ""
    is_error: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Cycle 20260424 additions
    display_text: Optional[str] = None
    persist_full: Optional[str] = None
    state_mutations: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    new_messages: List[Dict[str, Any]] = field(default_factory=list)
    mcp_meta: Optional[Dict[str, Any]] = None

    def to_api_format(self, tool_use_id: str) -> Dict[str, Any]:
        """Convert to Anthropic API tool_result format.

        Cycle 20260424: when ``display_text`` is set, it is used as the
        ``content`` payload (this keeps the LLM-facing summary in sync
        with large ``content`` objects).

        Structured-error payloads (``content`` is a dict with a top-level
        ``"error"`` object containing ``code`` and ``message``) are
        rendered with a leading ``ERROR <code>: <message>`` header line so
        the model has a predictable affordance to detect failure without
        parsing the JSON body.
        """
        result: Dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
        }

        # Prefer explicit display_text when provided (e.g. persisted
        # results send a short notice + path instead of the full body).
        if self.display_text is not None:
            result["content"] = self.display_text
            if self.is_error:
                result["is_error"] = True
            return result

        content = self.content
        if isinstance(content, str):
            result["content"] = content
        elif isinstance(content, list):
            result["content"] = content
        elif isinstance(content, dict):
            import json as _json

            err_block = content.get("error")
            if (
                isinstance(err_block, dict)
                and isinstance(err_block.get("code"), str)
                and isinstance(err_block.get("message"), str)
            ):
                header = f"ERROR {err_block['code']}: {err_block['message']}"
                body = _json.dumps(content, ensure_ascii=False, default=str)
                result["content"] = f"{header}\n{body}"
            else:
                result["content"] = _json.dumps(content, ensure_ascii=False, default=str)
        else:
            result["content"] = str(content)

        if self.is_error:
            result["is_error"] = True

        return result


# ─────────────────────────────────────────────────────────────────
# Tool ABC
# ─────────────────────────────────────────────────────────────────


class Tool(ABC):
    """Tool interface — maps 1:1 to Anthropic API tool definitions.

    Implement this to create custom tools that Claude can call.

    Cycle 20260424 — the ABC now carries optional metadata hooks for
    Stage 10 orchestration, Permission matrix, Tool Review (stage 11),
    and UI rendering. All new methods / attributes provide sensible
    defaults so existing Tool subclasses keep working without change.
    """

    # Optional — alternate names the tool also responds to.
    aliases: Tuple[str, ...] = ()

    # MCP metadata (populated by MCPToolAdapter at wire time).
    is_mcp: bool = False
    mcp_info: Optional[Dict[str, Any]] = None

    # ── Required contract ─────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool unique name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description shown to the model."""
        ...

    @property
    @abstractmethod
    def input_schema(self) -> Dict[str, Any]:
        """JSON Schema for tool input parameters."""
        ...

    @abstractmethod
    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        """Execute the tool with given input."""
        ...

    # ── Optional overrides (sensible defaults) ───────────────────

    def output_schema(self) -> Optional[Dict[str, Any]]:
        """Optional JSON Schema describing successful ``ToolResult.content``.

        When provided, Stage 11 (Tool Review) validates outputs against
        this schema. Default is ``None`` — no structural check.
        """
        return None

    def validate_input(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-execution input validation hook.

        Default is pass-through. Tools can override to normalize or
        reject bad inputs early. Raising any exception aborts the
        tool call with that error propagated as ``ToolResult.is_error``.
        """
        return raw

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        """Describe how this invocation should be orchestrated.

        Capabilities are *input-dependent* — the same tool may be
        ``read_only=True`` for ``ls`` and ``destructive=True`` for
        ``rm -rf``. Default is the fail-closed baseline.
        """
        return ToolCapabilities()

    async def check_permissions(
        self, input: Dict[str, Any], context: ToolContext
    ) -> PermissionDecision:
        """Return this tool's opinion on whether the invocation is allowed.

        Called after the global Permission matrix (Stage 4 Guard) has
        consulted rule sources. Tool-level logic acts as the final
        gate for input it understands best.

        Default allows everything; tools representing side-effects
        should override.
        """
        return PermissionDecision(behavior="allow")

    async def prepare_permission_matcher(
        self, input: Dict[str, Any]
    ) -> Callable[[str], bool]:
        """Build a closure that tests permission rule patterns.

        Rule patterns look like ``"Bash(git *)"`` — the matcher decides
        whether *this particular invocation* satisfies the pattern.

        Default matches exact tool names only; tools with structured
        inputs (Bash, FileEdit) override to support sub-patterns.
        """
        tool_name = self.name

        def _match(pattern: str) -> bool:
            return pattern == tool_name

        return _match

    # ── Lifecycle hooks (fired by Stage 10 orchestrator) ─────────

    async def on_enter(self, input: Dict[str, Any], context: ToolContext) -> None:
        """Hook fired just before ``execute``. Default is no-op."""
        return None

    async def on_exit(self, result: ToolResult, context: ToolContext) -> None:
        """Hook fired after ``execute`` completes successfully. Default no-op."""
        return None

    async def on_error(self, error: BaseException, context: ToolContext) -> None:
        """Hook fired when ``execute`` raises. Default no-op."""
        return None

    # ── UI / display hints ───────────────────────────────────────

    def user_facing_name(self, input: Dict[str, Any]) -> str:
        """Short label for UIs / logs. Default returns ``self.name``."""
        return self.name

    def activity_description(self, input: Dict[str, Any]) -> Optional[str]:
        """One-line description of what this invocation is doing.

        Useful for progress indicators in interactive hosts. Default is
        ``None`` (UI falls back to ``user_facing_name``).
        """
        return None

    def is_enabled(self) -> bool:
        """Toggle a tool off without unregistering it.

        Stage 3 (System) filters out tools whose ``is_enabled()`` returns
        False when assembling the tools array for the model.
        """
        return True

    # ── API format ───────────────────────────────────────────────

    def to_api_format(self) -> Dict[str, Any]:
        """Convert to Anthropic API tools parameter format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


# ─────────────────────────────────────────────────────────────────
# build_tool() factory — lightweight construction without subclassing
# ─────────────────────────────────────────────────────────────────


def build_tool(
    *,
    name: str,
    description: str,
    input_schema: Dict[str, Any],
    execute: Callable[[Dict[str, Any], ToolContext], Awaitable[ToolResult]],
    capabilities: Optional[ToolCapabilities] = None,
    aliases: Tuple[str, ...] = (),
    output_schema: Optional[Dict[str, Any]] = None,
    check_permissions: Optional[
        Callable[[Dict[str, Any], ToolContext], Awaitable[PermissionDecision]]
    ] = None,
    prepare_permission_matcher: Optional[
        Callable[[Dict[str, Any]], Awaitable[Callable[[str], bool]]]
    ] = None,
    validate_input: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    on_enter: Optional[Callable[[Dict[str, Any], ToolContext], Awaitable[None]]] = None,
    on_exit: Optional[Callable[[ToolResult, ToolContext], Awaitable[None]]] = None,
    on_error: Optional[Callable[[BaseException, ToolContext], Awaitable[None]]] = None,
    user_facing_name: Optional[Callable[[Dict[str, Any]], str]] = None,
    activity_description: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
    is_enabled: Optional[Callable[[], bool]] = None,
) -> Tool:
    """Construct a ``Tool`` instance without defining a subclass.

    Lifecycle hooks, permission callbacks, and capability descriptors
    are all optional — sensible defaults kick in when omitted.

    Example:
        search = build_tool(
            name="search",
            description="Search the indexed docs.",
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
            execute=_search_impl,
            capabilities=ToolCapabilities(concurrency_safe=True, read_only=True),
        )
    """
    caps = capabilities or ToolCapabilities()

    class _BuiltTool(Tool):
        pass

    # Install static attributes on the *class* so multiple instances
    # share them (matching how subclass-based tools behave).
    _BuiltTool.aliases = aliases  # type: ignore[assignment]

    # Properties can't be installed via ``setattr`` on instance so we
    # mint a fresh class-level property for each field.
    def _make_prop(value: Any) -> property:
        return property(lambda self, _v=value: _v)

    _BuiltTool.name = _make_prop(name)  # type: ignore[assignment]
    _BuiltTool.description = _make_prop(description)  # type: ignore[assignment]
    _BuiltTool.input_schema = _make_prop(input_schema)  # type: ignore[assignment]

    async def _execute(self: Tool, inp: Dict[str, Any], ctx: ToolContext) -> ToolResult:
        return await execute(inp, ctx)

    _BuiltTool.execute = _execute  # type: ignore[assignment]

    def _capabilities(self: Tool, _inp: Dict[str, Any]) -> ToolCapabilities:
        return caps

    _BuiltTool.capabilities = _capabilities  # type: ignore[assignment]

    if output_schema is not None:
        _BuiltTool.output_schema = (  # type: ignore[assignment]
            lambda self, _s=output_schema: _s
        )
    if check_permissions is not None:
        _BuiltTool.check_permissions = (  # type: ignore[assignment]
            lambda self, inp, ctx, _f=check_permissions: _f(inp, ctx)
        )
    if prepare_permission_matcher is not None:
        _BuiltTool.prepare_permission_matcher = (  # type: ignore[assignment]
            lambda self, inp, _f=prepare_permission_matcher: _f(inp)
        )
    if validate_input is not None:
        _BuiltTool.validate_input = (  # type: ignore[assignment]
            lambda self, raw, _f=validate_input: _f(raw)
        )
    if on_enter is not None:
        _BuiltTool.on_enter = (  # type: ignore[assignment]
            lambda self, inp, ctx, _f=on_enter: _f(inp, ctx)
        )
    if on_exit is not None:
        _BuiltTool.on_exit = (  # type: ignore[assignment]
            lambda self, res, ctx, _f=on_exit: _f(res, ctx)
        )
    if on_error is not None:
        _BuiltTool.on_error = (  # type: ignore[assignment]
            lambda self, err, ctx, _f=on_error: _f(err, ctx)
        )
    if user_facing_name is not None:
        _BuiltTool.user_facing_name = (  # type: ignore[assignment]
            lambda self, inp, _f=user_facing_name: _f(inp)
        )
    if activity_description is not None:
        _BuiltTool.activity_description = (  # type: ignore[assignment]
            lambda self, inp, _f=activity_description: _f(inp)
        )
    if is_enabled is not None:
        _BuiltTool.is_enabled = (  # type: ignore[assignment]
            lambda self, _f=is_enabled: _f()
        )

    # ABC metaclass computes ``__abstractmethods__`` at class creation
    # time; since we inject the 4 required members *after* subclassing
    # Tool, the original ``{name, description, input_schema, execute}``
    # abstract set is still recorded. Clear it so instantiation works.
    _BuiltTool.__abstractmethods__ = frozenset()  # type: ignore[attr-defined]

    return _BuiltTool()
