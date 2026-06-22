"""Pipeline engine — executes stages in order with loop control."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Deque,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from geny_executor.core.config import ModelOverrides, PipelineConfig
from geny_executor.core.errors import (
    ExecutorErrorCode,
    GenyExecutorError,
    StageError,
)
from geny_executor.core.result import PipelineResult
from geny_executor.core.shared_keys import SharedKeys
from geny_executor.core.stage import Stage, StageDescription
from geny_executor.core.state import PipelineState
from geny_executor.events.bus import EventBus
from geny_executor.events.types import PipelineEvent
from geny_executor.hooks.events import HookEvent, HookEventPayload
from geny_executor.llm_client.credentials import (
    ConfigError,
    CredentialBundle,
    ProviderCredentials,
)
from geny_executor.llm_client.registry import ClientRegistry

if TYPE_CHECKING:
    from geny_executor.core.environment import EnvironmentManifest, StageManifestEntry
    from geny_executor.tools.provider import ToolProvider
    from geny_executor.tools.providers import AdhocToolProvider
    from geny_executor.tools.registry import ToolRegistry


logger = logging.getLogger(__name__)


#: Sentinel pushed into every ``Pipeline.events()`` subscriber queue at
#: ``aclose()`` so tap generators terminate instead of awaiting forever
#: on a pipeline that will never emit again.
_TAP_CLOSED = object()


def _error_event_data(exc: Exception) -> Dict[str, Any]:
    """Build a structured event payload for ``pipeline.error`` /
    ``stage.error`` / similar terminal-failure events.

    Carries:
      - ``error``: stringified message (legacy field, preserved for
        backward compat — every existing consumer reads this).
      - ``code``: the stable :class:`ExecutorErrorCode` value when the
        exception is a :class:`GenyExecutorError` subclass; otherwise
        ``"exec.unknown"``. Hosts use this for i18n / telemetry
        grouping without parsing the message text.
      - ``exception_type``: fully qualified class name, useful for
        ad-hoc filtering when no code is attached.

    Stable since 2.1.0 — adding fields is non-breaking, removing
    fields is a major-version change.
    """
    code_str = ExecutorErrorCode.EXEC_UNKNOWN.value
    if isinstance(exc, GenyExecutorError) and exc.code is not None:
        code_str = exc.code.value
    return {
        "error": str(exc),
        "code": code_str,
        "exception_type": f"{type(exc).__module__}.{type(exc).__name__}",
    }


def _pipeline_config_from_manifest(manifest: "EnvironmentManifest") -> PipelineConfig:
    """Build a :class:`PipelineConfig` from manifest pipeline+model blocks.

    The manifest stores ``pipeline`` and ``model`` as plain dicts; reunite
    them into the nested ``PipelineConfig(model=ModelConfig(...))`` shape the
    runtime expects.

    Credentials no longer flow through ``PipelineConfig.api_key`` — they live
    in the :class:`CredentialBundle` passed to ``from_manifest_async`` and are
    consulted by ``_resolve_llm_client`` based on
    ``stages[6].config["provider"]``.
    """
    raw = dict(manifest.pipeline or {})
    if manifest.model:
        # ``pipeline.model`` (if present) loses to the top-level ``model``
        # block — the latter is the canonical location in v2 manifests.
        raw["model"] = dict(manifest.model)
    # Drop any stale api_key the legacy manifest format may have embedded.
    raw.pop("api_key", None)
    return PipelineConfig.from_dict(raw)


def _creds_to_client_kwargs(provider: str, creds: ProviderCredentials) -> Dict[str, Any]:
    """Map ``ProviderCredentials`` into vendor-shaped kwargs for client construction.

    Each provider's client takes a slightly different constructor surface;
    this is the single place those shapes are encoded.
    """
    # Branded local (OpenAI-compatible) providers — ollama / lmstudio /
    # custom (+ aliases). Their constructor takes base_url + the
    # num_ctx / think wire knobs; the mapping lives next to the profiles.
    from geny_executor.llm_client.profiles import (
        is_profiled_provider,
        profiled_client_kwargs,
    )

    if is_profiled_provider(provider):
        return profiled_client_kwargs(provider, creds)

    if provider == "vllm":
        kwargs: Dict[str, Any] = {}
        if creds.api_key:
            kwargs["api_key"] = creds.api_key
        if creds.base_url is not None:
            kwargs["base_url"] = creds.base_url
        if creds.default_headers is not None:
            kwargs["default_headers"] = dict(creds.default_headers)
        return kwargs

    if provider == "claude_code_cli":
        extras = dict(creds.extras or {})
        kwargs = {"api_key": creds.api_key}
        if creds.binary_path:
            kwargs["binary_path"] = creds.binary_path
        # ``auth_mode`` lives on ProviderCredentials itself (2.2.0) — it is
        # the host's declaration of which credential channel drives the
        # CLI ('api_key' | 'oauth' | 'setup_token' | 'auto'). Without this
        # threading, the client falls back to 'auto' (key-presence
        # resolution) and an explicit subscription declaration from the
        # host's auth modal would be silently ignored.
        if getattr(creds, "auth_mode", "auto") != "auto":
            kwargs["auth_mode"] = creds.auth_mode
        # Map known extras to constructor kwargs; unknown extras pass through
        # to ``extra_args`` (caller's escape hatch).
        for key in (
            "workspace_dir",
            "workspace_root",
            "settings_path",
            "bare_mode",
            "max_budget_usd",
            "default_permission_mode",
            "mcp_config",
            "allow_tools",
            "disallow_tools",
            "extra_args",
            "timeout_s",
            "strict_wire",
        ):
            if key in extras:
                # workspace_root is the settings-side name; the client takes workspace_dir
                if key == "workspace_root":
                    kwargs["workspace_dir"] = extras[key]
                else:
                    kwargs[key] = extras[key]
        return kwargs

    # API providers (anthropic / openai / google)
    kwargs = {"api_key": creds.api_key}
    if creds.base_url is not None:
        kwargs["base_url"] = creds.base_url
    if creds.default_headers is not None:
        kwargs["default_headers"] = dict(creds.default_headers)
    return kwargs


def _validate_manifest_provider_locations(manifest: "EnvironmentManifest") -> None:
    """Reject manifests that store provider in the legacy ``strategies`` slot.

    The single source of truth is ``stages[6].config["provider"]``. Manifests
    with ``strategies["provider"]`` are rejected at strict load time so the
    silent-divergence class of bug is impossible.
    """
    for entry in manifest.stage_entries():
        strategies = entry.strategies or {}
        if "provider" in strategies:
            raise ConfigError(
                f"Stage {entry.order} ({entry.name!r}) uses legacy "
                f"strategies['provider']={strategies['provider']!r}. "
                "Move it to config['provider']; strategies['provider'] is no "
                "longer recognised."
            )
        if entry.name == "api" and entry.active:
            cfg = entry.config or {}
            if not cfg.get("provider"):
                raise ConfigError(
                    "Stage 6 ('api') is active but no provider is configured. "
                    "Set stages[6].config['provider'] (e.g. 'anthropic')."
                )


def _mcp_configs_from_manifest(manifest: "EnvironmentManifest") -> Dict[str, Any]:
    """Extract ``MCPServerConfig`` instances from ``manifest.tools.mcp_servers``.

    Manifests store MCP server definitions as plain dicts; the manager
    expects :class:`MCPServerConfig` dataclasses keyed by name. Entries
    missing a ``name`` are skipped silently (they cannot be routed to
    anyway).
    """
    from geny_executor.tools.mcp.manager import MCPServerConfig

    configs: Dict[str, MCPServerConfig] = {}
    for raw in manifest.tools.mcp_servers or []:
        if not isinstance(raw, dict):
            continue
        name = raw.get("name")
        if not name:
            continue
        configs[name] = MCPServerConfig(
            name=name,
            command=raw.get("command", ""),
            args=list(raw.get("args", [])),
            env=dict(raw.get("env", {})),
            transport=raw.get("transport", "stdio"),
            url=raw.get("url", ""),
            headers=dict(raw.get("headers", {})),
        )
    return configs


def _provider_wants_mcp_passthrough(provider_name: str) -> bool:
    """True when *provider_name*'s client class executes tools inside its
    own subprocess AND can receive MCP servers via its own config channel.

    The 2.2.1 incident this gates on: a manifest declared
    ``tools.mcp_servers`` on a ``claude_code_cli`` environment. The
    pipeline dutifully connected the server HOST-side (spawning the MCP
    child inside the host process) and registered its tools into the
    pipeline ToolRegistry — but Stage 10 never dispatches for subprocess
    backends (the CLI runs its own agentic loop), and the CLI subprocess
    only sees servers passed through ``--mcp-config``. Net effect: the
    user attached an MCP server in the environment editor, the session
    built cleanly, and the LLM had no idea the tools existed.

    For such providers the manifest's MCP servers must be PASSED THROUGH
    to the client instead of host-connected. Detection is class-level
    (no client construction, no credentials needed): ``is_subprocess``
    AND ``supports_mcp_passthrough`` on the registered capabilities.
    Unknown/unregistered providers return False — the host-side path is
    the safe default.
    """
    if not provider_name or provider_name not in ClientRegistry.available():
        return False
    try:
        caps = ClientRegistry.get(provider_name).capabilities
    except Exception:  # noqa: BLE001 — registry factories may raise on missing extras
        return False
    return bool(
        getattr(caps, "is_subprocess", False) and getattr(caps, "supports_mcp_passthrough", False)
    )


def _mcp_servers_to_cli_config(
    configs: Dict[str, Any],
) -> tuple[Dict[str, Any], List[str]]:
    """Translate manifest ``MCPServerConfig`` entries into the Claude Code
    CLI's ``--mcp-config`` JSON shape.

    Returns ``(cli_config, skipped_names)`` where ``cli_config`` is
    ``{"mcpServers": {name: entry}}`` and ``skipped_names`` lists servers
    whose transport could not be expressed (logged by the caller).

    Shape mapping (per the CLI's documented mcp-config schema):
      stdio      → {"type": "stdio", "command", "args", "env"}
      sse        → {"type": "sse", "url", "headers"}
      http       → {"type": "http", "url", "headers"}
    """
    servers: Dict[str, Any] = {}
    skipped: List[str] = []
    for name, cfg in configs.items():
        transport = (getattr(cfg, "transport", "") or "stdio").lower()
        if transport == "stdio":
            entry: Dict[str, Any] = {
                "type": "stdio",
                "command": cfg.command,
                "args": list(cfg.args or []),
            }
            if cfg.env:
                entry["env"] = dict(cfg.env)
        elif transport in ("sse", "http"):
            entry = {"type": transport, "url": cfg.url}
            if cfg.headers:
                entry["headers"] = dict(cfg.headers)
        else:
            skipped.append(name)
            continue
        servers[name] = entry
    return {"mcpServers": servers}, skipped


def _merge_cli_mcp_config(
    host_config: Any,
    manifest_config: Dict[str, Any],
) -> Any:
    """Merge manifest-declared MCP servers into a host-supplied CLI
    mcp-config.

    Precedence: the HOST config wins on server-name collision — a host
    that wires a session-scoped bridge (Geny's ``geny`` server) or
    deliberately overrides a server keeps control. ``host_config`` may
    be:

    - ``None`` / empty → the manifest config is used as-is.
    - a dict ``{"mcpServers": {...}}`` → key-merged.
    - a file path (str) → the file is read and key-merged; on any read/
      parse failure the host path is returned unchanged and the manifest
      servers are dropped with a warning (the CLI accepts exactly one
      ``--mcp-config``, so an unreadable host file cannot be merged
      without guessing).
    """
    manifest_servers = dict(manifest_config.get("mcpServers", {}))
    if not manifest_servers:
        return host_config
    if not host_config:
        return {"mcpServers": manifest_servers}

    if isinstance(host_config, str):
        try:
            with open(host_config, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if not isinstance(loaded, dict):
                raise ValueError("mcp-config file is not a JSON object")
            host_config = loaded
        except Exception:  # noqa: BLE001
            logger.warning(
                "CLI MCP passthrough: host mcp_config path %r could not be "
                "read/parsed — manifest-declared MCP servers %s will NOT "
                "reach the CLI subprocess (the CLI accepts a single "
                "--mcp-config).",
                host_config,
                sorted(manifest_servers),
            )
            return host_config

    if not isinstance(host_config, dict):
        logger.warning(
            "CLI MCP passthrough: host mcp_config has unsupported type %s — "
            "manifest MCP servers %s dropped.",
            type(host_config).__name__,
            sorted(manifest_servers),
        )
        return host_config

    merged = dict(host_config)
    host_servers = dict(merged.get("mcpServers", {}))
    for name, entry in manifest_servers.items():
        if name in host_servers:
            logger.info(
                "CLI MCP passthrough: server %r declared in both the host "
                "mcp_config and the manifest — host definition wins.",
                name,
            )
            continue
        host_servers[name] = entry
    merged["mcpServers"] = host_servers
    return merged


@dataclass
class ToolResolutionReport:
    """What happened to every tool name the manifest asked for.

    Registration used to warn-and-pray (audit §3.5): a manifest could
    declare ten external tools, have zero of them resolve, and the
    pipeline would build fine with the only evidence buried in logs.
    This report makes the outcome a first-class artifact — stored on
    ``pipeline.tool_resolution_report`` after ``from_manifest`` /
    ``from_manifest_async`` so hosts can render it (env editor
    diagnostics panel) or assert on it (deploy smoke checks).

    Fields:
      - ``resolved``: names successfully registered (built-in +
        external, in registration order).
      - ``unresolved``: names that could not be resolved — unknown
        built-in names, external names no provider claimed.
      - ``shadowed``: names where one registration displaced or
        pre-empted another (a built-in skipped because the registry
        already held the name, or an external entry overwriting an
        earlier registration). A name can appear in both ``resolved``
        and ``shadowed`` when an external override replaced a
        built-in — the name *is* live, just not from the source that
        registered first.
      - ``required_unresolved``: the subset of ``unresolved`` that the
        manifest marked ``{"name": ..., "required": true}``. Non-empty
        here means strict mode would have refused to build.
    """

    resolved: List[str] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)
    shadowed: List[str] = field(default_factory=list)
    required_unresolved: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class DroppedStage:
    """Record of a manifest stage that failed to construct in lenient mode.

    ``from_manifest(strict=False)`` used to swallow stage-construction
    failures with a bare ``continue`` — a recovery load could silently
    produce a pipeline missing half its stages with zero evidence
    (2.2.0, audit 2026-06-09 §1-1 masked-degradation class). Lenient
    mode still drops the stage (that is its contract), but every drop
    now warns and lands here, on ``pipeline.dropped_stages``, so hosts
    can render the degradation or refuse to serve the session.

    Fields:
      - ``name``: manifest stage name (e.g. ``"api"``).
      - ``order``: manifest slot order (1-21).
      - ``error``: ``"ExcType: message"`` summary of the construction
        failure (full traceback goes to the warning log only).
    """

    name: str
    order: int
    error: str


def _register_built_in_tools(
    manifest: "EnvironmentManifest",
    registry: "ToolRegistry",
    report: Optional[ToolResolutionReport] = None,
) -> None:
    """Register framework-shipped tools named in ``manifest.tools.built_in``.

    The executor ships a baseline toolkit (:data:`~geny_executor.tools.
    built_in.BUILT_IN_TOOL_CLASSES`) — filesystem ops, shell, and
    search — so consumers do not have to reimplement basic tools
    against the :class:`~geny_executor.tools.base.Tool` ABC. The
    manifest opts into which of those ship into this pipeline.

    Accepted values for ``manifest.tools.built_in``:
      * ``["*"]`` — register every class in
        :data:`~geny_executor.tools.built_in.BUILT_IN_TOOL_CLASSES`.
      * ``["Read", "Write", ...]`` — register only the named classes.
      * empty list / missing field — no framework tools attached
        (preserves pre-v0.26.3 behaviour for manifests authored before
        built-ins were routable).

    An unknown name warns and is skipped — a manifest error worth
    surfacing but not worth crashing the build. If a name is already
    present in the registry (e.g. an ``AdhocToolProvider`` beat us
    to it) the existing registration wins silently.

    When *report* is given, each name's fate lands in it: registered →
    ``resolved``, unknown → ``unresolved``, skipped-because-present →
    ``shadowed`` (audit §3.5 — registration outcomes must be
    inspectable, not just logged).
    """
    from geny_executor.tools.built_in import BUILT_IN_TOOL_CLASSES

    names = list(getattr(manifest.tools, "built_in", []) or [])
    if not names:
        return

    if names == ["*"]:
        names = list(BUILT_IN_TOOL_CLASSES.keys())

    for name in names:
        cls = BUILT_IN_TOOL_CLASSES.get(name)
        if cls is None:
            logger.warning(
                "manifest.tools.built_in contains unknown name '%s' — expected one of %s",
                name,
                sorted(BUILT_IN_TOOL_CLASSES.keys()),
            )
            if report is not None:
                report.unresolved.append(name)
            continue
        if registry.get(name) is not None:
            if report is not None:
                report.shadowed.append(name)
            continue
        registry.register(cls())
        if report is not None:
            report.resolved.append(name)


def _register_external_tools(
    manifest: "EnvironmentManifest",
    registry: "ToolRegistry",
    providers: Sequence["AdhocToolProvider"],
    report: Optional[ToolResolutionReport] = None,
    strict: bool = False,
) -> None:
    """Register every ``manifest.tools.external`` entry against *providers*.

    Walks ``manifest.tools.external`` in declared order. For each name,
    queries the providers left-to-right and registers the first
    non-``None`` :class:`Tool` they return.

    Entry forms (2.2.0, audit §3.5):
      * ``"name"`` — plain string, **optional** semantics (back-compat):
        if no provider claims it, log a warning and keep building. The
        manifest may legitimately reference a tool a given deployment
        chose not to wire.
      * ``{"name": "...", "required": true}`` — the manifest declares
        the environment is *broken* without this tool. Unresolved +
        ``strict=True`` raises :class:`ConfigError` at build time
        instead of shipping a pipeline whose LLM will hallucinate calls
        into a void; non-strict downgrades to a louder warning so
        recovery tooling can still load the manifest.

    Dict entries are parsed here at the registration site rather than
    in :class:`ToolsSnapshot` — the snapshot's ``external`` field is
    typed ``List[str]`` and round-trips foreign values untouched, so
    the dict form survives serialization without a schema change.

    When *report* is given, every entry's fate is recorded (resolved /
    unresolved / shadowed / required_unresolved).
    """
    raw_entries = list(getattr(manifest.tools, "external", []) or [])
    if not raw_entries:
        return

    # Normalize entries → (name, required). Malformed entries warn and
    # count as unresolved-but-unnamed; they cannot be required because
    # we cannot even tell what the author wanted.
    entries: List[Tuple[str, bool]] = []
    for raw in raw_entries:
        if isinstance(raw, str):
            entries.append((raw, False))
        elif isinstance(raw, Mapping):
            name = str(raw.get("name") or "")
            if not name:
                logger.warning("manifest.tools.external entry %r has no 'name' — skipping", raw)
                continue
            entries.append((name, bool(raw.get("required", False))))
        else:
            logger.warning(
                "manifest.tools.external entry %r is neither a string nor a "
                "{'name', 'required'} mapping — skipping",
                raw,
            )

    if not providers:
        names = [name for name, _ in entries]
        logger.warning(
            "manifest declares %d external tool(s) but no AdhocToolProvider was supplied: %s",
            len(names),
            names,
        )
        required_missing = [name for name, required in entries if required]
        if report is not None:
            report.unresolved.extend(names)
            report.required_unresolved.extend(required_missing)
        if strict and required_missing:
            raise ConfigError(
                f"required external tool(s) {required_missing} declared in "
                "manifest.tools.external but no AdhocToolProvider was supplied. "
                "Pass adhoc_providers= to from_manifest, or mark the entries "
                "optional (plain string / required: false)."
            )
        return

    for name, required in entries:
        tool = None
        for provider in providers:
            tool = provider.get(name)
            if tool is not None:
                break
        if tool is None:
            if report is not None:
                report.unresolved.append(name)
                if required:
                    report.required_unresolved.append(name)
            if required:
                if strict:
                    raise ConfigError(
                        f"required external tool '{name}' was declared in "
                        "manifest.tools.external but no AdhocToolProvider "
                        "supplied it. Wire a provider for it or mark the entry "
                        "optional (plain string / required: false)."
                    )
                logger.warning(
                    "REQUIRED external tool '%s' was declared in manifest but no "
                    "AdhocToolProvider supplied it — continuing because "
                    "strict=False; this environment is degraded",
                    name,
                )
            else:
                logger.warning(
                    "external tool '%s' was declared in manifest but no "
                    "AdhocToolProvider supplied it — skipping",
                    name,
                )
            continue
        if report is not None:
            if registry.get(name) is not None:
                # Last-write-wins: this external entry displaces an
                # earlier registration (typically a built-in being
                # intentionally hardened by the host).
                report.shadowed.append(name)
            report.resolved.append(name)
        registry.register(tool)


def _stage_kwargs_for_entry(entry: "StageManifestEntry") -> Dict[str, Any]:
    """Minimum kwargs required to instantiate *entry* via ``create_stage``.

    Stage 6 ('api') now reads its provider from ``entry.config["provider"]``
    (set via ``update_config`` post-construction). It no longer needs an
    ``api_key`` at construction time — credentials flow through
    ``state.llm_client`` resolved by ``Pipeline._resolve_llm_client``.
    """
    cfg = dict(entry.config or {})
    if entry.name == "api":
        provider = cfg.get("provider", "anthropic")
        return {"provider": str(provider)}
    return {}


class Pipeline:
    """Stage들을 순서대로 실행하는 파이프라인 엔진.

    Execution model (21-slot layout, S9a.3):
      Phase A: Input (Stage 1, once)
      Phase B: Agent Loop (Stage 2~16, repeats)
      Phase C: Finalize (Stage 17~21, once)

    Pipelines built via :meth:`from_manifest_async` also carry their
    associated :class:`~geny_executor.tools.mcp.manager.MCPManager` and
    :class:`~geny_executor.tools.registry.ToolRegistry` on
    ``pipeline.mcp_manager`` / ``pipeline.tool_registry`` so callers
    can reach either without re-plumbing.
    """

    # Loop boundary constants. Sub-phase 9a (S9a.3) extended the
    # 16-slot layout to 21 slots — the loop body now spans the
    # five new mid-pipeline stages (tool_review / task_registry /
    # hitl) and the finalize tail is five stages long instead of
    # three (emit / memory / summarize / persist / yield).
    LOOP_START = 2
    LOOP_END = 16  # inclusive
    FINALIZE_START = 17
    FINALIZE_END = 21  # inclusive
    EVENT_DATA_TRUNCATE = 500  # max chars for event data preview

    # Filled by ``from_manifest`` / ``from_manifest_async`` with the
    # outcome of built-in + external tool registration (audit §3.5).
    # Class-level default keeps hand-constructed ``Pipeline()``
    # instances attribute-safe (None = "no manifest registration ran").
    tool_resolution_report: Optional[ToolResolutionReport] = None

    # Default names for unregistered stage slots (used in bypass events)
    _DEFAULT_STAGE_NAMES: Dict[int, str] = {
        1: "input",
        2: "context",
        3: "system",
        4: "guard",
        5: "cache",
        6: "api",
        7: "token",
        8: "think",
        9: "parse",
        10: "tool",
        11: "tool_review",
        12: "agent",
        13: "task_registry",
        14: "evaluate",
        15: "hitl",
        16: "loop",
        17: "emit",
        18: "memory",
        19: "summarize",
        20: "persist",
        21: "yield",
    }

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        *,
        event_journal_size: int = 2048,
    ):
        self._config = config or PipelineConfig()
        self._stages: Dict[int, Stage] = {}
        self._event_bus = EventBus()
        # ── 2.2.0 unified event channel (audit 2026-06-09 §3.2) ──
        # Every engine event — bus-native (stage.enter/exit, pipeline
        # lifecycle) AND state-originated (text.delta, api.*, tool.*)
        # — funnels through _record_event, which stamps a monotonically
        # increasing .seq, journals the event in a bounded ring, and
        # fans it out to every events() tap. The journal is the replay
        # buffer for late-joining subscribers; cap it via the
        # ``event_journal_size`` constructor kwarg (events beyond the
        # cap are dropped oldest-first; replay then starts at the
        # oldest retained seq).
        if event_journal_size < 1:
            raise ValueError(f"event_journal_size must be >= 1 (got {event_journal_size})")
        self._event_seq: int = 0
        self._event_journal: Deque[PipelineEvent] = deque(maxlen=event_journal_size)
        self._event_taps: List[asyncio.Queue] = []
        self._mcp_manager: Any = None  # MCPManager | None — set by from_manifest_async
        self._tool_registry: Any = None  # ToolRegistry | None — set by from_manifest_async
        # CLI MCP passthrough (2.2.1): manifest mcp_servers translated to
        # the CLI's mcp-config shape, merged into the client's mcp_config
        # at _build_client_for time. Empty for SDK providers (host-side
        # MCPManager path) and for manifests without MCP servers.
        self._cli_mcp_passthrough: Dict[str, Any] = {}
        self._cli_mcp_passthrough_provider: str = ""
        # MemoryProvider built from the manifest's ``memory`` block
        # (2.2.0 Wave 3); None for hosts that wire memory themselves.
        self._memory_provider: Any = None
        self._tool_providers: List[
            Any
        ] = []  # started ToolProvider list — set by from_manifest_async
        self._has_started: bool = (
            False  # flips once run()/run_stream() begins; gates attach_runtime
        )
        self._attached_llm_client: Any = None  # set by attach_runtime; propagated in _init_state
        self._attached_sandbox: Any = None  # SandboxHandle; wraps a resolved claude_code_cli client in a container runner
        self._credentials: CredentialBundle = CredentialBundle()  # set by from_manifest_async
        self._subagent_registry: Any = None  # set by attach_runtime; populates state + agent stage
        self._attached_session_runtime: Any = None  # v0.30.0 plugin slot; propagated in _init_state
        # S9c.1 Pipeline.resume: token → asyncio.Future[HITLDecision].
        # The HITL stage's PipelineResumeRequester registers a future
        # here when it issues a request and awaits it. ``resume(token,
        # decision)`` resolves the future from outside the stage —
        # typically a websocket handler or HTTP endpoint receiving the
        # human's verdict.
        self._pending_hitl: Dict[str, Any] = {}
        # ── 2.2.0 lifecycle ownership (audit 2026-06-09 §3.1/§3.3) ──
        # Provider Stage 6 declared in the MANIFEST specifically — "" for
        # hand-built / builder / fixture pipelines. Gates the
        # attach_runtime(llm_client=) #866 guard: only a manifest
        # declaration is strong enough to refuse a conflicting client.
        self._manifest_provider: str = ""
        # Stages dropped by from_manifest(strict=False) — see DroppedStage.
        self.dropped_stages: List[DroppedStage] = []
        # Hook runner handle for pipeline-level lifecycle events
        # (pipeline start/end, stage enter/exit, loop iteration end).
        # None keeps every fire-site a single attribute check — near
        # zero overhead for the common no-hooks pipeline.
        self._hook_runner: Any = None
        # Client generation counter. invalidate_client() / a runtime
        # llm_client refresh bumps it; states stamp the generation they
        # captured a pipeline-resolved client under, and _init_state
        # re-resolves on mismatch. Closes the "rotation never lands on
        # a reused state" hole (_init_state used to fill only-if-None).
        self._client_generation: int = 0
        # Number of run()/run_stream() executions currently in flight
        # (counter, not bool — overlapping runs on one loop must not
        # unlock each other early). Exposed via .run_in_progress; the
        # mutator and refresh_runtime() consult it.
        self._runs_in_flight: int = 0
        # aclose() idempotency latch.
        self._closed: bool = False
        self._close_task: Optional[Any] = None  # keeps fire-and-forget close() task alive
        # state=None loudness — warn once per pipeline, not per turn.
        self._warned_state_none: bool = False
        # Deferred PIPELINE-scoped run-start events: (event_type, data)
        # pairs queued by attach/refresh (runtime.llm_client_override).
        # Flushed into state.add_event at the top of the NEXT run's
        # _run_phases — exactly once overall, by whichever run starts
        # next — because the attach/refresh that queued them is a
        # pipeline-level act, not a per-run one. By flush time
        # run_stream's bus subscription is attached, so streaming
        # consumers see them too (add_event from _init_state would land
        # pre-subscription). RUN-scoped events (config.override_applied
        # from per-run overrides) deliberately do NOT live here: they
        # ride ``state._pending_run_events`` so overlapping runs cannot
        # flush each other's overrides under the wrong run_id (2.2.0
        # review B2).
        self._pending_runtime_events: List[Tuple[str, Dict[str, Any]]] = []

    @property
    def mcp_manager(self) -> Any:
        """The :class:`MCPManager` this pipeline owns (if any).

        Set by :meth:`from_manifest_async` when the manifest declared any
        ``tools.mcp_servers``; ``None`` otherwise. Callers that need to
        dynamically add/remove servers at runtime reach for this.
        """
        return self._mcp_manager

    @property
    def tool_registry(self) -> Any:
        """The :class:`ToolRegistry` populated during async manifest load.

        Holds the MCP adapters discovered at session start. Returns
        ``None`` when the pipeline was built via the sync
        :meth:`from_manifest` path.
        """
        return self._tool_registry

    @property
    def memory_provider(self) -> Any:
        """The :class:`MemoryProvider` built from ``manifest.memory`` (if any).

        Set by :meth:`from_manifest` / :meth:`from_manifest_async` when
        the manifest's ``memory`` block was non-empty (2.2.0 Wave 3);
        ``None`` otherwise — including when the host wires memory
        runtime objects itself via :meth:`attach_runtime`. Exposed so
        hosts can reach the declared provider (e.g. to install
        :class:`MemoryHooks` or drive promotion) without re-building it.
        """
        return self._memory_provider

    @property
    def tool_providers(self) -> List[Any]:
        """Started :class:`ToolProvider` bundles owned by this pipeline.

        Populated by :meth:`from_manifest_async` when ``tool_providers=``
        was passed. Callers can introspect names / inspect roster, but
        lifecycle (startup/shutdown) is owned by the pipeline.
        """
        return list(self._tool_providers)

    async def shutdown_tool_providers(self) -> None:
        """Tear down any started :class:`ToolProvider` bundles.

        Hosts that long-live a pipeline across multiple sessions should
        call this during teardown. Safe to call when no providers were
        ever started.
        """
        from geny_executor.tools.provider import shutdown_providers

        if self._tool_providers:
            await shutdown_providers(self._tool_providers)
            self._tool_providers = []

    async def aclose(self) -> None:
        """Tear down every resource this pipeline owns. **Required host teardown.**

        Why (2.2.0, audit 2026-06-09 §2.4): the library never owned
        teardown — ``disconnect_all`` had exactly one caller (the
        build-failure unwind inside ``from_manifest_async``), so a host
        stopping a session with declared ``mcp_servers`` orphaned a
        stdio MCP child process every time (live leak in Geny prod).
        This method is the single aggregation point; hosts call it once
        when the session that owns the pipeline ends::

            pipeline = await Pipeline.from_manifest_async(manifest, ...)
            try:
                ...runs...
            finally:
                await pipeline.aclose()

        Tears down, in order:
          1. pending HITL futures — cancelled (``HITLDecision.CANCEL``)
             so any stage coroutine blocked on an approval unwinds
             instead of awaiting forever;
          2. live :meth:`events` taps — each subscriber queue receives
             a close sentinel so tap generators return instead of
             awaiting an event that will never come (2.2.0 — a host
             ``async for`` over a closed pipeline's tap would otherwise
             hang its consumer task forever);
          3. the owned :class:`MCPManager` — ``disconnect_all()``
             (reaps stdio child processes);
          4. started :class:`ToolProvider` bundles —
             :meth:`shutdown_tool_providers`.

        Best-effort and idempotent: each step's failure is logged and
        the remaining steps still run (a wedged MCP server must not
        leak the tool providers behind it); a second ``aclose()`` is a
        no-op. Safe on any pipeline regardless of how construction
        went — hand-built pipelines and partially-attached ones simply
        have nothing to tear down. The pipeline is not usable after
        closing — ``run()`` / ``run_stream()`` raise ``RuntimeError``
        (2.2.0 review N2; running with MCP disconnected silently
        degraded every tool call); build a fresh one per session.
        """
        if self._closed:
            return
        self._closed = True

        # 1. Unblock anything awaiting a human verdict. cancel_pending_hitl
        # is already tolerant of resolved/unknown tokens.
        for token in list(self._pending_hitl.keys()):
            try:
                self.cancel_pending_hitl(token)
            except Exception:  # noqa: BLE001 — teardown must not raise
                logger.warning("aclose: cancelling HITL token %r failed", token, exc_info=True)

        # 2. Wake every events() tap with the close sentinel; the tap
        # generators see it and return, detaching their queues.
        for tap_queue in list(self._event_taps):
            try:
                tap_queue.put_nowait(_TAP_CLOSED)
            except Exception:  # noqa: BLE001 — teardown must not raise
                logger.warning("aclose: closing an events() tap failed", exc_info=True)

        # 3. MCP servers (stdio child processes — the §2.4 leak).
        manager = self._mcp_manager
        if manager is not None and hasattr(manager, "disconnect_all"):
            try:
                await manager.disconnect_all()
            except Exception:  # noqa: BLE001
                logger.warning("aclose: MCPManager.disconnect_all failed", exc_info=True)

        # 4. Tool provider bundles.
        try:
            await self.shutdown_tool_providers()
        except Exception:  # noqa: BLE001
            logger.warning("aclose: shutdown_tool_providers failed", exc_info=True)

    def close(self) -> None:
        """Best-effort sync wrapper around :meth:`aclose`.

        For hosts whose teardown path is synchronous (atexit handlers,
        ``__del__``-adjacent cleanup). Outside a running event loop it
        blocks via ``asyncio.run``; inside one it schedules
        :meth:`aclose` as a task (kept referenced on the pipeline so
        the loop cannot garbage-collect it mid-flight) and returns
        immediately. Prefer ``await pipeline.aclose()`` whenever the
        caller is async — only that form guarantees teardown completed
        before the next statement.
        """
        if self._closed:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.aclose())
            return
        self._close_task = loop.create_task(self.aclose())

    # ── Construction from serialized state ──

    @classmethod
    def from_manifest(
        cls,
        manifest: "EnvironmentManifest",
        *,
        credentials: Optional[CredentialBundle] = None,
        api_key: Optional[str] = None,
        subagent_registry: Optional[Any] = None,
        subagent_env_resolver: Optional[Callable[[str], Any]] = None,
        strict: bool = True,
        adhoc_providers: Sequence["AdhocToolProvider"] = (),
        tool_registry: Optional["ToolRegistry"] = None,
    ) -> "Pipeline":
        """Construct a ready-to-run Pipeline from an :class:`EnvironmentManifest`.

        Steps:
          1. Validate the manifest — Stage 6 must declare a provider via
             ``config["provider"]``; ``strategies["provider"]`` is rejected
             (clean break from the legacy slot). Since 2.2.0 the full
             :func:`~geny_executor.core.environment.validate_manifest`
             catalogue check also runs here: in strict mode its
             error-severity findings (required stages inactive, unknown
             strategy/impl/artifact names on active stages,
             ``strategy_configs`` aimed at no-op ``configure``
             implementations) raise :class:`ConfigError` listing every
             issue, and warnings are logged; in lenient mode everything
             is logged and the build proceeds degraded.
          2. Build a :class:`PipelineConfig` from ``manifest.pipeline`` and
             ``manifest.model``.
          3. Instantiate each ``active`` stage via
             :func:`~geny_executor.core.artifact.create_stage`. Stage 6
             receives its ``provider`` string from the manifest entry's
             ``config["provider"]``.
          4. Run :meth:`PipelineMutator.restore` to apply per-stage configs,
             chain ordering, tool bindings, and model overrides.
          5. Store the ``credentials`` bundle on the pipeline. It is
             consulted by ``_resolve_llm_client`` when building the
             ``state.llm_client`` for Stage 6 (and by sub-pipelines for
             their own providers).
          6. Compile ``manifest.subagents`` into
             :class:`SubagentTypeDescriptor` registrations and wire
             Stage 12's orchestrator (2.2.0 Wave 3, audit §1-1). A
             host-supplied ``subagent_registry`` MERGES with the
             manifest entries — the explicit registry wins on
             ``agent_type`` collision (logged at info).
          7. When ``manifest.memory`` is non-empty, build the declared
             :class:`MemoryProvider` via
             :func:`~geny_executor.memory.factory.
             provider_from_manifest_memory` (the ``credentials`` bundle
             flows in so embedding keys stay single-channel) and wire
             it through the same slot wiring ``attach_runtime``'s
             memory kwargs use. Precedence: runtime objects beat
             declarations — a host that later calls
             ``attach_runtime(memory_retriever=... /
             memory_strategy=...)`` replaces the manifest-built wiring.

        Args:
            manifest: The environment template to materialize.
            credentials: Single-channel credential bundle. The required
                provider (Stage 6) must have an entry; otherwise
                ``ConfigError`` is raised at strict load.
            subagent_registry: Pre-built
                :class:`SubagentTypeRegistry` (Geny's path today).
                Merged with ``manifest.subagents`` — explicit
                registrations win on collision.
            subagent_env_resolver: Host callback resolving a
                ``subagents`` entry's ``env_id`` to a stored
                :class:`EnvironmentManifest` (or its dict form); sync
                or async. Only needed when the manifest declares
                ``env_id`` entries — without it those entries raise an
                actionable ``ConfigError`` at first dispatch.
            strict: Fail on stage instantiation / schema errors versus
                dropping the offending stage.
            adhoc_providers: Host-supplied
                :class:`~geny_executor.tools.providers.AdhocToolProvider`
                implementations.
            tool_registry: Existing registry to populate.

        Returns:
            A :class:`Pipeline` with every registered stage reflecting the
            manifest's template state.
        """
        from geny_executor.core.artifact import create_stage
        from geny_executor.core.environment import validate_manifest
        from geny_executor.core.mutation import PipelineMutator
        from geny_executor.tools.registry import ToolRegistry

        if strict:
            # Kept ahead of validate_manifest so the established, message-
            # pinned errors for the two provider-location contracts fire
            # first (hosts match on these strings).
            _validate_manifest_provider_locations(manifest)

        # Write-time manifest validation (2.2.0, audit §1-1 / §2.1).
        # Strict: error-severity issues refuse the build — required
        # stages must be active, strategy selections must resolve, and
        # strategy_configs aimed at no-op ``configure`` implementations
        # (the masked-degradation class that broke Geny prod's worker
        # loop) are rejected instead of silently dropped. Warnings log.
        # Lenient: everything logs (errors included, downgraded) — the
        # recovery path keeps loading, but no longer silently.
        issues = validate_manifest(manifest)
        validation_errors = [i for i in issues if i.severity == "error"]
        for issue in issues:
            if issue.severity == "error" and strict:
                continue  # raised in aggregate below — don't double-log
            logger.warning(
                "from_manifest%s: manifest issue [%s] %s",
                "" if strict else "(strict=False)",
                issue.code,
                issue.message,
            )
        if strict and validation_errors:
            details = "\n".join(f"  - [{i.code}] {i.message}" for i in validation_errors)
            raise ConfigError(
                f"Manifest failed strict validation with "
                f"{len(validation_errors)} error(s):\n{details}\n"
                "Fix the manifest (validate_manifest() reports the same "
                "issues at write time) or load with strict=False to build "
                "a degraded pipeline."
            )

        # Build the effective credential bundle. The canonical input is the
        # ``credentials`` kwarg. ``api_key`` is accepted as a test/legacy
        # convenience that auto-wraps a single Anthropic key. Pass both
        # and ``credentials`` wins.
        if credentials is None:
            if api_key:
                credentials = CredentialBundle(
                    by_provider={
                        "anthropic": ProviderCredentials(api_key=api_key),
                    }
                )
            else:
                credentials = CredentialBundle()

        pipeline_config = _pipeline_config_from_manifest(manifest)
        pipeline = cls(pipeline_config)
        pipeline._credentials = credentials

        registry = tool_registry if tool_registry is not None else ToolRegistry()

        entries = sorted(manifest.stage_entries(), key=lambda e: e.order)

        # Record the manifest's Stage 6 provider declaration. This is
        # deliberately captured from the MANIFEST entry, not from the
        # constructed APIStage — hand-built fixture pipelines derive a
        # ``_provider_name`` from whatever provider object they were
        # given (e.g. "mock"), and the attach_runtime(llm_client=)
        # guard must never fire for those. Only an explicit manifest
        # declaration carries enough intent to refuse a conflicting
        # runtime client (#866 guard, see attach_runtime).
        api_entry = next((e for e in entries if e.name == "api" and e.active), None)
        if api_entry is not None:
            pipeline._manifest_provider = str((api_entry.config or {}).get("provider") or "")

        for entry in entries:
            if not entry.active:
                continue
            kwargs = _stage_kwargs_for_entry(entry)
            try:
                stage = create_stage(entry.name, entry.artifact, **kwargs)
            except Exception as exc:
                if strict:
                    raise
                # Lenient mode keeps building (that is its recovery
                # contract) but the drop must be loud + inspectable —
                # a bare ``continue`` here let a degraded pipeline ship
                # with zero evidence (audit §1-1 masked degradation).
                logger.warning(
                    "from_manifest(strict=False): stage %r (order %d) failed to "
                    "construct and was DROPPED from the pipeline: %s: %s",
                    entry.name,
                    entry.order,
                    type(exc).__name__,
                    exc,
                )
                pipeline.dropped_stages.append(
                    DroppedStage(
                        name=entry.name,
                        order=entry.order,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            pipeline.register_stage(stage)

        restore_report = PipelineMutator(pipeline).restore(manifest.to_snapshot(), report=True)
        if strict and restore_report.has_skips:
            # Strict loads promised "the manifest is the source of
            # truth" — a silently skipped slot/impl/chain is exactly
            # the masked-degradation class the audit flagged (§2.1:
            # Geny prod ran with its evaluator config dropped on the
            # floor). Full promotion to a hard error is a separate
            # workstream's call; warn loudly with the specifics so
            # operators see the drift today.
            logger.warning(
                "from_manifest(strict): snapshot restore skipped manifest "
                "declarations — slots=%s impls=%s chains=%s errors=%s",
                restore_report.skipped_slots,
                restore_report.skipped_impls,
                restore_report.skipped_chains,
                restore_report.errors,
            )

        if strict:
            for stage in pipeline.stages:
                schema_fn = getattr(stage, "get_config_schema", None)
                if schema_fn is None:
                    continue
                schema = schema_fn()
                if schema is None:
                    continue
                stage_config = stage.get_config() if hasattr(stage, "get_config") else {}
                errors = schema.validate(stage_config) if hasattr(schema, "validate") else []
                if errors:
                    raise ValueError(
                        f"Stage {stage.name} (order {stage.order}) config invalid: "
                        f"{'; '.join(errors)}"
                    )

        # Built-ins register first so every pipeline has a working
        # default tool surface (Read/Write/Edit/Bash/Glob/Grep). An
        # external provider that declares the same name then shadows
        # the built-in — ``ToolRegistry.register`` is last-write-wins,
        # so host code can replace any framework tool with a hardened
        # variant by exposing an equally-named ``AdhocToolProvider``
        # entry and listing the name in ``manifest.tools.external``.
        # Both passes feed one ToolResolutionReport so the outcome of
        # registration is inspectable on the pipeline, not just logged
        # (audit §3.5); ``strict`` makes unresolved *required* external
        # entries a build failure instead of a runtime surprise.
        resolution_report = ToolResolutionReport()
        _register_built_in_tools(manifest, registry, report=resolution_report)
        _register_external_tools(
            manifest,
            registry,
            adhoc_providers,
            report=resolution_report,
            strict=strict,
        )
        pipeline.tool_resolution_report = resolution_report
        pipeline._tool_registry = registry

        # Stages that hold a tool-registry reference are instantiated at
        # line 287 *before* `_register_external_tools` populates the
        # shared registry. Unless we rebind post-hoc, those stages keep
        # their construction-time references (None for SystemStage; a
        # freshly-allocated empty `ToolRegistry()` for ToolStage) and
        # never see the populated tools at execute time.
        #
        # SystemStage (``_tool_registry``): builds ``state.tools`` from
        # the registry; a stale reference leaves ``state.tools`` empty
        # so the API stage sends ``tools=None`` to Anthropic.
        # ToolStage (``_registry``): the router looks up tool instances
        # here; a stale empty reference makes every tool call resolve
        # to ``unknown_tool`` even though the LLM was shown the schema.
        #
        # Both are rebound to the shared ``registry``. Callers that
        # wired their own registry explicitly are left alone: for
        # SystemStage, a non-None existing reference wins; for
        # ToolStage, if the stage already holds the same object as
        # ``registry`` (as happens when the caller passed it via the
        # outer ``tool_registry`` kwarg), no rebind is needed.
        for stage in pipeline._stages.values():
            if hasattr(stage, "_tool_registry") and getattr(stage, "_tool_registry", None) is None:
                stage._tool_registry = registry
            if getattr(stage, "name", None) == "tool" and hasattr(stage, "_registry"):
                if getattr(stage, "_registry") is not registry:
                    stage._registry = registry

        # ── Sub-agents: manifest section + explicit registry merge ──
        # (2.2.0 Wave 3, audit §1-1: sub-agent environments become
        # manifest-expressible.) Manifest entries compile into
        # library-backed descriptors; a host-supplied registry merges
        # on top with explicit registrations winning per agent_type —
        # runtime objects beat declarations, same precedence rule as
        # the memory block below.
        effective_subagent_registry = subagent_registry
        manifest_subagents = list(getattr(manifest, "subagents", []) or [])
        if manifest_subagents:
            from geny_executor.stages.s12_agent.subagent_type import (
                SubagentTypeRegistry,
                compile_subagent_descriptors,
            )

            compiled = compile_subagent_descriptors(
                manifest_subagents, env_resolver=subagent_env_resolver
            )
            if effective_subagent_registry is None:
                effective_subagent_registry = SubagentTypeRegistry()
            for descriptor in compiled:
                if descriptor.agent_type in effective_subagent_registry:
                    logger.info(
                        "from_manifest: subagents entry %r is also present in "
                        "the host-supplied subagent_registry — the explicit "
                        "registration wins; the manifest entry is skipped.",
                        descriptor.agent_type,
                    )
                    continue
                effective_subagent_registry.register(descriptor)
        if effective_subagent_registry is not None:
            pipeline._subagent_registry = effective_subagent_registry
            pipeline._wire_subagent_orchestrator(effective_subagent_registry)

        # ── Memory: manifest block → provider build + slot wiring ──
        # (2.2.0 Wave 3, audit §1-1.) Wiring goes through
        # _apply_runtime — the exact path attach_runtime's memory
        # kwargs take — so there is one slot-wiring implementation. A
        # host that attaches memory runtime objects afterwards
        # overwrites these slots: runtime objects beat declarations.
        manifest_memory = dict(getattr(manifest, "memory", {}) or {})
        if manifest_memory:
            try:
                from geny_executor.memory.factory import provider_from_manifest_memory
                from geny_executor.memory.retriever import MemoryAwareRetriever
                from geny_executor.memory.strategy import ProviderDrivenStrategy

                memory_provider = provider_from_manifest_memory(
                    manifest_memory, credentials=credentials
                )
            except Exception as exc:
                if strict:
                    raise ConfigError(
                        f"manifest.memory could not be built: "
                        f"{type(exc).__name__}: {exc}. Fix the block "
                        "(validate_manifest reports memory.* issues at "
                        "write time) or load with strict=False to build "
                        "without manifest memory."
                    ) from exc
                logger.warning(
                    "from_manifest(strict=False): manifest.memory failed to "
                    "build and was DROPPED — the pipeline runs without the "
                    "declared memory provider: %s: %s",
                    type(exc).__name__,
                    exc,
                )
            else:
                pipeline._memory_provider = memory_provider
                pipeline._apply_runtime(
                    memory_retriever=MemoryAwareRetriever(memory_provider),
                    memory_strategy=ProviderDrivenStrategy(memory_provider),
                )

        return pipeline

    @classmethod
    async def from_manifest_async(
        cls,
        manifest: "EnvironmentManifest",
        *,
        credentials: Optional[CredentialBundle] = None,
        api_key: Optional[str] = None,
        subagent_registry: Optional[Any] = None,
        subagent_env_resolver: Optional[Callable[[str], Any]] = None,
        strict: bool = True,
        adhoc_providers: Sequence["AdhocToolProvider"] = (),
        tool_registry: Optional["ToolRegistry"] = None,
        tool_providers: Optional[Sequence["ToolProvider"]] = None,
    ) -> "Pipeline":
        """Async sibling of :meth:`from_manifest` that also wires MCP.

        In addition to the stage assembly and external-provider
        registration :meth:`from_manifest` performs, this variant:

        1. Reads ``manifest.tools.mcp_servers`` and builds an
           :class:`MCPManager`.
        2. Calls ``manager.connect_all(...)`` — every server connects,
           initializes, and announces its tools *before* the pipeline
           returns. Any failure propagates as
           :class:`MCPConnectionError` and leaves no half-connected
           state behind.
        3. Registers each discovered adapter into ``tool_registry``
           (created fresh when the caller omits it) using the
           ``mcp__{server}__{tool}`` namespace set in PR2.
        4. Attaches both the manager and the registry to the returned
           pipeline so downstream callers can reach them via
           ``pipeline.mcp_manager`` / ``pipeline.tool_registry``.

        The ``adhoc_providers`` kwarg is forwarded to the inner
        :meth:`from_manifest` call, so ``manifest.tools.external`` names
        get registered into the same registry the MCP adapters land in —
        a single unified tool surface.

        Manifests with no MCP servers skip the connect pass entirely —
        ``pipeline.mcp_manager`` is an empty :class:`MCPManager` in that
        case and ``pipeline.tool_registry`` is the registry populated
        with whatever external providers claimed from
        ``manifest.tools.external``.

        Raises:
            MCPConnectionError: If any declared MCP server fails to
                connect, initialize, or announce its tools. No partial
                state is retained.
        """
        from geny_executor.tools.mcp.manager import MCPManager
        from geny_executor.tools.provider import register_providers
        from geny_executor.tools.registry import ToolRegistry

        registry = tool_registry if tool_registry is not None else ToolRegistry()

        pipeline = cls.from_manifest(
            manifest,
            credentials=credentials,
            api_key=api_key,
            subagent_registry=subagent_registry,
            subagent_env_resolver=subagent_env_resolver,
            strict=strict,
            adhoc_providers=adhoc_providers,
            tool_registry=registry,
        )

        # Register self-contained ToolProvider bundles before MCP so
        # manifest-declared + provider-shipped tools are present when
        # MCP adapters arrive. Name collisions at this layer are logged
        # by ``register_providers``; MCP registrations would still fail
        # cleanly via the registry's own dedupe if they conflicted.
        started_providers: List["ToolProvider"] = []
        if tool_providers:
            started_providers = await register_providers(list(tool_providers), registry)

        manager = MCPManager()

        configs = _mcp_configs_from_manifest(manifest)
        if configs:
            # ── CLI MCP passthrough (2.2.1) ───────────────────────────
            # Subprocess backends (claude_code_cli) run their own agentic
            # loop: Stage 10 never dispatches for them, so HOST-side MCP
            # connections are invisible to the actual LLM — the only MCP
            # channel that reaches it is the client's own --mcp-config.
            # Connecting host-side here therefore spawned the MCP child
            # for nothing while the user-attached server's tools never
            # surfaced in the conversation (the 2.2.1 incident). When the
            # manifest's Stage-6 provider is such a backend, hand the
            # servers to the client instead of the manager; the merge
            # into the client's mcp_config happens in _build_client_for
            # (host-supplied config wins on name collisions, e.g. Geny's
            # per-session bridge server).
            primary_provider = ""
            for entry in manifest.stage_entries():
                if entry.name == "api" and entry.active:
                    primary_provider = str((entry.config or {}).get("provider") or "")
                    break
            if _provider_wants_mcp_passthrough(primary_provider):
                cli_config, skipped = _mcp_servers_to_cli_config(configs)
                if skipped:
                    logger.warning(
                        "CLI MCP passthrough: server(s) %s have transports the "
                        "CLI mcp-config cannot express — they will not reach "
                        "the %s subprocess.",
                        sorted(skipped),
                        primary_provider,
                    )
                if cli_config.get("mcpServers"):
                    pipeline._cli_mcp_passthrough = cli_config
                    pipeline._cli_mcp_passthrough_provider = primary_provider
                    logger.info(
                        "CLI MCP passthrough: %d manifest MCP server(s) %s "
                        "routed to the %s subprocess via --mcp-config "
                        "(host-side connection skipped).",
                        len(cli_config["mcpServers"]),
                        sorted(cli_config["mcpServers"]),
                        primary_provider,
                    )
            else:
                try:
                    await manager.connect_all(configs)
                    adapters = await manager.discover_all()
                    for adapter in adapters:
                        registry.register(adapter)
                except BaseException:
                    # Unwind providers if MCP bring-up fails mid-flight so no
                    # half-started resources leak out.
                    from geny_executor.tools.provider import shutdown_providers

                    await shutdown_providers(started_providers)
                    await manager.disconnect_all()
                    raise

        pipeline._mcp_manager = manager
        pipeline._tool_registry = registry
        pipeline._tool_providers = started_providers
        return pipeline

    # ── Stage management ──

    def register_stage(self, stage: Stage) -> Pipeline:
        """Register or replace a stage. Supports chaining."""
        self._stages[stage.order] = stage
        return self

    def replace_stage(self, order: int, stage: Stage) -> Pipeline:
        """Replace stage at given order."""
        self._stages[order] = stage
        return self

    def remove_stage(self, order: int) -> Pipeline:
        """Remove stage (that slot will be bypassed)."""
        self._stages.pop(order, None)
        return self

    def get_stage(self, order: int) -> Optional[Stage]:
        """Get registered stage by order."""
        return self._stages.get(order)

    @property
    def stages(self) -> List[Stage]:
        """All registered stages, sorted by order."""
        return sorted(self._stages.values(), key=lambda s: s.order)

    # ── Runtime injection (for manifest-built pipelines) ──

    def attach_runtime(
        self,
        *,
        memory_retriever: Optional[Any] = None,
        memory_strategy: Optional[Any] = None,
        memory_persistence: Optional[Any] = None,
        system_builder: Optional[Any] = None,
        tool_context: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        session_runtime: Optional[Any] = None,
        hook_runner: Optional[Any] = None,
        mcp_manager: Optional[Any] = None,
        permission_rules: Optional[Any] = None,
        permission_mode: Optional[str] = None,
        subagent_registry: Optional[Any] = None,
        sandbox: Optional[Any] = None,
        override_manifest: bool = False,
    ) -> None:
        """Inject session-scoped runtime objects into a manifest-built pipeline.

        Manifests carry declarative stage layout (stage order, artifact name,
        strategy choices, configs). They intentionally cannot encode runtime
        objects like memory managers, LLM callbacks, or per-session paths
        (working directory, session id) — those are per-session and not
        serializable. After constructing a pipeline via
        :meth:`from_manifest_async`, hosts call this helper to plug those
        objects in before :meth:`run` / :meth:`run_stream`.

        For each kwarg that is not ``None`` this helper finds the relevant
        stage and replaces the corresponding slot's ``.strategy`` with the
        provided instance — except ``tool_context``, which overwrites the
        Tool stage's ``_context`` attribute (a :class:`ToolContext` carrier,
        not a pluggable strategy):

        - ``memory_retriever`` → Stage 2 (Context), slot ``retriever``.
        - ``memory_strategy`` → Stage 15 (Memory), slot ``strategy``.
        - ``memory_persistence`` → Stage 15 (Memory), slot ``persistence``.
        - ``system_builder`` → Stage 3 (System), slot ``builder``.
        - ``tool_context`` → Stage 10 (Tool), ``_context`` attribute.
        - ``session_runtime`` → ``state.session_runtime`` (free-shape
          plugin container; see dedicated arg docs below).

        If a target stage is absent (manifest excluded it) the kwarg for
        that stage is silently ignored — a pipeline without a Memory stage
        simply has nowhere to attach memory runtime.

        Where this sits among the configuration channels (per-run
        overrides > mutator/refresh > attach_runtime > manifest >
        defaults), see docs/architecture.md#configuration-precedence.

        Args:
            memory_retriever: A :class:`MemoryRetriever` subclass instance
                (e.g. :class:`GenyMemoryRetriever`). Host is responsible for
                constructing it with any ``llm_gate`` or
                ``curated_knowledge_manager`` callbacks it needs.
                Precedence (2.2.0 Wave 3): runtime objects beat
                declarations — attaching this (or ``memory_strategy``)
                replaces any wiring built from the manifest's
                ``memory`` block, which uses the same slot path.
            memory_strategy: A :class:`MemoryUpdateStrategy` subclass
                instance (e.g. :class:`GenyMemoryStrategy`). Host wires any
                ``llm_reflect`` callback at construction time.
            memory_persistence: A :class:`ConversationPersistence` subclass
                instance (e.g. :class:`GenyPersistence`).
            system_builder: A :class:`PromptBuilder` subclass instance
                (e.g. :class:`ComposablePromptBuilder` with
                :class:`PersonaBlock` + :class:`DateTimeBlock` +
                :class:`MemoryContextBlock`). Manifests can only serialize
                a static prompt string; host-composed multi-block builders
                with runtime behavior (date injection, memory weaving) must
                attach here.
            tool_context: A :class:`ToolContext` carrying session-scoped
                path and id info (``session_id``, ``working_dir``,
                ``storage_path``, ``env_vars``, ``allowed_paths``,
                ``metadata``). Note: ``session_id`` is still overwritten
                from the pipeline's per-run state inside Stage 10's
                ``execute`` — the attached context supplies the *host-level*
                fields that persist across runs.
            llm_client: A pre-built :class:`BaseClient` that becomes the
                Stage 6 client. **This is the most dangerous kwarg in
                the signature** and was undocumented until 2.2.0
                (audit 2026-06-09 §2.7): an attached client
                unconditionally beats the manifest's Stage 6
                ``config["provider"]`` in ``_resolve_llm_client``,
                which is how incident #866 shipped — a host attached
                an Anthropic client onto a manifest that declared
                ``claude_code_cli`` and every run silently used the
                wrong backend. Client precedence (highest wins):

                1. ``attach_runtime(llm_client=...)`` / a
                   ``refresh_runtime(llm_client=...)`` between turns;
                2. the manifest's ``stages[6].config["provider"]``
                   resolved via the :class:`CredentialBundle`;
                3. ``None`` (Stage 6 may still recover via its own
                   legacy/fixture provider, else raises at execute).

                Since 2.2.0 the #866 class is guarded: when this
                pipeline was built from a manifest that declares a
                Stage 6 provider, and the supplied client reports a
                different ``client.provider``, this method raises
                :class:`ConfigError` unless ``override_manifest=True``
                is passed alongside. Clients without a ``provider``
                attribute, same-provider clients, and pipelines with
                no manifest declaration (builder / fixture pipelines)
                are unaffected.
            override_manifest: Explicit acknowledgement that
                ``llm_client`` may contradict the manifest's declared
                Stage 6 provider. ``True`` allows the mismatch and
                emits a ``runtime.llm_client_override`` event (carrying
                both provider names) into the event stream at the next
                run start, so the override is visible in any transcript
                instead of being a silent foot-gun. Has no effect when
                no mismatch exists.
            session_runtime: Free-shape carrier for host-side
                session-scoped objects (e.g. game state, persona
                providers, emitter chains). The executor is intentionally
                ignorant of its type — when supplied, it is propagated
                into ``state.session_runtime`` at run start so any stage
                or plugin can reach it via ``state.session_runtime``.

                **Plugin compatibility guideline (non-binding):** plugins
                that share a pipeline should agree on attribute names
                and treat missing attributes as opt-out (use
                ``getattr(state.session_runtime, "creature_state", None)``
                rather than direct attribute access). The executor does
                not enforce a Protocol — schema collisions between
                competing plugins are a host-policy concern.

        Raises:
            RuntimeError: If the pipeline has already started a run. State
                from the prior run has already captured references to the
                pre-attach slot values; swapping them now would produce
                a mixed-runtime pipeline whose behavior is hard to reason
                about. Build a fresh pipeline and attach before running.

        Notes:
            Idempotent when called multiple times *before* the first run —
            the last call wins for each kwarg. After a run has started,
            this method is a hard error rather than a quiet no-op so hosts
            notice construction-order bugs immediately. For *between-turn*
            runtime updates on a long-lived pipeline (credential rotation,
            tool_context swap), use :meth:`refresh_runtime` — the same
            wiring without the before-first-run gate.
        """
        if self._has_started:
            raise RuntimeError(
                "Pipeline.attach_runtime() called after the pipeline has "
                "started running. Runtime objects must be attached before "
                "the first run() / run_stream() invocation; otherwise prior "
                "stage state has already captured references to the old "
                "values. Use refresh_runtime() for legal between-turn "
                "updates, or construct a fresh pipeline via "
                "from_manifest_async and attach before running."
            )
        self._apply_runtime(
            memory_retriever=memory_retriever,
            memory_strategy=memory_strategy,
            memory_persistence=memory_persistence,
            system_builder=system_builder,
            tool_context=tool_context,
            llm_client=llm_client,
            session_runtime=session_runtime,
            hook_runner=hook_runner,
            mcp_manager=mcp_manager,
            permission_rules=permission_rules,
            permission_mode=permission_mode,
            subagent_registry=subagent_registry,
            sandbox=sandbox,
            override_manifest=override_manifest,
        )

    def refresh_runtime(self, **attach_runtime_kwargs: Any) -> None:
        """Between-turn runtime update — :meth:`attach_runtime` without the gate.

        Why (2.2.0, audit 2026-06-09 §3.3 / host-compensation table):
        ``attach_runtime`` is construction-time-only by contract, so a
        host that needed to rotate credentials or swap a tool_context
        on a long-lived pipeline had no legal API — Geny shipped a
        ~220-line ``queue_runtime_refresh`` module that reached into
        private setters between turns. This method is that operation,
        owned by the library: identical kwargs, identical wiring
        (including the ``llm_client`` #866 guard and
        ``override_manifest`` escape hatch), legal at any *turn
        boundary*.

        A refreshed ``llm_client`` also bumps the pipeline's client
        generation, so reused states (the long-lived-state model)
        re-resolve their captured client at the next ``_init_state``
        instead of riding the stale one forever. See
        :meth:`invalidate_client` for the rotate-without-replacement
        variant.

        Raises:
            RuntimeError: If a run is currently in progress
                (``run_in_progress``). Swapping strategies mid-run
                would hand later stages different runtime objects than
                earlier stages already used — the exact mixed-runtime
                hazard the attach gate exists to prevent. Wait for the
                run to finish (or cancel it) first.
            TypeError: On kwargs :meth:`attach_runtime` does not accept.
        """
        if self.run_in_progress:
            raise RuntimeError(
                "Pipeline.refresh_runtime() called while a run is in "
                "progress. Runtime objects may only be swapped between "
                "turns; a mid-run swap would mix old and new runtime "
                "within one turn. Await the in-flight run() / drain the "
                "run_stream() iterator first."
            )
        self._apply_runtime(**attach_runtime_kwargs)

    def _apply_runtime(
        self,
        *,
        memory_retriever: Optional[Any] = None,
        memory_strategy: Optional[Any] = None,
        memory_persistence: Optional[Any] = None,
        system_builder: Optional[Any] = None,
        tool_context: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        session_runtime: Optional[Any] = None,
        hook_runner: Optional[Any] = None,
        mcp_manager: Optional[Any] = None,
        permission_rules: Optional[Any] = None,
        permission_mode: Optional[str] = None,
        subagent_registry: Optional[Any] = None,
        sandbox: Optional[Any] = None,
        override_manifest: bool = False,
    ) -> None:
        """Shared wiring behind :meth:`attach_runtime` / :meth:`refresh_runtime`.

        Gate-free by design — callers own the lifecycle policy
        (before-first-run for attach, between-turns for refresh); this
        method owns only the mechanics.
        """
        if memory_retriever is not None:
            self._set_stage_slot_strategy(
                stage_name="context", slot_name="retriever", strategy=memory_retriever
            )

        if memory_strategy is not None:
            self._set_stage_slot_strategy(
                stage_name="memory", slot_name="strategy", strategy=memory_strategy
            )

        if memory_persistence is not None:
            self._set_stage_slot_strategy(
                stage_name="memory", slot_name="persistence", strategy=memory_persistence
            )

        if system_builder is not None:
            self._set_stage_slot_strategy(
                stage_name="system", slot_name="builder", strategy=system_builder
            )

        if tool_context is not None:
            self._set_tool_stage_context(tool_context)

        if llm_client is not None:
            # #866 guard (2.2.0, audit §2.7): an attached client beats
            # the manifest unconditionally in _resolve_llm_client, so a
            # provider mismatch here means every subsequent run silently
            # uses a backend the environment never declared. Refuse
            # unless the caller explicitly acknowledged the override.
            client_provider = str(getattr(llm_client, "provider", "") or "")
            declared = self._manifest_provider
            if declared and client_provider and client_provider != declared:
                if not override_manifest:
                    raise ConfigError(
                        f"attach_runtime(llm_client=...): the supplied client "
                        f"reports provider {client_provider!r} but this "
                        f"pipeline's manifest declares Stage 6 provider "
                        f"{declared!r}. An attached client overrides the "
                        f"manifest unconditionally — this exact mismatch "
                        f"shipped incident #866 (runs silently used the "
                        f"wrong backend). Either drop the llm_client kwarg "
                        f"and let the credential bundle resolve "
                        f"{declared!r}, fix the manifest, or pass "
                        f"override_manifest=True to acknowledge the "
                        f"override (it will be announced via a "
                        f"'runtime.llm_client_override' event at the next "
                        f"run start)."
                    )
                # Acknowledged override — make it visible in the event
                # stream at the next run start instead of staying a
                # silent foot-gun.
                self._pending_runtime_events.append(
                    (
                        "runtime.llm_client_override",
                        {
                            "manifest_provider": declared,
                            "client_provider": client_provider,
                        },
                    )
                )
            self._attached_llm_client = llm_client
            # Bump the generation so reused states drop any previously
            # pipeline-resolved client and capture this one at the next
            # _init_state (credential-rotation symmetry, audit §3.3).
            self._client_generation += 1

        if sandbox is not None:
            # A sandbox handle (container_name + async ensure()). When the
            # pipeline resolves a ``claude_code_cli`` client from the
            # credential bundle, it wraps it in a ContainerCLIRunner so the
            # agent CLI spawns inside the sandbox container — see
            # ``_build_client_for``. Ignored for SDK providers (they never
            # spawn the CLI). Bump the generation so reused states rebuild
            # their client through the sandbox on the next turn.
            self._attached_sandbox = sandbox
            self._client_generation += 1
            # Also stamp it onto the Tool stage's context so the built-in
            # fs/shell tools run inside the container on the SDK-provider path
            # (the CLI path runs its own tools in-container already).
            self._set_tool_stage_sandbox(sandbox)

        if session_runtime is not None:
            self._attached_session_runtime = session_runtime

        if hook_runner is not None:
            # Stash on the Tool stage's context so Stage 10's
            # RegistryRouter sees it on every dispatch. The host
            # constructs a HookRunner once (per session typically) and
            # attaches it before the first run. The pipeline also keeps
            # its own handle for the lifecycle events it fires itself
            # (pipeline start/end, stage enter/exit, loop iteration end
            # — 2.2.0, previously documented-but-never-fired).
            self._hook_runner = hook_runner
            self._set_tool_stage_hook_runner(hook_runner)

        if permission_rules is not None or permission_mode is not None:
            # Phase 7 (S7.4): bind the permission matrix on the Tool
            # stage's context. Either / both kwargs are honoured —
            # passing only mode toggles the policy without changing
            # the rule set; passing only rules adopts the existing mode.
            self._set_tool_stage_permission_matrix(
                permission_rules=permission_rules,
                permission_mode=permission_mode,
            )

        if mcp_manager is not None:
            # Late-bind a host-managed :class:`MCPManager`. Replaces any
            # manager the manifest pass built (typical for hosts that
            # want to add / disable / enable servers at runtime). The
            # manager owns its servers — the pipeline just keeps a
            # handle for ``pipeline.mcp_manager`` lookup.
            self._mcp_manager = mcp_manager
            # Re-register the manager's currently-discovered tools into
            # the ``tool_registry`` so the next turn sees them. The
            # registry is the source of truth for the tool roster
            # (Stage 10 reads off it); reassigning the manager without
            # populating the registry would leave the new tools
            # invisible until the host added them itself.
            registry = self._tool_registry
            if registry is not None:
                self._reseed_registry_from_mcp(mcp_manager, registry)

        if subagent_registry is not None:
            # Hosts wire a SubagentTypeRegistry that Stage 12's
            # ``subagent_type`` orchestrator consumes. We store it on the
            # pipeline (propagated to ``state.subagent_registry`` in
            # ``_init_state``) and, when the agent stage is registered,
            # rebuild its orchestrator slot so the registry is bound.
            self._subagent_registry = subagent_registry
            self._wire_subagent_orchestrator(subagent_registry)

    def _wire_subagent_orchestrator(self, registry: Any) -> None:
        """Set the agent stage's orchestrator to a SubagentTypeOrchestrator
        bound to ``registry``. No-op when the pipeline has no agent stage."""
        agent_stage = next((s for s in self._stages.values() if s.name == "agent"), None)
        if agent_stage is None:
            return
        from geny_executor.stages.s12_agent.subagent_type import (
            SubagentTypeOrchestrator,
        )

        slots = agent_stage.get_strategy_slots()
        slot = slots.get("orchestrator")
        if slot is None:
            return
        slot.strategy = SubagentTypeOrchestrator(registry)

    def _reseed_registry_from_mcp(self, manager: Any, registry: Any) -> None:
        """Register a freshly attached MCP manager's tools into ``registry``.

        Called from :meth:`attach_runtime` when the host swaps in a
        live ``MCPManager``. Walks every CONNECTED server, discovers
        its tools, and registers each :class:`MCPToolAdapter` under the
        ``mcp__{server}__{tool}`` namespace. Already-registered names
        are kept in place — re-registering would clobber adhoc /
        built-in tools that share the manager-prefixed naming.

        Skipped silently when the manager has no live connections —
        the host can connect after attaching and the tools land via
        ``manager.add_server`` / ``manager.discover_tools`` later.
        """
        from geny_executor.tools.mcp.adapter import MCPToolAdapter

        # MCPManager keeps a private ``_servers`` dict — read in
        # read-only mode here so we don't depend on the public
        # ``discover_tools`` (which is async). Adapter construction is
        # cheap; deferring the discovery is purely about avoiding an
        # async-from-sync hop in attach_runtime's signature.
        servers = getattr(manager, "_servers", {})
        for conn in servers.values():
            if not getattr(conn, "is_connected", False):
                continue
            for defn in getattr(conn, "_tools", []):
                adapter = MCPToolAdapter(server=conn, definition=defn)
                if registry.get(adapter.name) is not None:
                    continue
                registry.register(adapter)

    def _set_tool_stage_permission_matrix(
        self,
        *,
        permission_rules: Optional[Any] = None,
        permission_mode: Optional[str] = None,
    ) -> None:
        """Attach permission rules / mode to the Tool stage's ``ToolContext``.

        ``permission_rules=None`` leaves the existing rule list intact;
        ``permission_mode=None`` likewise leaves the mode. Pass either
        to update independently. Silently a no-op when there is no Tool
        stage in the pipeline.
        """
        for stage in self._stages.values():
            if getattr(stage, "name", "") != "tool":
                continue
            ctx = getattr(stage, "_context", None)
            if ctx is None:
                from geny_executor.tools.base import ToolContext

                ctx = ToolContext()
                stage._context = ctx
            if permission_rules is not None:
                ctx.permission_rules = list(permission_rules)
            if permission_mode is not None:
                ctx.permission_mode = permission_mode
            return

    def _set_tool_stage_hook_runner(self, hook_runner: Any) -> None:
        """Attach ``hook_runner`` to the Tool stage's ``ToolContext``.

        Idempotent — calling twice replaces the runner. Silently a
        no-op if no Tool stage is registered (manifest excluded it).
        """
        for stage in self._stages.values():
            if getattr(stage, "name", "") != "tool":
                continue
            ctx = getattr(stage, "_context", None)
            if ctx is None:
                from geny_executor.tools.base import ToolContext

                ctx = ToolContext()
                stage._context = ctx
            ctx.hook_runner = hook_runner
            return

    def _set_tool_stage_sandbox(self, sandbox: Any) -> None:
        """Attach ``sandbox`` to the Tool stage's ``ToolContext`` so the
        built-in fs/shell tools run inside the container (SDK-path sandboxing).

        Idempotent; no-op if no Tool stage is registered.
        """
        for stage in self._stages.values():
            if getattr(stage, "name", "") != "tool":
                continue
            ctx = getattr(stage, "_context", None)
            if ctx is None:
                from geny_executor.tools.base import ToolContext

                ctx = ToolContext()
                stage._context = ctx
            ctx.sandbox = sandbox
            return

    def _set_stage_slot_strategy(self, *, stage_name: str, slot_name: str, strategy: Any) -> None:
        """Replace a named slot's strategy on the stage registered under *stage_name*.

        Silent no-op when the stage is absent — callers inspect the manifest
        to know whether a stage is present; attach_runtime should tolerate
        manifests that omit Context or Memory.
        """
        for stage in self._stages.values():
            if stage.name != stage_name:
                continue
            slots = stage.get_strategy_slots() if hasattr(stage, "get_strategy_slots") else {}
            slot = slots.get(slot_name)
            if slot is None:
                logger.debug(
                    "attach_runtime: stage '%s' has no slot '%s' (skipping)",
                    stage_name,
                    slot_name,
                )
                return
            slot.strategy = strategy
            return

    def _set_tool_stage_context(self, tool_context: Any) -> None:
        """Overwrite the Tool stage's ``_context`` attribute with the
        supplied :class:`ToolContext`.

        Unlike memory / system injections, ``ToolContext`` is not a
        strategy slot — it is a carrier of session-scoped path and id
        data used by Stage 10's ``execute`` to build per-call
        :class:`ToolContext` instances. Hosts supply it via
        ``attach_runtime`` because values like ``working_dir`` and
        ``storage_path`` depend on the session's on-disk scratch
        directory, which is allocated at session creation time and
        cannot live in a static manifest.

        Silent no-op when no Tool stage is registered.
        """
        for stage in self._stages.values():
            if stage.name == "tool":
                stage._context = tool_context
                return
        logger.debug("attach_runtime: no 'tool' stage registered (tool_context skipped)")

    # ── Execution ──

    def _ensure_not_closed(self) -> None:
        """Refuse to run on an :meth:`aclose`'d pipeline (review N2).

        After teardown the MCP servers are disconnected and the tool
        providers shut down, so a run would proceed silently degraded —
        tool calls failing one by one with no hint why. Fail fast with
        the actual remedy instead.
        """
        if self._closed:
            raise RuntimeError(
                "pipeline is closed — build a new one. aclose() has torn "
                "down this pipeline's runtime (MCP servers disconnected, "
                "tool providers shut down); running it would silently "
                "degrade every tool call."
            )

    @property
    def run_in_progress(self) -> bool:
        """True while any ``run()`` / ``run_stream()`` is executing phases.

        Covers the streaming background task's full lifetime, not just
        the generator's visible iteration. This is the engine-wired
        execution lock (2.2.0, audit §3.5): :meth:`refresh_runtime`
        refuses while it is set, and :class:`PipelineMutator` raises
        :class:`MutationLocked` from its config/strategy mutations —
        previously ``MutationLocked`` could not fire in prod because
        nothing ever called the manual ``lock_stage`` API.
        """
        return self._runs_in_flight > 0

    def invalidate_client(self) -> None:
        """Drop the resolved/attached LLM client; reused states re-resolve.

        Why (2.2.0, audit §3.3 sticky-client asymmetry): ``_init_state``
        fills ``state.llm_client`` only-if-None, so on a long-lived
        state a credential rotation or provider swap never landed — the
        first-ever resolved client rode the session forever. Calling
        this method:

        1. clears any ``attach_runtime(llm_client=...)`` client, and
        2. bumps the pipeline's client generation, so every state that
           captured a *pipeline-resolved* client re-resolves at its
           next ``_init_state`` (against the current credential bundle
           / any client supplied via :meth:`refresh_runtime` since).

        Clients a host placed **directly** onto ``state.llm_client``
        are never touched — the state never recorded a pipeline
        generation for them, and clobbering host property writes is not
        this method's mandate. Legal at any turn boundary; like
        :meth:`refresh_runtime` it must not race a run in progress.

        Raises:
            RuntimeError: If a run is in progress (mid-turn client swap
                would mix backends within one turn).
        """
        if self.run_in_progress:
            raise RuntimeError(
                "Pipeline.invalidate_client() called while a run is in "
                "progress. Invalidate between turns so a single turn never "
                "mixes two clients."
            )
        self._attached_llm_client = None
        self._client_generation += 1

    async def run(
        self,
        input: Any,
        state: Optional[PipelineState] = None,
        *,
        overrides: Optional[ModelOverrides] = None,
    ) -> PipelineResult:
        """Execute the full pipeline.

        Phase A: Stage 1 (Input) — runs once
        Phase B: Stage 2~16 (Agent Loop) — repeats until loop_decision != "continue"
        Phase C: Stage 17~21 (Finalize) — runs once

        Args:
            input: The user turn (string or rich input Stage 1 accepts).
            state: Conversation state. Pass the previous turn's state
                (or ``result.state``) to continue a conversation —
                omitting it starts a FRESH conversation, and doing so
                on a pipeline that has already run logs a one-time
                warning (audit §3.3: GAPT shipped prod amnesia exactly
                this way). Reused states get their per-turn fields
                reset via :meth:`PipelineState.begin_turn`.
            overrides: Optional :class:`ModelOverrides` applied AFTER
                config → state stomping, for THIS run only. The next
                run's ``apply_to_state`` reverts them by construction.

        Returns:
            :class:`PipelineResult` — its ``.state`` attribute carries
            the state actually used (the only handle when ``state`` was
            omitted).

        Raises:
            RuntimeError: If the pipeline has been :meth:`aclose`'d —
                its MCP servers / tool providers are gone; build a new
                pipeline per session.
        """
        self._ensure_not_closed()
        state = self._init_state(state, overrides=overrides)
        # Run-lock up BEFORE the first await (review N1): the
        # pipeline.start emit suspends, and a mutation landing in that
        # window would bypass MutationLocked while the run is morally
        # already in flight.
        self._runs_in_flight += 1
        success = False
        try:
            await self._emit(
                "pipeline.start",
                data={"input": str(input)[: self.EVENT_DATA_TRUNCATE]},
                session_id=state.session_id,
                run_id=state._run_id,
            )
            if self._hook_runner is not None:
                await self._fire_lifecycle_hook(
                    HookEvent.PIPELINE_START, state, details={"streaming": False}
                )

            await self._run_phases(input, state)

            result = PipelineResult.from_state(state)
            success = result.success
            await self._emit(
                "pipeline.complete",
                data={"iterations": state.iteration},
                session_id=state.session_id,
                run_id=state._run_id,
            )
            return result

        except Exception as e:
            await self._emit(
                "pipeline.error",
                data=_error_event_data(e),
                session_id=state.session_id,
                run_id=state._run_id,
            )
            return PipelineResult.error_result(str(e), state)
        finally:
            self._runs_in_flight -= 1
            self._end_turn(state)
            if self._hook_runner is not None:
                await self._fire_lifecycle_hook(
                    HookEvent.PIPELINE_END,
                    state,
                    details={"success": success, "iterations": state.iteration},
                )

    async def run_stream(
        self,
        input: Any,
        state: Optional[PipelineState] = None,
        *,
        overrides: Optional[ModelOverrides] = None,
    ) -> AsyncIterator[PipelineEvent]:
        """Streaming mode — yields PipelineEvents in real-time.

        Uses an asyncio.Queue so events emitted mid-stage (e.g. text.delta
        during streaming API calls) are yielded immediately, not buffered
        until stage completion.

        State ownership: unlike :meth:`run`, this generator yields
        events, not a result object — there is nothing to hang a
        ``.state`` attribute on, so **the caller must construct and
        hold the state themselves** to continue the conversation next
        turn::

            state = PipelineState(session_id=...)   # turn 1; reuse next turn
            async for event in pipeline.run_stream(user_text, state):
                ...

        Passing ``state=None`` on a pipeline that has already run logs
        the same one-time amnesia warning as :meth:`run` — the
        internally created state is unreachable afterwards (audit
        §3.3). ``overrides`` has :meth:`run`'s semantics: applied after
        config stomping, this run only.

        Channel unification (2.2.0, audit §3.2): this generator now
        subscribes ONCE, to the event bus. State events (text.delta,
        api.*) reach the bus through the bridge ``_init_state``
        installs, so the old second collector (and its
        double-delivery hazard) is gone; ``pipeline.start`` /
        ``pipeline.complete`` / ``pipeline.error`` are emitted through
        the bus as well, so ``pipeline.on()`` subscribers and
        :meth:`events` taps finally see the full lifecycle in
        streaming mode too. The collector filters on this run's
        ``run_id`` — overlapping runs on a shared pipeline no longer
        cross-pollinate each other's streams (host-emitted bus events
        without a run_id still pass, preserving the old behaviour).

        Raises:
            RuntimeError: On first iteration when the pipeline has been
                :meth:`aclose`'d — build a new pipeline per session.
        """
        self._ensure_not_closed()
        state = self._init_state(state, overrides=overrides)
        run_id = state._run_id
        queue: asyncio.Queue[PipelineEvent] = asyncio.Queue()
        _SENTINEL = object()

        # Single subscription: bus-native AND bridged state events.
        def bus_collector(event: PipelineEvent) -> None:
            if event.run_id and event.run_id != run_id:
                return  # another run's traffic on a shared pipeline
            queue.put_nowait(event)

        unsubscribe = self._event_bus.on("*", bus_collector)

        # Run-lock up BEFORE the first await (review N1): the
        # pipeline.start emit below suspends before the background task
        # exists, and a mutation in that window would bypass
        # MutationLocked. The counter is handed off to the task once it
        # is created — its finally owns the decrement from then on,
        # covering consumers that abandon the stream mid-run.
        self._runs_in_flight += 1
        counter_owned_by_task = False

        async def _run_pipeline() -> None:
            """Execute pipeline phases, then push sentinel to signal completion.

            The run-in-progress counter brackets the run from the
            generator's pre-emit increment through this task's whole
            lifetime (not just the generator's visible iteration) so
            refresh_runtime / mutator locking stay correct even when a
            consumer abandons the stream mid-run — the background task
            keeps executing until phases finish.
            """
            success = False
            try:
                if self._hook_runner is not None:
                    await self._fire_lifecycle_hook(
                        HookEvent.PIPELINE_START, state, details={"streaming": True}
                    )
                await self._run_phases(input, state)
                success = state.loop_decision != "error"

                await self._emit(
                    "pipeline.complete",
                    data={
                        # `result` is the canonical final text consumers
                        # forward to the user — it must not be truncated.
                        # EVENT_DATA_TRUNCATE only applies to preview-only
                        # event payloads (see pipeline.start.input).
                        "result": state.final_text,
                        "iterations": state.iteration,
                        "total_cost_usd": state.total_cost_usd,
                    },
                    session_id=state.session_id,
                    run_id=run_id,
                )
            except Exception as e:
                await self._emit(
                    "pipeline.error",
                    data={
                        **_error_event_data(e),
                        "total_cost_usd": state.total_cost_usd,
                    },
                    session_id=state.session_id,
                    run_id=run_id,
                )
            finally:
                self._runs_in_flight -= 1
                self._end_turn(state)
                if self._hook_runner is not None:
                    await self._fire_lifecycle_hook(
                        HookEvent.PIPELINE_END,
                        state,
                        details={"success": success, "iterations": state.iteration},
                    )
                queue.put_nowait(_SENTINEL)  # type: ignore[arg-type]

        try:
            # Through the bus (not a direct yield) so on() subscribers,
            # the journal and events() taps see the run open; the
            # collector above echoes it into our queue synchronously.
            await self._emit(
                "pipeline.start",
                data={"input": str(input)[: self.EVENT_DATA_TRUNCATE]},
                session_id=state.session_id,
                run_id=run_id,
            )

            # Run pipeline in background task so we can yield events as they arrive
            task = asyncio.create_task(_run_pipeline())
            counter_owned_by_task = True

            while True:
                event = await queue.get()
                if event is _SENTINEL:
                    break
                yield event

            await task  # propagate any unexpected errors

        except Exception as e:
            # Generator-machinery failure (queue handling, the awaited
            # task re-raising). The run task already announced its own
            # pipeline.error through the bus; this direct yield covers
            # failures outside it so the consumer is never left without
            # a terminal event. Recorded for the journal/taps too.
            error_event = PipelineEvent(
                type="pipeline.error",
                data=_error_event_data(e),
                session_id=state.session_id,
                run_id=run_id,
            )
            self._record_event(error_event)
            yield error_event

        finally:
            if not counter_owned_by_task:
                # The pre-emit increment was never handed to a task
                # (emit / task creation failed) — balance it here.
                self._runs_in_flight -= 1
            unsubscribe()

    # ── Events ──

    def on(self, event_type: str, handler: Callable) -> Callable:
        """Register event handler. Returns unsubscribe function.

        Since 2.2.0 the bus carries EVERYTHING the engine emits —
        state events (text.delta, api.*, tool.*) are bridged into it,
        so ``pipeline.on("*")`` is a complete feed, not just stage
        transitions (audit §3.2: subscribers previously never saw the
        state channel at all). Handler-based; for an async-iterator
        surface with replay and clean teardown, prefer :meth:`events`.
        """
        return self._event_bus.on(event_type, handler)

    @property
    def event_bus(self) -> EventBus:
        """Access the event bus directly."""
        return self._event_bus

    async def events(self, replay_from: int = -1) -> AsyncIterator[PipelineEvent]:
        """Multi-subscriber async-iterator tap over the unified event stream.

        Why (2.2.0, audit §3.2): with only ``run_stream`` (single
        consumer, run-scoped) and ``on()`` (callback, no replay), a
        host UI that attached mid-session had no way to catch up —
        Geny shipped a 50ms polling loop over ``state.events`` to
        compensate. This tap is the library-owned replacement::

            async for event in pipeline.events(replay_from=0):
                render(event)   # event.seq is the resume cursor

        Semantics:
          - Every engine event (bus-native + bridged state events,
            across ALL runs and sessions on this pipeline) flows
            through, stamped with ``seq`` / ``run_id`` /
            ``session_id``. Events a host emits directly on the raw
            :attr:`event_bus` are NOT journaled and do not appear here.
          - ``replay_from`` is a ``seq`` cursor: events with
            ``seq > replay_from`` still held in the bounded ring
            journal (``event_journal_size`` events, constructor kwarg)
            are yielded first, then the live feed continues with no
            gap and no duplicate. The default ``-1`` means live-only
            — strictly: replay is skipped entirely, which on a fresh
            pipeline is the same thing. Pass ``0`` to replay from the
            beginning (as far as the journal still reaches; older
            events have been evicted oldest-first).
          - Multiple concurrent taps each get the full sequence,
            independently paced (each tap has its own unbounded queue;
            an abandoned-but-unclosed tap therefore accumulates — close
            the generator or :meth:`aclose` the pipeline).
          - Termination: closing the generator (``aclose()`` /
            ``break`` + GC) detaches its queue immediately;
            :meth:`Pipeline.aclose` wakes every live tap with a close
            sentinel so host consumer tasks unwind instead of awaiting
            forever. No background tasks are spawned — nothing to leak.
        """
        if self._closed:
            return
        tap_queue: asyncio.Queue = asyncio.Queue()
        self._event_taps.append(tap_queue)
        last_seq = int(replay_from)
        try:
            if replay_from >= 0:
                # Snapshot AFTER registering the queue: anything that
                # arrives during replay is also in our queue, and the
                # seq cursor below deduplicates the overlap.
                for event in list(self._event_journal):
                    if event.seq > last_seq:
                        last_seq = event.seq
                        yield event
            while True:
                event = await tap_queue.get()
                if event is _TAP_CLOSED:
                    return
                if event.seq <= last_seq:
                    continue  # already delivered during journal replay
                last_seq = event.seq
                yield event
        finally:
            if tap_queue in self._event_taps:
                self._event_taps.remove(tap_queue)

    def _record_event(self, event: PipelineEvent) -> PipelineEvent:
        """Stamp ``seq``, journal, and fan out to :meth:`events` taps.

        The single funnel every engine event passes through before any
        subscriber sees it — keeping seq assignment here makes the
        ordering contract trivial: one pipeline, one monotonic
        sequence, no matter which channel produced the event.
        """
        self._event_seq += 1
        event.seq = self._event_seq
        self._event_journal.append(event)
        for tap_queue in list(self._event_taps):
            tap_queue.put_nowait(event)
        return event

    def _make_state_event_bridge(self, state: PipelineState, run_id: str) -> Callable:
        """Build the ``add_event`` → event-channel forwarder for one run.

        Installed on ``state._bus_emitter`` by :meth:`_init_state`.
        Wraps each event dict in a :class:`PipelineEvent` carrying the
        run's correlation ids, records it (seq / journal / taps) and
        dispatches synchronously on the bus — ``emit_sync`` preserves
        the inline-delivery semantics streaming consumers relied on
        when this was a per-state listener.
        """

        def _bridge(event_dict: Dict[str, Any]) -> None:
            event = PipelineEvent(
                type=event_dict["type"],
                stage=event_dict.get("stage", ""),
                iteration=event_dict.get("iteration", 0),
                timestamp=event_dict.get("timestamp", ""),
                data=event_dict.get("data", {}),
                session_id=state.session_id,
                run_id=run_id,
            )
            self._record_event(event)
            self._event_bus.emit_sync(event)

        return _bridge

    # ── Pipeline.resume API (S9c.1) ─────────────────────────────

    def list_pending_hitl(self) -> List[str]:
        """Tokens of unresolved HITL requests this pipeline is awaiting.

        Useful for UIs that want to show "X approvals waiting" or for
        admin endpoints that audit which sessions are paused.
        """
        return [t for t, fut in self._pending_hitl.items() if not fut.done()]

    def resume(self, token: str, decision: Any) -> None:
        """Resolve a pending HITL request by token.

        ``decision`` is normally an :class:`HITLDecision`; strings
        ``"approve"`` / ``"reject"`` / ``"cancel"`` are accepted and
        coerced. Resolves the asyncio.Future the HITL stage's
        :class:`PipelineResumeRequester` is awaiting on, which lets
        the pipeline continue from where it paused.

        Raises:
            KeyError: If the token is unknown.
            RuntimeError: If the token has already been resolved.
        """
        # Local import to avoid a runtime cycle with the HITL stage.
        from geny_executor.stages.s15_hitl.types import (
            HITLDecision,
            coerce_decision,
        )

        future = self._pending_hitl.get(token)
        if future is None:
            raise KeyError(f"unknown HITL token: {token!r}")
        if future.done():
            raise RuntimeError(f"HITL token already resolved: {token!r}")
        if isinstance(decision, HITLDecision):
            verdict = decision
        else:
            coerced = coerce_decision(decision)
            if coerced is None:
                raise ValueError(
                    f"unknown HITL decision: {decision!r} "
                    f"(expected approve/reject/cancel or HITLDecision)"
                )
            verdict = coerced
        future.set_result(verdict)

    def cancel_pending_hitl(self, token: str) -> bool:
        """Cancel a pending HITL request without supplying a decision.

        Equivalent to :meth:`resume` with :attr:`HITLDecision.CANCEL`
        but distinct in intent — used for "session terminated, drop
        any in-flight approvals" cleanup. Returns True when the token
        was unresolved (and is now cancelled), False when the token is
        unknown or already resolved.
        """
        from geny_executor.stages.s15_hitl.types import HITLDecision

        future = self._pending_hitl.get(token)
        if future is None or future.done():
            return False
        future.set_result(HITLDecision.CANCEL)
        return True

    # ── UI metadata ──

    def describe(self) -> List[StageDescription]:
        """Return pipeline structure for UI rendering.

        Iterates the canonical 21-slot layout (S9a.3). Slots without a
        registered stage emit an ``unregistered`` placeholder so UIs can
        still render the full row.
        """
        descriptions = []
        for order in range(1, self.FINALIZE_END + 1):
            stage = self._stages.get(order)
            if stage:
                desc = stage.describe()
                descriptions.append(desc)
            else:
                descriptions.append(
                    StageDescription(
                        name=self._DEFAULT_STAGE_NAMES.get(order, f"stage_{order}"),
                        order=order,
                        category="unregistered",
                        is_active=False,
                    )
                )
        return descriptions

    # ── Internal: Phase execution ──

    async def _run_phases(self, input: Any, state: PipelineState) -> None:
        """Execute all three pipeline phases (single source of truth).

        Sub-phase 9a (S9a.3) widened the layout from 16 → 21 stages.

        Phase A: Stage 1 (Input) — once
        Phase B: Stages 2~16 (Agent Loop) — repeats
        Phase C: Stages 17~21 (Finalize) — once

        ``_try_run_stage`` silently skips slots that have no stage
        registered, so presets that don't opt the new scaffolds in
        (orders 11/13/15/19/20) still run identically to pre-9a.
        """
        # Flush deferred run-start events now that any streaming
        # consumer is subscribed on the bus — _init_state queues them
        # because events added there would predate run_stream's
        # subscription and never stream. Two queues (review B2):
        #   * pipeline-scoped (runtime.llm_client_override from attach/
        #     refresh) — pipeline-global, delivered once overall, by
        #     whichever run starts next;
        #   * run-scoped (config.override_applied from THIS run's
        #     overrides) — on the state, so overlapping runs never
        #     flush each other's overrides under the wrong run_id.
        if self._pending_runtime_events:
            pending, self._pending_runtime_events = self._pending_runtime_events, []
            for event_type, data in pending:
                state.add_event(event_type, data)
        if state._pending_run_events:
            run_pending, state._pending_run_events = state._pending_run_events, []
            for event_type, data in run_pending:
                state.add_event(event_type, data)

        # Phase A: Input
        current = await self._run_stage(1, input, state)

        # Phase B: Agent Loop
        has_loop_stage = self.LOOP_END in self._stages
        while True:
            for order in range(self.LOOP_START, self.LOOP_END + 1):
                current = await self._try_run_stage(order, current, state)

            # If no Loop stage is registered, auto-complete after one pass
            if not has_loop_stage and state.loop_decision == "continue":
                state.loop_decision = "complete"

            # single_turn: complete after one pass regardless of loop decision
            if state.single_turn and state.loop_decision == "continue":
                state.loop_decision = "complete"

            # One full loop-body pass finished — fire the lifecycle
            # hook (2.2.0; previously documented-but-never-fired) with
            # the verdict the controller just produced.
            if self._hook_runner is not None:
                await self._fire_lifecycle_hook(
                    HookEvent.LOOP_ITERATION_END,
                    state,
                    details={
                        "iteration": state.iteration,
                        "loop_decision": state.loop_decision,
                    },
                )

            if state.loop_decision != "continue":
                break

            state.iteration += 1

            # Hard limits — checked at pipeline level, not delegated to stages
            if state.is_over_iterations:
                state.loop_decision = "complete"
                state.completion_signal = "MAX_ITERATIONS"
                state.add_event(
                    "loop.force_complete",
                    {"reason": "max_iterations", "iteration": state.iteration},
                )
                break
            if state.is_over_budget:
                state.loop_decision = "complete"
                state.completion_signal = "COST_BUDGET"
                state.add_event(
                    "loop.force_complete",
                    {
                        "reason": "cost_budget",
                        "total_cost_usd": state.total_cost_usd,
                        "budget_usd": state.cost_budget_usd,
                    },
                )
                break

        # Phase C: Finalize
        for order in range(self.FINALIZE_START, self.FINALIZE_END + 1):
            current = await self._try_run_stage(order, current, state)

    # ── Internal: Stage execution ──

    def _init_state(
        self,
        state: Optional[PipelineState],
        *,
        overrides: Optional[ModelOverrides] = None,
    ) -> PipelineState:
        """Initialize or apply config to state — the turn-boundary owner.

        Responsibilities (2.2.0, audit §3.3 — see the
        :class:`PipelineState` docstring for the field-lifetime
        contract this enforces):

        1. ``state=None`` loudness — warn once per pipeline when a
           pipeline that has already run gets no state (the GAPT
           amnesia class: history silently discarded).
        2. Reused-state detection (prior run on record, or pre-seeded
           ``messages`` from a checkpoint rehydration) →
           :meth:`PipelineState.begin_turn` resets the per-turn fields.
        3. Config → state stomping, then per-run ``overrides`` on top
           (one-run lifetime by construction — the next stomp reverts).
        4. Client resolution with generation stamping: fill-if-None as
           always, but also RE-resolve when the captured generation no
           longer matches (``invalidate_client`` / refreshed client
           since capture). Host-set clients (no generation on record)
           are never clobbered.
        5. Publish the resolved Stage 6 provider into
           ``state.shared[SharedKeys.PRIMARY_PROVIDER]`` so sub-agent
           factories can inherit the parent backend (audit §2.8: the
           read side existed for a release with no producer).
        6. Event correlation + channel bridge (2.2.0, audit §3.2):
           mint this run's ``run_id`` and install the
           ``state.add_event`` → event-bus forwarder. Re-installed
           every run so a state migrated between pipelines always
           feeds its CURRENT owner, and each turn's events carry that
           turn's run_id.
        """
        if state is None and self._has_started and not self._warned_state_none:
            self._warned_state_none = True
            logger.warning(
                "Pipeline.run/run_stream called with state=None on a pipeline "
                "that has run before: passing no state discards conversation "
                "history; pass the prior state (run() exposes it as "
                "result.state) or use a session store. This warning fires "
                "once per pipeline."
            )
        state = state or PipelineState()

        # Turn boundary: a state that already served a run — or arrives
        # pre-seeded with conversation history (checkpoint / host
        # rehydration) — must not leak the previous turn's loop verdict,
        # iteration count, or event log into this one.
        if state._run_count > 0 or state.messages:
            state.begin_turn()
        state._run_count += 1

        if not state.pipeline_id:
            state.pipeline_id = uuid.uuid4().hex[:12]

        # Per-run correlation id + the add_event → bus bridge. The
        # bridge closure carries the run_id so every event this state
        # produces during THIS turn is attributable; overlapping runs
        # each get their own state, hence their own bridge.
        state._run_id = uuid.uuid4().hex
        state._bus_emitter = self._make_state_event_bridge(state, state._run_id)

        self._config.apply_to_state(state)
        # Per-run override events queue ON THE STATE (review B2): the
        # pipeline-global queue is flushed by whichever run starts next,
        # so overlapping runs would misattribute these to the wrong
        # run_id. Reset first — a prior turn that failed before its
        # _run_phases flush must not leak its overrides into this one.
        state._pending_run_events = []
        if overrides is not None:
            # After apply_to_state on purpose: per-run overrides sit at
            # the top of the config funnel for exactly one run. Events
            # are deferred to _run_phases so streaming listeners see them.
            for field_name, value in overrides.apply_to_state(state).items():
                state._pending_run_events.append(
                    (
                        "config.override_applied",
                        {"field": field_name, "value": value, "source": "per_run"},
                    )
                )
        if state.credentials is None:
            state.credentials = self._credentials
        if state.subagent_registry is None and self._subagent_registry is not None:
            state.subagent_registry = self._subagent_registry
        if state.llm_client is None:
            state.llm_client = self._resolve_llm_client()
            state._client_generation = self._client_generation
        elif (
            state._client_generation is not None
            and state._client_generation != self._client_generation
        ):
            # The client this state captured came from a pipeline
            # resolution that has since been invalidated (credential
            # rotation / refresh_runtime(llm_client=...)). Re-resolve —
            # riding the stale client was the §3.3 asymmetry.
            state.llm_client = self._resolve_llm_client()
            state._client_generation = self._client_generation
        if state.session_runtime is None and self._attached_session_runtime is not None:
            state.session_runtime = self._attached_session_runtime

        provider_name = self._resolved_provider_name(state)
        if provider_name:
            state.shared[SharedKeys.PRIMARY_PROVIDER] = provider_name

        # Tool dispatch handle (2.3.0): Stage 6's tool_loop="internal"
        # strategy dispatches through Stage 10's exact machinery via
        # this state-carried handle. Rebuilt every run (two attribute
        # writes) so a PipelineMutator stage replacement is picked up
        # at the next turn boundary; None when no Tool stage exists —
        # the strategy then degrades to pipeline behaviour with a
        # one-time warning.
        tool_stage = next(
            (s for s in self._stages.values() if getattr(s, "name", "") == "tool"),
            None,
        )
        if tool_stage is not None and hasattr(tool_stage, "build_dispatch_context"):
            from geny_executor.stages.s10_tool.dispatcher import ToolDispatcher

            state.tool_dispatcher = ToolDispatcher(tool_stage)
        else:
            state.tool_dispatcher = None

        # Budget-recovery auto-wire (2.5.0): give the Guard stage the same
        # compactor the Context stage uses so a ``compact`` guard signal
        # (token-budget pressure) can shrink history and re-check instead
        # of hard-rejecting. Re-synced each turn so a host that swaps the
        # Context compactor (e.g. an LLM-backed one) is picked up — unless
        # the host wired recovery explicitly via attach_budget_recovery.
        guard_stage = self._stages.get(4)
        context_stage = self._stages.get(2)
        if (
            guard_stage is not None
            and context_stage is not None
            and hasattr(guard_stage, "attach_budget_recovery")
            and not getattr(guard_stage, "_budget_recovery_explicit", False)
        ):
            compactor = getattr(context_stage, "_compactor", None)
            provider = getattr(context_stage, "_provider", None)
            if compactor is not None:
                guard_stage._budget_compactor = compactor
                guard_stage._memory_provider = provider

        self._has_started = True
        return state

    def _end_turn(self, state: PipelineState) -> None:
        """Per-turn bookkeeping at run end (success OR failure paths).

        Folds the turn's cost accumulator into the session-cumulative
        counter. ``total_cost_usd`` itself is NOT zeroed here — the
        result/event consumers built right after the run still read it
        as "this turn's cost"; the reset happens at the NEXT turn's
        ``begin_turn``.
        """
        state.session_cost_usd += state.total_cost_usd

    def _resolved_provider_name(self, state: PipelineState) -> str:
        """Best available name for the provider Stage 6 will actually use.

        Preference order mirrors ``_resolve_llm_client``: the live
        client's own ``provider`` attribute (ground truth — covers
        attached/override clients), then the Stage 6 declaration
        (covers fixture pipelines whose client is resolved lazily
        inside the stage). Empty string when undeterminable.
        """
        client = state.llm_client
        if client is not None:
            name = str(getattr(client, "provider", "") or "")
            if name:
                return name
        api_stage = next((s for s in self._stages.values() if s.name == "api"), None)
        if api_stage is None:
            return ""
        provider_name = str(getattr(api_stage, "_provider_name", "") or "")
        if not provider_name and hasattr(api_stage, "get_config"):
            provider_name = str((api_stage.get_config() or {}).get("provider", "") or "")
        return provider_name

    async def _fire_lifecycle_hook(
        self,
        event: HookEvent,
        state: PipelineState,
        *,
        stage: Optional[Stage] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Fire a pipeline-level lifecycle hook; never let it break the run.

        2.2.0 (audit §3.5): pipeline start/end, stage enter/exit and
        loop-iteration-end were documented in :class:`HookEvent` but no
        engine path ever fired them — hosts bound dead handlers. The
        pipeline now mirrors its bus events to the hook runner. Call
        sites guard with ``if self._hook_runner is not None`` so the
        no-hooks fast path costs one attribute check; outcomes are
        observational here (lifecycle hooks cannot block the pipeline —
        blocking semantics remain a tool-invocation feature) and any
        runner failure is logged and swallowed.
        """
        runner = self._hook_runner
        if runner is None:
            return
        try:
            payload = HookEventPayload(
                event=event,
                session_id=state.session_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                pipeline_id=state.pipeline_id,
                stage_order=getattr(stage, "order", None) if stage is not None else None,
                stage_name=getattr(stage, "name", None) if stage is not None else None,
                details=details or {},
            )
            await runner.fire(event, payload)
        except Exception:  # noqa: BLE001 — observability must not kill the run
            logger.warning(
                "lifecycle hook %s failed (ignored)",
                getattr(event, "value", event),
                exc_info=True,
            )

    def _resolve_llm_client(self) -> Any:
        """Choose the LLM client to attach to fresh state.

        Preference order:
        1. An ``llm_client`` explicitly passed to :meth:`attach_runtime`.
        2. The Stage 6 ``config["provider"]`` resolved via
           :class:`ClientRegistry` + the host-supplied
           :class:`CredentialBundle`.
        3. ``None`` — pipelines built without a credential bundle (manual
           ``register_stage`` flow, or no api stage) simply report a
           ``None`` client. Stages that need a client surface that at
           execute time (Stage 6 raises an APIError).
        """
        if self._attached_llm_client is not None:
            return self._attached_llm_client
        api_stage = next((s for s in self._stages.values() if s.name == "api"), None)
        if api_stage is None:
            return None
        provider_name = getattr(api_stage, "_provider_name", "")
        if not provider_name:
            cfg = api_stage.get_config() if hasattr(api_stage, "get_config") else {}
            provider_name = str(cfg.get("provider", "") or "")
        if not provider_name:
            return None
        # If the host did not supply credentials for this provider, leave
        # the client as None. APIStage._resolve_client may still recover
        # via its legacy_client (test fixtures) before surfacing an error.
        if not self._credentials.has(provider_name):
            return None
        try:
            return self._build_client_for(provider_name)
        except ConfigError:
            return None

    def _build_client_for(self, provider: str) -> Any:
        """Build a fresh :class:`BaseClient` for *provider* using the
        bundle stored on this pipeline. Raises :class:`ConfigError` when
        either the provider is unknown or its credentials are missing."""
        if provider not in ClientRegistry.available():
            raise ConfigError(
                f"Unknown LLM provider {provider!r}. "
                f"Registered: {sorted(ClientRegistry.available())}"
            )
        creds = self._credentials.require(provider)
        client_cls = ClientRegistry.get(provider)
        kwargs = _creds_to_client_kwargs(provider, creds)
        # CLI MCP passthrough (2.2.1): hand manifest-declared MCP servers
        # to the subprocess client. Host-supplied mcp_config (e.g. Geny's
        # per-session bridge) wins on name collision. Manifest-declared
        # servers are also auto-allowed (``mcp__<server>``) — the
        # operator attached them in the environment editor, and the CLI's
        # --print mode has no human to answer a permission prompt, so
        # without the allow entry the passthrough would be dead on
        # arrival.
        if provider == self._cli_mcp_passthrough_provider and self._cli_mcp_passthrough.get(
            "mcpServers"
        ):
            kwargs["mcp_config"] = _merge_cli_mcp_config(
                kwargs.get("mcp_config"), self._cli_mcp_passthrough
            )
            allow = list(kwargs.get("allow_tools", ()) or ())
            for server_name in self._cli_mcp_passthrough["mcpServers"]:
                entry = f"mcp__{server_name}"
                if entry not in allow:
                    allow.append(entry)
            kwargs["allow_tools"] = tuple(allow)
        # Sandbox wrap: when a SandboxHandle is attached and this is the CLI
        # provider, build the client so every spawn (and the --version probe)
        # runs inside the sandbox container via ContainerCLIRunner. Reuses the
        # exact kwargs resolved above (api_key, mcp_config, allow_tools,
        # workspace_dir, ...) — the host never replicates them. SDK providers
        # ignore the sandbox (they don't spawn a CLI).
        if provider == "claude_code_cli" and self._attached_sandbox is not None:
            from geny_executor.llm_client.claude_code import (
                build_container_cli_client,
            )

            return build_container_cli_client(
                sandbox=self._attached_sandbox, **kwargs
            )
        return client_cls(**kwargs)

    async def _try_run_stage(self, order: int, current: Any, state: PipelineState) -> Any:
        """Run a stage if it exists and should not be bypassed."""
        stage = self._stages.get(order)
        if stage is None:
            # Emit bypass event so the UI shows unregistered stages as skipped
            name = self._DEFAULT_STAGE_NAMES.get(order, f"stage_{order}")
            await self._emit(
                "stage.bypass",
                stage=name,
                iteration=state.iteration,
                session_id=state.session_id,
                run_id=state._run_id,
            )
            return current
        if stage.should_bypass(state):
            await self._emit(
                "stage.bypass",
                stage=stage.name,
                iteration=state.iteration,
                session_id=state.session_id,
                run_id=state._run_id,
            )
            return current
        return await self._run_stage(order, current, state)

    async def _run_stage(self, order: int, input: Any, state: PipelineState) -> Any:
        """Execute a single stage with lifecycle hooks."""
        stage = self._stages.get(order)
        if stage is None:
            return input

        state.current_stage = stage.name
        state.stage_history.append(stage.name)
        await self._emit(
            "stage.enter",
            stage=stage.name,
            iteration=state.iteration,
            session_id=state.session_id,
            run_id=state._run_id,
        )
        if self._hook_runner is not None:
            # Mirror the bus event to the hook surface (2.2.0 — these
            # HookEvent kinds were reserved-but-never-fired before).
            await self._fire_lifecycle_hook(
                HookEvent.STAGE_ENTER, state, stage=stage, details={"iteration": state.iteration}
            )

        await stage.on_enter(state)
        try:
            result = await stage.execute(input, state)
            await stage.on_exit(result, state)
            await self._emit(
                "stage.exit",
                stage=stage.name,
                iteration=state.iteration,
                session_id=state.session_id,
                run_id=state._run_id,
            )
            if self._hook_runner is not None:
                await self._fire_lifecycle_hook(
                    HookEvent.STAGE_EXIT,
                    state,
                    stage=stage,
                    details={"iteration": state.iteration},
                )
            return result
        except Exception as e:
            await self._emit(
                "stage.error",
                stage=stage.name,
                iteration=state.iteration,
                data=_error_event_data(e),
                session_id=state.session_id,
                run_id=state._run_id,
            )
            recovery = await stage.on_error(e, state)
            if recovery is not None:
                return recovery
            raise StageError(str(e), stage_name=stage.name, stage_order=order, cause=e) from e

    async def _emit(self, event_type: str, **kwargs: Any) -> None:
        """Emit a pipeline event (bus-native channel).

        Records first (seq / journal / events() taps), then dispatches
        on the bus with async handlers awaited inline — the pre-2.2.0
        contract for bus subscribers is unchanged. ``kwargs`` map to
        :class:`PipelineEvent` fields; call sites pass ``session_id`` /
        ``run_id`` so correlation holds across overlapping runs.
        """
        event = PipelineEvent(type=event_type, **kwargs)
        self._record_event(event)
        await self._event_bus.emit(event)
