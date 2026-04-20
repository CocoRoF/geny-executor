"""Pipeline engine — executes stages in order with loop control."""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Dict, List, Optional

from geny_executor.core.config import PipelineConfig
from geny_executor.core.errors import StageError
from geny_executor.core.result import PipelineResult
from geny_executor.core.stage import Stage, StageDescription
from geny_executor.core.state import PipelineState
from geny_executor.events.bus import EventBus
from geny_executor.events.types import PipelineEvent

if TYPE_CHECKING:
    from geny_executor.core.environment import EnvironmentManifest, StageManifestEntry


# Stages whose default/openai/google artifacts require an ``api_key`` kwarg.
# Other artifacts (mock, etc.) construct without credentials.
_API_KEY_REQUIRING = {
    ("s06_api", "default"),
    ("s06_api", "openai"),
    ("s06_api", "google"),
}


def _pipeline_config_from_manifest(
    manifest: "EnvironmentManifest", *, api_key: Optional[str]
) -> PipelineConfig:
    """Build a :class:`PipelineConfig` from manifest pipeline+model blocks.

    The manifest stores ``pipeline`` and ``model`` as plain dicts; reunite
    them into the nested ``PipelineConfig(model=ModelConfig(...))`` shape the
    runtime expects. An explicit ``api_key`` kwarg wins over anything
    embedded in the manifest.
    """
    raw = dict(manifest.pipeline or {})
    if manifest.model:
        # ``pipeline.model`` (if present) loses to the top-level ``model``
        # block — the latter is the canonical location in v2 manifests.
        raw["model"] = dict(manifest.model)
    if api_key is not None:
        raw["api_key"] = api_key
    return PipelineConfig.from_dict(raw)


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


def _stage_kwargs_for_entry(entry: "StageManifestEntry", *, api_key: str) -> Dict[str, Any]:
    """Minimum kwargs required to instantiate *entry* via ``create_stage``.

    Most stages take no constructor args; API artifacts need ``api_key``
    when the manifest did not wire in a provider directly. Short-name
    stage identifiers ("api") are resolved to their module name before the
    lookup, so manifests written either way work uniformly.
    """
    from geny_executor.core.artifact import _resolve_stage_module

    try:
        module_name = _resolve_stage_module(entry.name)
    except ValueError:
        module_name = entry.name
    key = (module_name, entry.artifact)
    if key in _API_KEY_REQUIRING and api_key:
        return {"api_key": api_key}
    return {}


class Pipeline:
    """Stage들을 순서대로 실행하는 파이프라인 엔진.

    Execution model:
      Phase A: Input (Stage 1, once)
      Phase B: Agent Loop (Stage 2~13, repeats)
      Phase C: Finalize (Stage 14~16, once)

    Pipelines built via :meth:`from_manifest_async` also carry their
    associated :class:`~geny_executor.tools.mcp.manager.MCPManager` and
    :class:`~geny_executor.tools.registry.ToolRegistry` on
    ``pipeline.mcp_manager`` / ``pipeline.tool_registry`` so callers
    can reach either without re-plumbing.
    """

    # Loop boundary constants
    LOOP_START = 2
    LOOP_END = 13  # inclusive
    FINALIZE_START = 14
    FINALIZE_END = 16  # inclusive
    EVENT_DATA_TRUNCATE = 500  # max chars for event data preview

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
        11: "agent",
        12: "evaluate",
        13: "loop",
        14: "emit",
        15: "memory",
        16: "yield",
    }

    def __init__(self, config: Optional[PipelineConfig] = None):
        self._config = config or PipelineConfig()
        self._stages: Dict[int, Stage] = {}
        self._event_bus = EventBus()
        self._mcp_manager: Any = None  # MCPManager | None — set by from_manifest_async
        self._tool_registry: Any = None  # ToolRegistry | None — set by from_manifest_async

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

    # ── Construction from serialized state ──

    @classmethod
    def from_manifest(
        cls,
        manifest: "EnvironmentManifest",
        *,
        api_key: Optional[str] = None,
        strict: bool = True,
    ) -> "Pipeline":
        """Construct a ready-to-run Pipeline from an :class:`EnvironmentManifest`.

        Steps:
          1. Build a :class:`PipelineConfig` from ``manifest.pipeline`` and
             ``manifest.model``. A caller-supplied ``api_key`` (kwarg) wins
             over whatever is inside the manifest — manifests are templates
             and credentials usually live outside them.
          2. Instantiate each ``active`` stage via
             :func:`~geny_executor.core.artifact.create_stage` with the
             recorded ``artifact`` name. Stages whose artifact requires an
             ``api_key`` (e.g. ``s06_api/default``) receive it here.
          3. Run :meth:`PipelineMutator.restore` over
             ``manifest.to_snapshot()`` to apply strategies, strategy
             configs, stage configs, chain ordering, tool bindings, and
             model overrides.
          4. When ``strict`` is true, every stage's config is validated
             against its ``ConfigSchema`` and instantiation failures
             propagate. When false, broken stages are silently skipped so
             a partial environment still yields a runnable pipeline.

        Args:
            manifest: The environment template to materialize.
            api_key: Credential injected into API-backed stage constructors.
                When omitted, the value (if any) embedded in
                ``manifest.pipeline`` is used instead.
            strict: Fail on stage instantiation / schema errors versus
                dropping the offending stage.

        Returns:
            A :class:`Pipeline` with every registered stage reflecting the
            manifest's template state. The returned pipeline is ready for
            ``.run()`` once a tool registry and runtime state are attached.
        """
        from geny_executor.core.artifact import create_stage
        from geny_executor.core.mutation import PipelineMutator

        pipeline_config = _pipeline_config_from_manifest(manifest, api_key=api_key)
        pipeline = cls(pipeline_config)

        entries = sorted(manifest.stage_entries(), key=lambda e: e.order)
        effective_key = api_key if api_key is not None else pipeline_config.api_key

        for entry in entries:
            if not entry.active:
                continue
            kwargs = _stage_kwargs_for_entry(entry, api_key=effective_key)
            try:
                stage = create_stage(entry.name, entry.artifact, **kwargs)
            except Exception:
                if strict:
                    raise
                continue
            pipeline.register_stage(stage)

        PipelineMutator(pipeline).restore(manifest.to_snapshot())

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

        return pipeline

    @classmethod
    async def from_manifest_async(
        cls,
        manifest: "EnvironmentManifest",
        *,
        api_key: Optional[str] = None,
        strict: bool = True,
        tool_registry: Optional[Any] = None,
    ) -> "Pipeline":
        """Async sibling of :meth:`from_manifest` that also wires MCP.

        In addition to the stage assembly :meth:`from_manifest` performs,
        this variant:

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

        Manifests with no MCP servers skip the connect pass entirely —
        ``pipeline.mcp_manager`` is an empty :class:`MCPManager` in that
        case and ``pipeline.tool_registry`` is whatever the caller
        passed in (or a fresh empty registry).

        Raises:
            MCPConnectionError: If any declared MCP server fails to
                connect, initialize, or announce its tools. No partial
                state is retained.
        """
        from geny_executor.tools.mcp.manager import MCPManager, MCPServerConfig
        from geny_executor.tools.registry import ToolRegistry

        pipeline = cls.from_manifest(manifest, api_key=api_key, strict=strict)

        registry = tool_registry if tool_registry is not None else ToolRegistry()
        manager = MCPManager()

        configs = _mcp_configs_from_manifest(manifest)
        if configs:
            try:
                await manager.connect_all(configs)
                adapters = await manager.discover_all()
                for adapter in adapters:
                    registry.register(adapter)
            except BaseException:
                await manager.disconnect_all()
                raise

        pipeline._mcp_manager = manager
        pipeline._tool_registry = registry
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

    # ── Execution ──

    async def run(self, input: Any, state: Optional[PipelineState] = None) -> PipelineResult:
        """Execute the full pipeline.

        Phase A: Stage 1 (Input) — runs once
        Phase B: Stage 2~13 (Agent Loop) — repeats until loop_decision != "continue"
        Phase C: Stage 14~16 (Finalize) — runs once
        """
        state = self._init_state(state)
        await self._emit("pipeline.start", data={"input": str(input)[: self.EVENT_DATA_TRUNCATE]})

        try:
            await self._run_phases(input, state)

            result = PipelineResult.from_state(state)
            await self._emit("pipeline.complete", data={"iterations": state.iteration})
            return result

        except Exception as e:
            await self._emit("pipeline.error", data={"error": str(e)})
            return PipelineResult.error_result(str(e), state)

    async def run_stream(
        self, input: Any, state: Optional[PipelineState] = None
    ) -> AsyncIterator[PipelineEvent]:
        """Streaming mode — yields PipelineEvents in real-time.

        Uses an asyncio.Queue so events emitted mid-stage (e.g. text.delta
        during streaming API calls) are yielded immediately, not buffered
        until stage completion.
        """
        state = self._init_state(state)
        queue: asyncio.Queue[PipelineEvent] = asyncio.Queue()
        _SENTINEL = object()

        # Capture EventBus events (stage.enter/exit/bypass etc.)
        def bus_collector(event: PipelineEvent) -> None:
            queue.put_nowait(event)

        # Capture state.add_event() calls (text.delta, api.request etc.)
        def state_collector(event_dict: Dict[str, Any]) -> None:
            queue.put_nowait(
                PipelineEvent(
                    type=event_dict["type"],
                    stage=event_dict.get("stage", ""),
                    iteration=event_dict.get("iteration", 0),
                    timestamp=event_dict.get("timestamp", ""),
                    data=event_dict.get("data", {}),
                )
            )

        unsubscribe = self._event_bus.on("*", bus_collector)
        state._event_listener = state_collector

        async def _run_pipeline() -> None:
            """Execute pipeline phases, then push sentinel to signal completion."""
            try:
                await self._run_phases(input, state)

                queue.put_nowait(
                    PipelineEvent(
                        type="pipeline.complete",
                        data={
                            # `result` is the canonical final text consumers
                            # forward to the user — it must not be truncated.
                            # EVENT_DATA_TRUNCATE only applies to preview-only
                            # event payloads (see pipeline.start.input).
                            "result": state.final_text,
                            "iterations": state.iteration,
                            "total_cost_usd": state.total_cost_usd,
                        },
                    )
                )
            except Exception as e:
                queue.put_nowait(
                    PipelineEvent(
                        type="pipeline.error",
                        data={
                            "error": str(e),
                            "total_cost_usd": state.total_cost_usd,
                        },
                    )
                )
            finally:
                queue.put_nowait(_SENTINEL)  # type: ignore[arg-type]

        try:
            yield PipelineEvent(
                type="pipeline.start", data={"input": str(input)[: self.EVENT_DATA_TRUNCATE]}
            )

            # Run pipeline in background task so we can yield events as they arrive
            task = asyncio.create_task(_run_pipeline())

            while True:
                event = await queue.get()
                if event is _SENTINEL:
                    break
                yield event

            await task  # propagate any unexpected errors

        except Exception as e:
            yield PipelineEvent(type="pipeline.error", data={"error": str(e)})

        finally:
            state._event_listener = None
            unsubscribe()

    # ── Events ──

    def on(self, event_type: str, handler: Callable) -> Callable:
        """Register event handler. Returns unsubscribe function."""
        return self._event_bus.on(event_type, handler)

    @property
    def event_bus(self) -> EventBus:
        """Access the event bus directly."""
        return self._event_bus

    # ── UI metadata ──

    def describe(self) -> List[StageDescription]:
        """Return pipeline structure for UI rendering."""
        descriptions = []
        for order in range(1, 17):
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

        Phase A: Stage 1 (Input) — once
        Phase B: Stages 2~13 (Agent Loop) — repeats
        Phase C: Stages 14~16 (Finalize) — once
        """
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

    def _init_state(self, state: Optional[PipelineState]) -> PipelineState:
        """Initialize or apply config to state."""
        state = state or PipelineState()
        if not state.pipeline_id:
            state.pipeline_id = uuid.uuid4().hex[:12]
        self._config.apply_to_state(state)
        return state

    async def _try_run_stage(self, order: int, current: Any, state: PipelineState) -> Any:
        """Run a stage if it exists and should not be bypassed."""
        stage = self._stages.get(order)
        if stage is None:
            # Emit bypass event so the UI shows unregistered stages as skipped
            name = self._DEFAULT_STAGE_NAMES.get(order, f"stage_{order}")
            await self._emit("stage.bypass", stage=name, iteration=state.iteration)
            return current
        if stage.should_bypass(state):
            await self._emit("stage.bypass", stage=stage.name, iteration=state.iteration)
            return current
        return await self._run_stage(order, current, state)

    async def _run_stage(self, order: int, input: Any, state: PipelineState) -> Any:
        """Execute a single stage with lifecycle hooks."""
        stage = self._stages.get(order)
        if stage is None:
            return input

        state.current_stage = stage.name
        state.stage_history.append(stage.name)
        await self._emit("stage.enter", stage=stage.name, iteration=state.iteration)

        await stage.on_enter(state)
        try:
            result = await stage.execute(input, state)
            await stage.on_exit(result, state)
            await self._emit("stage.exit", stage=stage.name, iteration=state.iteration)
            return result
        except Exception as e:
            await self._emit(
                "stage.error",
                stage=stage.name,
                iteration=state.iteration,
                data={"error": str(e)},
            )
            recovery = await stage.on_error(e, state)
            if recovery is not None:
                return recovery
            raise StageError(str(e), stage_name=stage.name, stage_order=order, cause=e) from e

    async def _emit(self, event_type: str, **kwargs: Any) -> None:
        """Emit a pipeline event."""
        event = PipelineEvent(type=event_type, **kwargs)
        await self._event_bus.emit(event)
