"""Subagent-type registry + orchestrator.

After Phase D1 of the LLM backend upgrade, a sub-agent factory is no
longer zero-arg — it receives a :class:`SubAgentBuildContext` carrying
the parent's :class:`CredentialBundle`, descriptor, session ids, and
workspace snapshot. This is what makes **multi-provider sub-agents**
possible: a factory reads ``ctx.descriptor.provider`` and builds its
sub-pipeline manifest with the desired Stage 6 provider, then runs
``Pipeline.from_manifest`` with the shared bundle.

This module ships:

* :class:`SubagentTypeDescriptor` — frozen metadata + factory dataclass.
  Carries ``provider`` / ``provider_credentials_extras`` / ``parallel``
  / ``max_concurrent`` on top of the legacy fields.
* :class:`SubAgentBuildContext` — frozen build-time context passed to
  every factory.
* :class:`SubagentTypeRegistry` — id→descriptor map mirroring
  :class:`~geny_executor.tools.registry.ToolRegistry` (register /
  unregister / get / list).
* :class:`SubagentTypeOrchestrator` — :class:`AgentOrchestrator`
  subclass that consumes ``state.delegate_requests`` against the
  registry. Serial dispatch in D1; parallel fan-out arrives in D2.
  2.2.0 Wave 3 adds the single-call ``run_subagent`` surface —
  the call shape ``AgentTool`` / ``LocalAgentExecutor`` dispatch on.
* Manifest compilation (2.2.0 Wave 3, audit §1-1):
  :func:`compile_subagent_descriptors` turns a manifest ``subagents``
  section into descriptors backed by
  :class:`ManifestSubagentPipelineFactory`, and
  :func:`resolve_subagent_provider` is the single home for the
  sub-agent provider resolution order (entry override → parent
  inheritance → credential-bundle preference).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
from dataclasses import dataclass, field, replace
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from geny_executor.core.shared_keys import SharedKeys
from geny_executor.core.state import PipelineState
from geny_executor.llm_client.credentials import ConfigError
from geny_executor.stages.s12_agent.interface import AgentOrchestrator
from geny_executor.stages.s12_agent.types import AgentResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubAgentBuildContext:
    """Build-time context handed to every :data:`PipelineFactory`.

    The orchestrator builds one of these per dispatch and forwards it
    to the factory. The factory uses ``descriptor.provider`` (etc.) to
    shape its sub-pipeline manifest, and ``credentials`` to pass the
    parent's :class:`CredentialBundle` straight to
    ``Pipeline.from_manifest`` so authentication is single-channel
    end-to-end.
    """

    parent_session_id: str
    sub_session_id: str
    credentials: Any  # CredentialBundle | None — typed loosely to avoid import cycles
    descriptor: "SubagentTypeDescriptor"
    workspace_snapshot: Optional[Mapping[str, Any]] = None
    parent_state_shared: Mapping[str, Any] = field(default_factory=dict)
    #: Resolved Stage 6 provider of the PARENT pipeline — the typed
    #: short-term fix from audit 2026-06-09 (host_ergonomics #7): the
    #: orchestrator populates it from ``state.shared[SharedKeys.
    #: PRIMARY_PROVIDER]`` (the producer Wave 2 made real in
    #: ``Pipeline._init_state``) so factories no longer have to dig the
    #: bare string out of ``parent_state_shared``. ``None`` when the
    #: parent never resolved a provider (hand-built fixture pipelines).
    parent_provider: Optional[str] = None


# A factory takes a build context and returns a Pipeline (sync) or an
# Awaitable[Pipeline] (async). Hosts that do async setup (MCP, storage)
# write an async factory.
PipelineFactory = Callable[[SubAgentBuildContext], Union[Any, Awaitable[Any]]]


@dataclass(frozen=True)
class SubagentTypeDescriptor:
    """Static metadata describing one sub-agent type.

    Attributes:
        agent_type: Stable identifier — registry key + the value the
            LLM sees in ``[DELEGATE: <agent_type>]`` markers + the
            field used in ``state.delegate_requests`` entries.
        factory: Callable receiving a :class:`SubAgentBuildContext` and
            returning a ready-to-run :class:`Pipeline`. May be sync or
            async.
        description: One-line summary the LLM uses when choosing
            whether to delegate. Mirrors ``Tool.description``.
        allowed_tools: Tuple of tool names the sub-agent's pipeline
            should expose. Empty tuple means "inherit parent" — the
            host is responsible for applying this in the factory; the
            registry just records intent.
        provider: Override the sub-pipeline's Stage 6 provider
            (e.g. ``"openai"``, ``"claude_code_cli"``). ``None`` means
            "inherit parent" (factory may copy parent provider).
        provider_credentials_extras: Free-form bag merged into the
            parent's :class:`ProviderCredentials.extras` for *this*
            sub-agent when the factory chooses to. Common use: bumping
            ``max_budget_usd`` for a critic sub-agent.
        model_override: Canonical model id (``"claude-opus-4-7"``,
            etc.) the sub-agent should run on. ``None`` inherits.
        parallel: When ``True``, the orchestrator may dispatch this
            sub-agent concurrently with its parallel-marked peers.
        max_concurrent: Cap on simultaneous parallel sub-agents in a
            group; the orchestrator uses ``min(max_concurrent)`` of
            the group to size its semaphore. Ignored when
            ``parallel=False``.
        extras: Free-form bag for host-specific descriptor data
            (cost budget, persona ids, …).
        env_id: Stored-environment reference carried over from a
            manifest ``subagents`` entry (2.2.0 Wave 3). The library
            cannot resolve host storage itself — the default factory
            raises a :class:`ConfigError` telling the host to supply a
            ``subagent_env_resolver`` to ``Pipeline.from_manifest`` /
            ``from_manifest_async``. ``None`` for descriptors built
            outside the manifest path.
    """

    agent_type: str
    factory: PipelineFactory
    description: str = ""
    allowed_tools: Tuple[str, ...] = ()
    provider: Optional[str] = None
    provider_credentials_extras: Mapping[str, Any] = field(default_factory=dict)
    model_override: Optional[str] = None
    parallel: bool = False
    max_concurrent: int = 1
    extras: Mapping[str, Any] = field(default_factory=dict)
    env_id: Optional[str] = None


class SubagentTypeRegistry:
    """``agent_type`` → :class:`SubagentTypeDescriptor` map.

    Mirrors the surface of :class:`~geny_executor.tools.registry.
    ToolRegistry` for consistency. First-registration wins —
    duplicate ``agent_type`` is a ``ValueError`` so hosts catch
    bundled-vs-project collisions at boot time.
    """

    def __init__(self) -> None:
        self._descriptors: Dict[str, SubagentTypeDescriptor] = {}

    def register(self, descriptor: SubagentTypeDescriptor) -> "SubagentTypeRegistry":
        if descriptor.agent_type in self._descriptors:
            raise ValueError(f"subagent_type {descriptor.agent_type!r} already registered")
        self._descriptors[descriptor.agent_type] = descriptor
        return self

    def unregister(self, agent_type: str) -> None:
        self._descriptors.pop(agent_type, None)

    def get(self, agent_type: str) -> Optional[SubagentTypeDescriptor]:
        return self._descriptors.get(agent_type)

    def list_types(self) -> List[str]:
        return sorted(self._descriptors.keys())

    def __len__(self) -> int:
        return len(self._descriptors)

    def __contains__(self, agent_type: str) -> bool:
        return agent_type in self._descriptors


def resolve_subagent_provider(ctx: SubAgentBuildContext) -> Optional[str]:
    """Resolve the Stage 6 provider a sub-agent should run on — THE single home.

    Why one function (2.2.0 Wave 3, audit 2026-06-09 §2.8): provider
    inheritance was a dead contract — the read side
    (``parent_state_shared['primary_provider']``) shipped a full release
    with no producer, so ``descriptor.provider=None`` always fell
    through to host-global heuristics and a parent pinned to
    ``claude_code_cli`` could spawn sub-agents on a different backend
    (the #866 misrouting class, one level down). Wave 2 made the
    producer real (``Pipeline._init_state`` writes
    ``SharedKeys.PRIMARY_PROVIDER`` every run); this function encodes
    the full resolution order in one place so factories stop
    re-implementing (and re-drifting) it.

    Order:

    1. ``ctx.descriptor.provider`` — the entry's explicit override.
    2. ``ctx.parent_provider`` — typed inheritance field, populated by
       the orchestrator from the parent state's ``PRIMARY_PROVIDER``.
    3. ``ctx.parent_state_shared['primary_provider']`` — the legacy
       read-side key, honoured for contexts built by host code that
       predates the typed field.
    4. ``ctx.credentials.preferred_provider()`` — the Wave 1 bundle
       heuristic; the "nothing declared anywhere" fallback.

    Returns ``None`` when nothing resolves — callers must surface that
    loudly (the default factory raises :class:`ConfigError`) instead of
    inheriting a silent default.
    """
    descriptor_provider = getattr(ctx.descriptor, "provider", None)
    if descriptor_provider:
        return str(descriptor_provider)
    if ctx.parent_provider:
        return str(ctx.parent_provider)
    shared = ctx.parent_state_shared or {}
    inherited = str(shared.get(SharedKeys.PRIMARY_PROVIDER) or "")
    if inherited:
        return inherited
    credentials = ctx.credentials
    preferred = getattr(credentials, "preferred_provider", None)
    if callable(preferred):
        return preferred()
    return None


class ManifestSubagentPipelineFactory:
    """LIBRARY default :data:`PipelineFactory` for manifest ``subagents`` entries.

    2.2.0 Wave 3 (audit §1-1: "sub-agent environments are not
    first-class"): ``Pipeline.from_manifest`` compiles each manifest
    ``subagents`` entry into a :class:`SubagentTypeDescriptor` whose
    factory is one of these. Build behaviour per entry shape:

    - inline ``manifest`` dict → ``Pipeline.from_manifest_async(
      sub_manifest, credentials=<parent's>)`` — a fully declared
      sub-environment. When the entry *also* names an ``env_id``, the
      inline manifest wins (``validate_manifest`` flags the pair as
      ``subagent.dual_source``).
    - ``env_id`` → the library cannot resolve host storage; without a
      host-supplied resolver this raises :class:`ConfigError` telling
      the host to pass ``subagent_env_resolver=`` to
      ``Pipeline.from_manifest`` / ``from_manifest_async``. The
      resolver receives the ``env_id`` and returns an
      :class:`~geny_executor.core.environment.EnvironmentManifest` (or
      its dict form); it may be sync or async.
    - neither → ``build_manifest('worker_adaptive', provider=<resolved>,
      model=<descriptor.model_override>)`` with the descriptor's
      ``allowed_tools`` threaded into ``tools.built_in`` (empty tuple →
      ``["*"]``, the full built-in toolkit — the library default
      stand-in for "inherit parent"). Provider resolution goes through
      :func:`resolve_subagent_provider` — the single resolution-order
      home.

    The parent's :class:`CredentialBundle` (``ctx.credentials``) flows
    into every sub-build so authentication stays single-channel
    end-to-end.
    """

    def __init__(
        self,
        agent_type: str,
        *,
        inline_manifest: Optional[Mapping[str, Any]] = None,
        env_id: Optional[str] = None,
        env_resolver: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self._agent_type = agent_type
        self._inline_manifest = dict(inline_manifest) if inline_manifest else None
        self._env_id = env_id
        self._env_resolver = env_resolver

    @property
    def agent_type(self) -> str:
        return self._agent_type

    @property
    def env_id(self) -> Optional[str]:
        return self._env_id

    async def __call__(self, ctx: SubAgentBuildContext) -> Any:
        # Local imports: core.pipeline ←→ s12_agent would cycle at
        # module import time (the pipeline already imports this module
        # lazily for the same reason).
        from geny_executor.core.environment import EnvironmentManifest
        from geny_executor.core.pipeline import Pipeline

        if self._inline_manifest is not None:
            sub_manifest = EnvironmentManifest.from_dict(self._inline_manifest)
            return await Pipeline.from_manifest_async(sub_manifest, credentials=ctx.credentials)

        if self._env_id:
            if self._env_resolver is None:
                raise ConfigError(
                    f"subagents entry {self._agent_type!r} references "
                    f"env_id={self._env_id!r}, but stored environments are "
                    "host-resolved — the library has no access to the "
                    "host's environment storage. Pass "
                    "subagent_env_resolver=<callable> to "
                    "Pipeline.from_manifest / from_manifest_async; the "
                    "callable receives the env_id and returns an "
                    "EnvironmentManifest (or its dict form), sync or async."
                )
            resolved = self._env_resolver(self._env_id)
            if inspect.isawaitable(resolved):
                resolved = await resolved
            if isinstance(resolved, Mapping):
                resolved = EnvironmentManifest.from_dict(dict(resolved))
            return await Pipeline.from_manifest_async(resolved, credentials=ctx.credentials)

        # No sub-manifest anywhere — materialize the adaptive-worker
        # preset on the resolved provider.
        from geny_executor.core.manifest_factory import build_manifest

        provider = resolve_subagent_provider(ctx)
        if not provider:
            raise ConfigError(
                f"subagents entry {self._agent_type!r}: no provider could "
                "be resolved — the entry declares none, the parent "
                "published no PRIMARY_PROVIDER, and the credential bundle "
                "is empty. Declare 'provider' on the entry, or run the "
                "sub-agent under a parent pipeline whose Stage 6 provider "
                "resolved (Pipeline._init_state publishes it), or supply "
                "a CredentialBundle with at least one provider."
            )
        allowed = list(ctx.descriptor.allowed_tools) or ["*"]
        sub_manifest = build_manifest(
            "worker_adaptive",
            provider=provider,
            model=ctx.descriptor.model_override,
            built_in_tools=allowed,
            name=f"subagent:{self._agent_type}",
            description=ctx.descriptor.description
            or f"Sub-agent environment compiled from subagents entry {self._agent_type!r}.",
        )
        return await Pipeline.from_manifest_async(sub_manifest, credentials=ctx.credentials)


def compile_subagent_descriptors(
    subagents: Sequence[Mapping[str, Any]],
    *,
    env_resolver: Optional[Callable[[str], Any]] = None,
) -> List[SubagentTypeDescriptor]:
    """Compile manifest ``subagents`` entries into registrable descriptors.

    The library half of the 2.2.0 Wave 3 first-class sub-agent story:
    each well-formed entry becomes a :class:`SubagentTypeDescriptor`
    whose factory is a :class:`ManifestSubagentPipelineFactory`.
    Malformed entries (non-mapping, missing ``agent_type``) are skipped
    with a warning — :func:`~geny_executor.core.environment.
    validate_manifest` reports the same problems as ``subagent.*``
    errors at write time, so strict builds never reach this leniency.

    Args:
        subagents: ``manifest.subagents`` — plain dicts in the
            documented entry shape.
        env_resolver: Host callback for ``env_id`` entries (see
            :class:`ManifestSubagentPipelineFactory`). Optional —
            entries without ``env_id`` never need it.

    Returns:
        Descriptors in declaration order. Registration (and the
        explicit-registry-wins merge) is the caller's job —
        ``Pipeline.from_manifest`` owns that policy.
    """
    descriptors: List[SubagentTypeDescriptor] = []
    for raw in subagents or []:
        if not isinstance(raw, Mapping):
            logger.warning(
                "compile_subagent_descriptors: entry %r is not a mapping — skipped",
                raw,
            )
            continue
        agent_type = str(raw.get("agent_type") or "").strip()
        if not agent_type:
            logger.warning(
                "compile_subagent_descriptors: entry %r has no agent_type — "
                "skipped (validate_manifest flags this as subagent.missing_type)",
                raw,
            )
            continue
        inline = raw.get("manifest")
        env_id = raw.get("env_id")
        if inline and env_id:
            logger.warning(
                "compile_subagent_descriptors: entry %r sets both env_id and "
                "an inline manifest — the inline manifest wins "
                "(subagent.dual_source)",
                agent_type,
            )
        factory = ManifestSubagentPipelineFactory(
            agent_type,
            inline_manifest=inline if isinstance(inline, Mapping) else None,
            env_id=str(env_id) if env_id else None,
            env_resolver=env_resolver,
        )
        descriptors.append(
            SubagentTypeDescriptor(
                agent_type=agent_type,
                factory=factory,
                description=str(raw.get("description") or ""),
                allowed_tools=tuple(raw.get("allowed_tools") or ()),
                provider=str(raw["provider"]) if raw.get("provider") else None,
                model_override=(str(raw["model_override"]) if raw.get("model_override") else None),
                env_id=str(env_id) if env_id else None,
            )
        )
    return descriptors


async def _resolve_pipeline(factory: PipelineFactory, ctx: SubAgentBuildContext) -> Any:
    """Call a factory with the build context and unwrap an awaitable.

    For backward compatibility with zero-arg factories (the pre-D1
    shape), we try ``factory(ctx)`` first; if it raises ``TypeError``
    for an unexpected argument we fall back to ``factory()``.
    """
    try:
        result = factory(ctx)
    except TypeError as e:
        if "argument" not in str(e) and "positional" not in str(e):
            raise
        # Legacy zero-arg factory shape.
        result = factory()  # type: ignore[call-arg]
    if inspect.isawaitable(result):
        return await result
    return result


class SubagentTypeOrchestrator(AgentOrchestrator):
    """Dispatch ``state.delegate_requests`` against a registry.

    Each request is a dict with at minimum ``{"agent_type", "task"}``.
    Optional ``"args"`` is forwarded to the sub-pipeline as part of
    the run input. Results land on ``state.agent_results`` per the
    existing Stage 11 contract; the orchestrator only returns the
    aggregated :class:`AgentResult`.

    Failure isolation: an unknown ``agent_type`` produces a structured
    failure record (``success=False`` + ``error="unknown_agent_type"``)
    rather than aborting the whole batch. A factory crash is captured
    the same way.
    """

    def __init__(self, registry: Optional[SubagentTypeRegistry] = None):
        # ``registry`` is logically required, but accepting ``None`` lets
        # zero-arg construction work — which the ``StrategySlot`` machinery
        # uses while restoring a manifest that names ``"subagent_type"`` as
        # the orchestrator. The pipeline immediately replaces this instance
        # with one bound to the real registry via
        # ``Pipeline._wire_subagent_orchestrator``; until that runs, the
        # orchestrator behaves as if no descriptors are registered (every
        # delegate request lands as an "unknown_agent_type" failure).
        self._registry = registry if registry is not None else SubagentTypeRegistry()

    @property
    def name(self) -> str:
        return "subagent_type"

    @property
    def description(self) -> str:
        count = len(self._registry)
        return (
            f"Dispatch delegate_requests against {count} registered "
            f"subagent type{'s' if count != 1 else ''}"
        )

    @property
    def registry(self) -> SubagentTypeRegistry:
        return self._registry

    async def orchestrate(self, state: PipelineState) -> AgentResult:
        if not state.delegate_requests:
            return AgentResult(delegated=False)

        # Split requests into a serial group (parallel=False) and a
        # parallel group (parallel=True). Unknown agent_types go through
        # the serial path so the failure record is produced in the same
        # deterministic order as the request list.
        serial: List[Dict[str, Any]] = []
        parallel: List[Tuple[Dict[str, Any], SubagentTypeDescriptor]] = []
        for raw in state.delegate_requests:
            agent_type = str(raw.get("agent_type") or "").strip()
            desc = self._registry.get(agent_type)
            if desc is not None and desc.parallel:
                parallel.append((raw, desc))
            else:
                serial.append(raw)

        sub_results: List[Dict[str, Any]] = []
        # Serial first — preserves input order for deterministic logs.
        for raw in serial:
            sub_results.append(await self._dispatch_one(state, raw))

        # Parallel fan-out — bounded by min(max_concurrent) of the group.
        if parallel:
            cap = min(max(d.max_concurrent, 1) for _, d in parallel)
            sem = asyncio.Semaphore(cap)

            async def _bounded(raw_req: Dict[str, Any]) -> Dict[str, Any]:
                async with sem:
                    return await self._dispatch_one(state, raw_req)

            parallel_results = await asyncio.gather(
                *(_bounded(raw) for raw, _ in parallel),
                return_exceptions=False,
            )
            sub_results.extend(parallel_results)

        # Existing Stage 11 contract: requests are consumed once.
        state.delegate_requests = []
        return AgentResult(delegated=True, sub_results=sub_results)

    async def run_subagent(
        self,
        agent_type: str,
        prompt: str,
        *,
        model: Optional[str] = None,
        state: Optional[PipelineState] = None,
    ) -> Dict[str, Any]:
        """Single-call delegation surface (2.2.0 Wave 3).

        Why (audit 2026-06-09, "two incompatible delegation
        interfaces"): :class:`~geny_executor.tools.built_in.agent_tool.
        AgentTool` and :class:`~geny_executor.runtime.task_executors.
        LocalAgentExecutor` both dispatch via ``await runner(agent_type,
        prompt, model=...)`` looked up as ``run_subagent`` / ``spawn``
        — but this orchestrator only exposed the batch
        :meth:`orchestrate` shape, so wiring it into
        ``ToolContext.extras['agent_orchestrator']`` produced
        ORCHESTRATOR_API errors. This method is the call shape those
        consumers already speak.

        Args:
            agent_type: Registered descriptor id.
            prompt: Initial user prompt for the sub-pipeline.
            model: Optional per-call model override — applied as a
                one-shot ``descriptor.model_override`` replacement so
                the factory's normal model threading handles it.
            state: Parent :class:`PipelineState` — pass it whenever one
                exists so credentials / workspace / PRIMARY_PROVIDER
                inherit. When omitted (tool-context callers have no
                state handle) an ephemeral state is minted; provider
                inheritance then falls through to the descriptor /
                credential-bundle rungs of
                :func:`resolve_subagent_provider`.

        Returns:
            The structured sub-result record — the same shape as the
            entries :meth:`orchestrate` appends to ``sub_results``.

        Raises:
            KeyError: Unknown ``agent_type`` (AgentTool maps this to
                its UNKNOWN_TYPE error code).
            RuntimeError: Factory or sub-pipeline failure (the batch
                path's failure-isolation records, re-raised because a
                single-call consumer wants the error path, not a dict
                it must inspect).
        """
        descriptor = self._registry.get(agent_type)
        if descriptor is None:
            raise KeyError(agent_type)
        if model:
            descriptor = replace(descriptor, model_override=model)
        if state is None:
            state = PipelineState(session_id=f"subagent-adhoc-{uuid.uuid4().hex[:8]}")
        record = await self._run_descriptor(state, descriptor, agent_type, prompt)
        if not record.get("success", False):
            raise RuntimeError(record.get("error") or "sub-pipeline reported success=False")
        return record

    async def _dispatch_one(
        self,
        state: PipelineState,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        agent_type = str(request.get("agent_type") or "").strip()
        task = request.get("task", "")
        descriptor = self._registry.get(agent_type)

        if descriptor is None:
            logger.warning(
                "SubagentTypeOrchestrator: unknown agent_type %r — request rejected",
                agent_type,
            )
            return {
                "agent_type": agent_type,
                "task": task,
                "subagent_metadata": None,
                "success": False,
                "text": "",
                "error": f"unknown_agent_type: {agent_type!r}",
            }

        return await self._run_descriptor(state, descriptor, agent_type, task)

    async def _run_descriptor(
        self,
        state: PipelineState,
        descriptor: SubagentTypeDescriptor,
        agent_type: str,
        task: Any,
    ) -> Dict[str, Any]:
        """Build + run one sub-pipeline for *descriptor*; never raises.

        Shared by the batch path (:meth:`_dispatch_one`) and the
        single-call path (:meth:`run_subagent`) so both produce the
        identical record shape from the identical build context.
        """
        base_record: Dict[str, Any] = {
            "agent_type": agent_type,
            "task": task,
            "subagent_metadata": None,
        }

        # Attach the descriptor's static metadata so audit / UI
        # surfaces can render the sub-agent's name + roster without
        # walking the registry separately.
        base_record["subagent_metadata"] = {
            "description": descriptor.description,
            "allowed_tools": list(descriptor.allowed_tools),
            "provider": descriptor.provider,
            "model_override": descriptor.model_override,
            "parallel": descriptor.parallel,
            "max_concurrent": descriptor.max_concurrent,
            "extras": dict(descriptor.extras),
            "env_id": descriptor.env_id,
        }

        # Build the context handed to the factory. The parent's
        # CredentialBundle (populated by Pipeline._init_state from
        # the bundle passed to from_manifest_async) flows down so the
        # sub-pipeline's Stage 6 can authenticate with the right
        # provider without re-asking the host. ``parent_provider`` is
        # the typed inheritance field (audit host_ergonomics #7) fed
        # from the PRIMARY_PROVIDER key _init_state publishes.
        ws_snapshot = state.shared.get("workspace_snapshot")
        sub_session_id = f"{state.session_id}-{agent_type}-{uuid.uuid4().hex[:8]}"
        ctx = SubAgentBuildContext(
            parent_session_id=state.session_id,
            sub_session_id=sub_session_id,
            credentials=state.credentials,
            descriptor=descriptor,
            workspace_snapshot=ws_snapshot,
            parent_state_shared=dict(state.shared),
            parent_provider=(str(state.shared.get(SharedKeys.PRIMARY_PROVIDER) or "") or None),
        )

        try:
            sub_pipeline = await _resolve_pipeline(descriptor.factory, ctx)
        except Exception as exc:
            logger.warning(
                "SubagentTypeOrchestrator: factory for %r raised: %s",
                agent_type,
                exc,
                exc_info=True,
            )
            return {
                **base_record,
                "success": False,
                "text": "",
                "error": f"factory_error: {exc}",
            }

        sub_state = PipelineState(session_id=sub_session_id)

        # Thread workspace context to the sub-pipeline.
        if ws_snapshot is not None:
            sub_state.shared["workspace_snapshot"] = ws_snapshot

        try:
            result = await sub_pipeline.run(task, sub_state)
        except Exception as exc:
            logger.warning(
                "SubagentTypeOrchestrator: sub-pipeline for %r raised: %s",
                agent_type,
                exc,
                exc_info=True,
            )
            return {
                **base_record,
                "success": False,
                "text": "",
                "error": f"run_error: {exc}",
            }

        return {
            **base_record,
            "success": getattr(result, "success", True),
            "text": getattr(result, "text", ""),
            "error": getattr(result, "error", None),
        }
