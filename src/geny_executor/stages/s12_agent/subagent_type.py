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
"""

from __future__ import annotations

import inspect
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Tuple, Union

from geny_executor.core.state import PipelineState
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


async def _resolve_pipeline(
    factory: PipelineFactory, ctx: SubAgentBuildContext
) -> Any:
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

    def __init__(self, registry: SubagentTypeRegistry):
        self._registry = registry

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

        sub_results: List[Dict[str, Any]] = []
        for raw in state.delegate_requests:
            sub_results.append(await self._dispatch_one(state, raw))

        # Existing Stage 11 contract: requests are consumed once.
        state.delegate_requests = []
        return AgentResult(delegated=True, sub_results=sub_results)

    async def _dispatch_one(
        self,
        state: PipelineState,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        agent_type = str(request.get("agent_type") or "").strip()
        task = request.get("task", "")
        descriptor = self._registry.get(agent_type)

        base_record: Dict[str, Any] = {
            "agent_type": agent_type,
            "task": task,
            "subagent_metadata": None,
        }

        if descriptor is None:
            logger.warning(
                "SubagentTypeOrchestrator: unknown agent_type %r — request rejected",
                agent_type,
            )
            return {
                **base_record,
                "success": False,
                "text": "",
                "error": f"unknown_agent_type: {agent_type!r}",
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
        }

        # Build the context handed to the factory. The parent's
        # CredentialBundle (populated by Pipeline._init_state from
        # the bundle passed to from_manifest_async) flows down so the
        # sub-pipeline's Stage 6 can authenticate with the right
        # provider without re-asking the host.
        ws_snapshot = state.shared.get("workspace_snapshot")
        sub_session_id = f"{state.session_id}-{agent_type}-{uuid.uuid4().hex[:8]}"
        ctx = SubAgentBuildContext(
            parent_session_id=state.session_id,
            sub_session_id=sub_session_id,
            credentials=state.credentials,
            descriptor=descriptor,
            workspace_snapshot=ws_snapshot,
            parent_state_shared=dict(state.shared),
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
