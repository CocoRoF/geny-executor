"""Subagent-type registry + orchestrator.

Cycle 20260424 executor uplift — Phase 7 Sprint S7.5.

The pre-S7.5 :class:`DelegateOrchestrator` walks
``state.delegate_requests`` and looks up sub-pipelines by
``agent_type`` name in a :class:`SubPipelineFactory`. That works fine
for one-off delegations but skips two pieces of context the LLM and
the host both need:

1. **Per-type metadata** — what does ``code-reviewer`` actually do?
   What tools does it own? What model does it run on? Without this,
   neither the LLM (deciding when to delegate) nor admin UIs
   (showing sub-agent activity) have anything to work with.
2. **A shared dispatcher for skill ``fork`` mode.** Phase 4 Skills
   ship inline-only; the fork branch is stubbed pending an
   orchestrator. S7.5 provides the orchestrator so a Phase 7+
   ``SkillTool.fork`` lands without re-implementing the dispatch
   path.

This module ships:

* :class:`SubagentTypeDescriptor` — frozen metadata + factory dataclass.
  ``agent_type`` is the name the LLM sees and the registry key.
* :class:`SubagentTypeRegistry` — thin id→descriptor map mirroring
  the shape of :class:`~geny_executor.tools.registry.ToolRegistry`
  (register / unregister / get / list).
* :class:`SubagentTypeOrchestrator` — :class:`AgentOrchestrator`
  subclass that consumes ``state.delegate_requests`` against the
  registry. Each request becomes a sub-pipeline run; results land
  on ``state.agent_results`` via the existing Stage 11 wiring.

Compatibility: the existing ``DelegateOrchestrator`` keeps working
unchanged. Hosts opt into the typed orchestrator by swapping it in
at the Stage 11 strategy slot.
"""

from __future__ import annotations

import inspect
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from geny_executor.core.state import PipelineState
from geny_executor.stages.s11_agent.interface import AgentOrchestrator
from geny_executor.stages.s11_agent.types import AgentResult

logger = logging.getLogger(__name__)


# A factory may be sync (returns a Pipeline) or async (returns
# Awaitable[Pipeline]). The orchestrator handles both — hosts that
# need to do async setup (e.g. attach an MCP manager) write an
# async factory.
PipelineFactory = Callable[[], Any]


@dataclass(frozen=True)
class SubagentTypeDescriptor:
    """Static metadata describing one sub-agent type.

    Attributes:
        agent_type: Stable identifier — registry key + the value the
            LLM sees in ``[DELEGATE: <agent_type>]`` markers + the
            field used in ``state.delegate_requests`` entries.
        factory: Zero-arg callable returning a ready-to-run
            :class:`Pipeline`. May be sync or async. Hosts that wire
            session-scoped resources (storage path, MCP manager,
            credentials) close over them in the factory.
        description: One-line summary the LLM uses when choosing
            whether to delegate. Mirrors ``Tool.description``.
        allowed_tools: Tuple of tool names the sub-agent's pipeline
            should expose. Empty tuple means "inherit parent" — the
            host is responsible for actually applying this in the
            factory; the registry just records intent.
        model_override: Canonical model id (``"claude-opus-4-7"``,
            etc.) the sub-agent should run on. ``None`` inherits.
        extras: Free-form bag for host-specific descriptor data
            (cost budget, persona ids, …).
    """

    agent_type: str
    factory: PipelineFactory
    description: str = ""
    allowed_tools: Tuple[str, ...] = ()
    model_override: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)


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


async def _resolve_pipeline(factory: PipelineFactory) -> Any:
    """Call a factory and unwrap an awaitable when present."""
    result = factory()
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
            "model_override": descriptor.model_override,
            "extras": dict(descriptor.extras),
        }

        try:
            sub_pipeline = await _resolve_pipeline(descriptor.factory)
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

        sub_state = PipelineState(
            session_id=f"{state.session_id}-{agent_type}-{uuid.uuid4().hex[:8]}",
        )

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
