"""Persistent sub-agents — the *owned, stateful, autonomous* delegate.

Two delegation primitives now live in the executor, deliberately distinct:

* **sub-worker** (one-shot): :meth:`SubagentTypeOrchestrator.run_subagent`
  builds a sub-pipeline, runs it once, returns the result, and closes it.
  Stateless. Use it to delegate a *specific* task and consume the answer in
  the same reasoning step. (Surfaced to the LLM as the ``Agent`` tool.)

* **sub-agent** (persistent): this module. An owner spawns a *named, kept-
  alive* sub-agent instance; assigns it a task to complete **autonomously**
  (in the background); and is **notified on completion** via the owner's
  inbox. The instance keeps its conversation/state across assignments
  (multi-turn) and survives until explicitly stopped. This is the
  "fully delegate → it finishes on its own → you get the alarm" model.

The mechanism — the inbox, the completion notification, the lifecycle — is
provided *here*, in the framework, so any host (e.g. Geny) is just a
consumer: it injects a ``session_store`` (where to persist sub-agent state)
and an ``on_event`` callback (to mirror lifecycle into its own UI / task
list), then spawns / assigns / drains the inbox. Hosts do not re-implement
delegation, inboxing, or notification.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from geny_executor.core.shared_keys import SharedKeys
from geny_executor.core.state import PipelineState
from geny_executor.stages.s12_agent.subagent_type import (
    SubAgentBuildContext,
    SubagentTypeRegistry,
    _resolve_pipeline,
)

logger = logging.getLogger(__name__)


# ── Assignment transcript capture (2.34.0) ──────────────────────────────
# A sub-agent runs its OWN pipeline; without observing it the host only sees
# the final result, never *how* it got there. We subscribe to the sub-
# pipeline's event bus for the duration of an assignment and normalize the
# tool / result / error traffic into a compact, bounded transcript that the
# completion event carries. A host (e.g. Geny) can then render the sub-
# agent's own TOOL/RESULT trail instead of falling back to the owner's
# pipeline log. Capture is best-effort: a collector that raises must never
# break the run.

_MAX_TRANSCRIPT_STEPS = 400
_MAX_TRANSCRIPT_BYTES = 256_000
_TRANSCRIPT_FIELD_LIMIT = 4000
_TRANSCRIPT_INPUT_LIMIT = 1000
_MAX_CLIP_DEPTH = 4
_MAX_LIST_ITEMS = 50


def _truncate_text(value: Any, limit: int = _TRANSCRIPT_FIELD_LIMIT) -> str:
    s = value if isinstance(value, str) else ("" if value is None else str(value))
    return s if len(s) <= limit else (s[:limit] + f"… (+{len(s) - limit} chars)")


def _clip_input(value: Any, depth: int = 0) -> Any:
    """Keep an input's shape but bound long strings AND nested containers.

    Recurses (bounded depth + list width) so a tool input with deeply nested
    or list-of-long-string payloads (e.g. a structured file-write) cannot blow
    past the transcript budget — the pre-2.35 version only clipped top-level
    string values (audit 2026-06-25).
    """
    if depth >= _MAX_CLIP_DEPTH:
        return "… (nested)"
    if isinstance(value, dict):
        return {k: _clip_input(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        clipped = [_clip_input(v, depth + 1) for v in value[:_MAX_LIST_ITEMS]]
        if len(value) > _MAX_LIST_ITEMS:
            clipped.append(f"… (+{len(value) - _MAX_LIST_ITEMS} items)")
        return clipped
    if isinstance(value, str):
        return _truncate_text(value, _TRANSCRIPT_INPUT_LIMIT)
    return value


def _result_to_text(content: Any) -> str:
    """Flatten a tool_result ``content`` (str | list[block] | dict) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return _truncate_text(content)
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
            else:
                parts.append(str(block))
        return _truncate_text("\n".join(p for p in parts if p))
    if isinstance(content, dict):
        return _truncate_text(content.get("text") or content.get("content") or content)
    return _truncate_text(content)


class _TranscriptCollector:
    """Subscribe to a sub-pipeline bus and build a normalized step list.

    Steps (plain dicts, host-agnostic)::

        {"type": "tool", "id", "name", "input", "is_error", "duration_ms", "result", "ts"}
        {"type": "error", "message", "ts"}
        {"type": "truncated", "note"}     # appended once when a bound is hit

    Covers BOTH provider paths so the trail is complete regardless of backend:
      * Stage-10 dispatch — ``tool.call_start`` / ``tool.call_complete``
      * CLI provider      — ``api.cli_tool_call`` / ``api.tool_result``

    ``session_id`` scopes the feed: ``pipeline.on("*")`` is a complete bus feed,
    so on a pipeline shared across runs (e.g. a host factory that reuses the
    parent's pipeline) events from a *different* conversation would otherwise
    cross-contaminate this transcript. We drop any event whose ``session_id``
    is set and differs (audit 2026-06-25). Bounded by BOTH a step count and a
    cumulative byte budget; on either limit a single ``truncated`` sentinel is
    appended so the host can show the trail was cut.
    """

    def __init__(self, session_id: str = "") -> None:
        self.steps: List[Dict[str, Any]] = []
        self._by_id: Dict[str, Dict[str, Any]] = {}
        self._session_id = session_id or ""
        self._bytes = 0
        self._truncated = False

    def _full(self) -> bool:
        if self._truncated:
            return True
        if len(self.steps) >= _MAX_TRANSCRIPT_STEPS or self._bytes >= _MAX_TRANSCRIPT_BYTES:
            self.steps.append({"type": "truncated", "note": "trail truncated (limit reached)"})
            self._truncated = True
            return True
        return False

    def _append(self, step: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self._full():
            return None
        self.steps.append(step)
        try:
            self._bytes += len(repr(step))
        except Exception:  # noqa: BLE001
            pass
        return step

    def _open_tool(self, tid: str, name: str, tool_input: Any, ts: str) -> None:
        if not name:
            return
        step = self._append({
            "type": "tool",
            "id": tid,
            "name": name,
            "input": _clip_input(tool_input),
            "ts": ts,
        })
        if step is not None and tid:
            self._by_id[tid] = step

    def __call__(self, event: Any) -> None:
        try:
            # Run scoping: ignore traffic from other conversations on a shared bus.
            if self._session_id:
                ev_sid = getattr(event, "session_id", "") or ""
                if ev_sid and ev_sid != self._session_id:
                    return
            et = getattr(event, "type", "") or ""
            data = getattr(event, "data", None) or {}
            ts = getattr(event, "timestamp", "") or ""
            if et == "tool.call_start":
                self._open_tool(
                    str(data.get("tool_use_id") or ""),
                    str(data.get("name") or ""),
                    data.get("input"),
                    ts,
                )
            elif et == "api.cli_tool_call":
                name = str(data.get("name") or "")
                if name and not name.startswith("mcp__"):
                    self._open_tool(str(data.get("id") or ""), name, data.get("input"), ts)
            elif et == "tool.call_complete":
                step = self._by_id.get(str(data.get("tool_use_id") or ""))
                if step is not None:
                    step["is_error"] = bool(data.get("is_error"))
                    step["duration_ms"] = data.get("duration_ms")
            elif et == "api.tool_result":
                step = self._by_id.get(str(data.get("tool_use_id") or ""))
                if step is not None:
                    step["result"] = _result_to_text(data.get("content"))
                    if data.get("is_error"):
                        step["is_error"] = True
            elif et in ("api.error", "pipeline.error"):
                msg = data.get("message") if isinstance(data, dict) else data
                self._append({"type": "error", "message": _truncate_text(msg or data), "ts": ts})
        except Exception:  # noqa: BLE001 — observability must never break a run
            pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Inbox ────────────────────────────────────────────────────────────


@dataclass
class InboxMessage:
    """One message in an owner's inbox.

    ``kind`` is the routing discriminator a host renders on:
        ``completion`` — a sub-agent finished an assignment (the alarm).
        ``failed``     — an assignment errored.
        ``message``    — a free-form note from a sub-agent (multi-turn).
    """

    id: str
    owner: str  # recipient session id (who is notified)
    sender: str  # sub_agent_id that produced it
    kind: str
    body: str
    data: Dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "owner": self.owner,
            "sender": self.sender,
            "kind": self.kind,
            "body": self.body,
            "data": dict(self.data),
            "ts": self.ts,
        }


class SubAgentInbox:
    """In-process mailbox keyed by owner session id.

    The completion-notification substrate: when a persistent sub-agent
    finishes an assignment, the manager delivers an :class:`InboxMessage`
    here for its owner. The owner (host) drains it to surface the alarm.
    Bounded per owner so a runaway producer can't grow unboundedly.
    """

    def __init__(self, *, max_per_owner: int = 200) -> None:
        self._by_owner: Dict[str, List[InboxMessage]] = {}
        self._max = max_per_owner

    def deliver(self, msg: InboxMessage) -> None:
        box = self._by_owner.setdefault(msg.owner, [])
        box.append(msg)
        if len(box) > self._max:
            del box[: len(box) - self._max]  # drop oldest

    def peek(self, owner: str) -> List[InboxMessage]:
        return list(self._by_owner.get(owner, []))

    def drain(self, owner: str) -> List[InboxMessage]:
        return self._by_owner.pop(owner, [])

    def count(self, owner: str) -> int:
        return len(self._by_owner.get(owner, []))


# ── Persistent sub-agent handle ──────────────────────────────────────


@dataclass
class PersistentSubAgent:
    """A live, owned sub-agent instance.

    Holds the kept-alive sub-pipeline plus the persisted state that makes
    it stateful across assignments. ``status`` is observational:
    ``idle`` | ``running`` | ``stopped``.
    """

    sub_agent_id: str
    agent_type: str
    owner_session_id: str
    pipeline: Any
    state: PipelineState
    status: str = "idle"
    created_at: str = field(default_factory=_now_iso)
    last_assigned_at: Optional[str] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def summary(self) -> Dict[str, Any]:
        return {
            "sub_agent_id": self.sub_agent_id,
            "agent_type": self.agent_type,
            "owner_session_id": self.owner_session_id,
            "status": self.status,
            "created_at": self.created_at,
            "last_assigned_at": self.last_assigned_at,
            "messages": len(getattr(self.state, "messages", []) or []),
        }


# ── Manager ──────────────────────────────────────────────────────────

# session_store contract (host-supplied): an object with async or sync
# ``load(sub_agent_id) -> PipelineState | None`` and ``save(sub_agent_id,
# state) -> None``. Optional; without it sub-agents are still persistent
# in-process but do not survive a host restart.
EventCallback = Callable[[str, Dict[str, Any]], Union[None, Awaitable[None]]]


class SubAgentManager:
    """Owns persistent sub-agent instances + the notification inbox.

    Host wires one instance into ``ToolContext.extras['subagent_manager']``
    (and, for background tasks, onto its app state). The SubAgent* tools
    and the host both drive it through this surface.
    """

    def __init__(
        self,
        registry: SubagentTypeRegistry,
        *,
        inbox: Optional[SubAgentInbox] = None,
        session_store: Any = None,
        on_event: Optional[EventCallback] = None,
        credentials_provider: Optional[
            Callable[[str], Union[Optional[Dict[str, Any]], Awaitable[Optional[Dict[str, Any]]]]]
        ] = None,
    ) -> None:
        self._registry = registry
        self.inbox = inbox or SubAgentInbox()
        self._session_store = session_store
        self._on_event = on_event
        # Host callback ``owner_session_id -> {"credentials", "provider"} | None``,
        # consulted by spawn() when no explicit credentials are passed (e.g. an
        # ad-hoc SubAgentSpawn tool call, which can't know the owner's bundle).
        # Without it, an ad-hoc-spawned sub-agent has no credentials and its
        # Stage-6 auth fails — only the host-spawned owned companion (which
        # passes credentials=) worked (integrity audit 2026-06-25). May be async.
        self._credentials_provider = credentials_provider
        self._agents: Dict[str, PersistentSubAgent] = {}
        self._tasks: Dict[str, asyncio.Task] = {}  # assignment_id -> task
        # Serializes spawn(): the check→build→store sequence awaits (pipeline
        # build, state load), so without this two concurrent spawns of the same
        # sub_agent_id both build a pipeline and the loser leaks (live MCP child).
        self._spawn_lock = asyncio.Lock()

    # ---- lifecycle ----

    async def spawn(
        self,
        agent_type: str,
        owner_session_id: str,
        *,
        factory: Any = None,
        sub_agent_id: Optional[str] = None,
        credentials: Any = None,
        parent_provider: Optional[str] = None,
        workspace_snapshot: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> PersistentSubAgent:
        """Create (or reattach) a persistent sub-agent owned by *owner*.

        Builds the sub-pipeline via the descriptor's factory and keeps it
        alive. When a ``session_store`` is wired and a prior state exists
        for ``sub_agent_id``, the conversation is restored (restart /
        reattach). Raises ``KeyError`` for an unknown ``agent_type``.

        ``factory`` (optional) is a host-supplied ``PipelineFactory`` used
        instead of the registry's — e.g. to build the companion from the
        PARENT agent's environment (so it inherits the parent's tools / model
        / stages). When given, ``agent_type`` is just a label and the registry
        is not consulted.

        ``model`` / ``system_prompt`` are per-spawn overrides carried on the
        descriptor (``model_override`` / ``system_prompt``) so the factory can
        honour them when building the pipeline.
        """
        from dataclasses import replace as _replace
        from geny_executor.stages.s12_agent.subagent_type import (
            SubagentTypeDescriptor as _Descriptor,
        )

        if factory is not None:
            descriptor = _Descriptor(
                agent_type=agent_type or "owned",
                factory=factory,
                model_override=model,
                system_prompt=system_prompt,
            )
        else:
            descriptor = self._registry.get(agent_type)
            if descriptor is None:
                raise KeyError(agent_type)
            if model is not None or system_prompt is not None:
                descriptor = _replace(
                    descriptor,
                    model_override=model or descriptor.model_override,
                    system_prompt=(
                        system_prompt if system_prompt is not None else descriptor.system_prompt
                    ),
                )

        sid = sub_agent_id or f"{owner_session_id}-{agent_type}-{uuid.uuid4().hex[:8]}"

        # Serialize the check→build→store so concurrent spawns of the same id
        # don't both build a pipeline (the loser would leak its MCP child).
        async with self._spawn_lock:
            existing = self._agents.get(sid)
            if existing is not None:
                # Idempotent reattach. Refresh rotated credentials / snapshot so a
                # re-spawn with new auth doesn't keep authenticating with stale
                # creds (audit 2026-06-25); the live conversation is preserved.
                if credentials is not None:
                    try:
                        existing.state.credentials = credentials
                    except Exception:  # noqa: BLE001
                        pass
                if workspace_snapshot is not None:
                    existing.state.shared["workspace_snapshot"] = workspace_snapshot
                return existing

            # Resolve credentials/provider from the host when the caller didn't
            # supply them (the ad-hoc SubAgentSpawn tool path). The owned
            # companion is spawned host-side WITH credentials, so this only fires
            # for tool-initiated spawns that would otherwise fail Stage-6 auth.
            if credentials is None and self._credentials_provider is not None:
                try:
                    resolved = self._credentials_provider(owner_session_id)
                    if inspect.isawaitable(resolved):
                        resolved = await resolved
                    if resolved:
                        credentials = resolved.get("credentials")
                        parent_provider = parent_provider or resolved.get("provider")
                except Exception:  # noqa: BLE001 — best effort; fall through to resolve error
                    logger.debug(
                        "credentials_provider failed for owner %s",
                        owner_session_id,
                        exc_info=True,
                    )

            ctx = SubAgentBuildContext(
                parent_session_id=owner_session_id,
                sub_session_id=sid,
                credentials=credentials,
                descriptor=descriptor,
                workspace_snapshot=workspace_snapshot,
                parent_state_shared={SharedKeys.PRIMARY_PROVIDER: parent_provider or ""},
                parent_provider=parent_provider,
            )
            pipeline = await _resolve_pipeline(descriptor.factory, ctx)

            state = await self._load_state(sid)
            if state is None:
                state = PipelineState(session_id=sid)
            if credentials is not None:
                try:
                    state.credentials = credentials
                except Exception:  # noqa: BLE001
                    pass
            if workspace_snapshot is not None:
                state.shared["workspace_snapshot"] = workspace_snapshot

            agent = PersistentSubAgent(
                sub_agent_id=sid,
                agent_type=agent_type,
                owner_session_id=owner_session_id,
                pipeline=pipeline,
                state=state,
            )
            self._agents[sid] = agent
        await self._emit("subagent.spawned", agent.summary())
        return agent

    def get(self, sub_agent_id: str) -> Optional[PersistentSubAgent]:
        return self._agents.get(sub_agent_id)

    def list(self, owner_session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        out = [
            a.summary()
            for a in self._agents.values()
            if owner_session_id is None or a.owner_session_id == owner_session_id
        ]
        return sorted(out, key=lambda s: s["created_at"])

    async def cancel_assignment(self, assignment_id: str) -> bool:
        """Cancel ONE in-flight assignment WITHOUT tearing down the sub-agent.

        This is what a host's per-task "stop" should call: it cancels just that
        assignment's task and AWAITS it (so the run fully unwinds — state reload,
        lock release — before we return), leaving the persistent sub-agent alive
        and reusable. Idempotent; returns False for an unknown/finished id.

        Contrast :meth:`stop`, which destroys the whole sub-agent.
        """
        task = self._tasks.get(assignment_id)
        if task is None or task.done():
            return False
        task.cancel()
        try:
            await task  # let the assignment unwind (CancelledError path)
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        self._tasks.pop(assignment_id, None)
        return True

    async def stop(self, sub_agent_id: str) -> bool:
        """Cancel in-flight assignments, close the pipeline, drop the sub-agent.

        Destroys the persistent sub-agent entirely — use for an explicit "kill
        this companion" / shutdown, NOT for cancelling a single task (that's
        :meth:`cancel_assignment`). Cancels then AWAITS each assignment before
        closing the pipeline, so a still-running assignment can't resume on a
        half-closed pipeline or deliver a spurious completion (audit 2026-06-25).
        """
        agent = self._agents.pop(sub_agent_id, None)
        if agent is None:
            return False
        prefix = f"{sub_agent_id}:"
        to_cancel = [
            t for aid, t in list(self._tasks.items())
            if aid.startswith(prefix) and not t.done()
        ]
        for t in to_cancel:
            t.cancel()
        for t in to_cancel:
            try:
                await t
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        for aid in [a for a in list(self._tasks) if a.startswith(prefix)]:
            self._tasks.pop(aid, None)
        await self._aclose(agent.pipeline)
        agent.status = "stopped"
        await self._emit("subagent.stopped", {"sub_agent_id": sub_agent_id})
        return True

    async def shutdown(self) -> None:
        for sid in list(self._agents.keys()):
            await self.stop(sid)

    # ---- assignment (autonomous + notify) ----

    async def assign(
        self,
        sub_agent_id: str,
        task: str,
        *,
        background: bool = True,
    ) -> Dict[str, Any]:
        """Assign a task to a persistent sub-agent.

        ``background=True`` (default) schedules autonomous execution and
        returns immediately with ``{assignment_id, status: "running"}`` —
        the owner is notified via its inbox on completion (the alarm).
        ``background=False`` awaits and returns the full result record
        (useful for tests / synchronous hosts). Raises ``KeyError`` for an
        unknown ``sub_agent_id``.
        """
        agent = self._agents.get(sub_agent_id)
        if agent is None:
            raise KeyError(sub_agent_id)

        assignment_id = f"{sub_agent_id}:{uuid.uuid4().hex[:8]}"
        if not background:
            return await self._run_assignment(agent, assignment_id, task)

        async def _runner() -> None:
            try:
                await self._run_assignment(agent, assignment_id, task)
            finally:
                self._tasks.pop(assignment_id, None)

        self._tasks[assignment_id] = asyncio.ensure_future(_runner())
        return {
            "assignment_id": assignment_id,
            "sub_agent_id": sub_agent_id,
            "status": "running",
        }

    async def _run_assignment(
        self, agent: PersistentSubAgent, assignment_id: str, task: str
    ) -> Dict[str, Any]:
        # Serialize per-agent: one assignment at a time mutates its state.
        async with agent.lock:
            agent.status = "running"
            agent.last_assigned_at = _now_iso()
            await self._emit(
                "subagent.assigned",
                {"assignment_id": assignment_id, **agent.summary(), "task": task},
            )
            # Observe the sub-pipeline's OWN tool/result/error traffic for the
            # duration of this assignment so the completion event can carry a
            # real trail (see _TranscriptCollector). pipeline.on("*") is a
            # complete feed — Stage-10 tool.* and CLI api.* alike are bridged
            # onto the bus via state.add_event (executor ≥2.2.0).
            collector = _TranscriptCollector(session_id=agent.sub_agent_id)
            unsub: Optional[Callable[[], Any]] = None
            try:
                unsub = agent.pipeline.on("*", collector)
            except Exception:  # noqa: BLE001 — capture is strictly optional
                unsub = None
            record: Dict[str, Any]
            try:
                result = await agent.pipeline.run(task, agent.state)
                record = {
                    "assignment_id": assignment_id,
                    "sub_agent_id": agent.sub_agent_id,
                    "agent_type": agent.agent_type,
                    "success": bool(getattr(result, "success", True)),
                    "text": getattr(result, "text", "") or "",
                    "error": getattr(result, "error", None),
                }
            except asyncio.CancelledError:
                agent.status = "idle"
                # The turn was mutated in-place mid-run; discard the partial turn
                # by reloading the last persisted state so a cancelled assignment
                # can't corrupt the agent's conversation (audit 2026-06-25).
                try:
                    restored = await self._load_state(agent.sub_agent_id)
                    if restored is not None:
                        agent.state = restored
                except Exception:  # noqa: BLE001 — best effort on the cancel path
                    pass
                raise
            except Exception as exc:  # noqa: BLE001 — isolate; report as failed
                logger.warning(
                    "SubAgentManager: assignment %s for %r failed: %s",
                    assignment_id,
                    agent.agent_type,
                    exc,
                    exc_info=True,
                )
                record = {
                    "assignment_id": assignment_id,
                    "sub_agent_id": agent.sub_agent_id,
                    "agent_type": agent.agent_type,
                    "success": False,
                    "text": "",
                    "error": f"run_error: {exc}",
                }
            finally:
                if unsub is not None:
                    try:
                        unsub()
                    except Exception:  # noqa: BLE001
                        pass
            agent.status = "idle"

        # Persist accumulated state (multi-turn survives restart).
        await self._save_state(agent.sub_agent_id, agent.state)

        # Deliver the completion alarm to the owner's inbox + emit event.
        kind = "completion" if record.get("success") else "failed"
        msg = InboxMessage(
            id=uuid.uuid4().hex,
            owner=agent.owner_session_id,
            sender=agent.sub_agent_id,
            kind=kind,
            body=record.get("text") or (record.get("error") or ""),
            data=record,
        )
        self.inbox.deliver(msg)
        event_payload = {
            **record,
            "owner_session_id": agent.owner_session_id,
            "inbox_message_id": msg.id,
            # The sub-agent's own normalized tool/result/error trail (2.34.0).
            # Carried on the event (NOT the lean inbox record) so a host can
            # render the sub-agent's activity; absent ⇒ host falls back.
            "transcript": collector.steps,
        }
        # Literal event names (the catalogue honesty test scans _emit sites).
        if record.get("success"):
            await self._emit("subagent.completed", event_payload)
        else:
            await self._emit("subagent.failed", event_payload)
        return record

    # ---- inbox surface ----

    def read_inbox(self, owner_session_id: str, *, drain: bool = True) -> List[Dict[str, Any]]:
        msgs = self.inbox.drain(owner_session_id) if drain else self.inbox.peek(owner_session_id)
        return [m.to_dict() for m in msgs]

    # ---- internals ----

    async def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self._on_event is None:
            return
        try:
            result = self._on_event(event_type, payload)
            if asyncio.iscoroutine(result):
                await result
        except Exception:  # noqa: BLE001 — observers must not break the run
            logger.debug("SubAgentManager: on_event(%s) raised", event_type, exc_info=True)

    async def _load_state(self, sub_agent_id: str) -> Optional[PipelineState]:
        if self._session_store is None:
            return None
        try:
            load = getattr(self._session_store, "load", None)
            if load is None:
                return None
            result = load(sub_agent_id)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        except Exception:  # noqa: BLE001
            logger.debug("SubAgentManager: state load failed for %s", sub_agent_id, exc_info=True)
            return None

    async def _save_state(self, sub_agent_id: str, state: PipelineState) -> None:
        if self._session_store is None:
            return
        try:
            save = getattr(self._session_store, "save", None)
            if save is None:
                return
            result = save(sub_agent_id, state)
            if asyncio.iscoroutine(result):
                await result
        except Exception:  # noqa: BLE001
            logger.debug("SubAgentManager: state save failed for %s", sub_agent_id, exc_info=True)

    @staticmethod
    async def _aclose(pipeline: Any) -> None:
        aclose = getattr(pipeline, "aclose", None)
        if not callable(aclose):
            return
        try:
            result = aclose()
            if inspect.isawaitable(result):
                await result
        except Exception:  # noqa: BLE001
            logger.debug("SubAgentManager: pipeline aclose failed", exc_info=True)


__all__ = [
    "InboxMessage",
    "SubAgentInbox",
    "PersistentSubAgent",
    "SubAgentManager",
]
