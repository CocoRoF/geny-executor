"""Self-modifying environment — the live controller a session uses to edit
its OWN operating environment (system prompt, active tools, active skills)
at runtime.

A session reaches this through the built-in ``env_*`` tools (see
``geny_executor.tools.built_in.env_tools``); those are thin wrappers that call
the controller. The controller mutates the LIVE pipeline runtime so changes
take effect on the NEXT turn:

  * Tools/skills — register/unregister on the live :class:`ToolRegistry`. Its
    ``version`` bumps, so Stage 3 (System) re-derives ``state.tools`` next turn.
  * Prompt — edit the installed :class:`MutablePromptBuilder`; Stage 3 calls
    ``build()`` every turn, so the edit shows up next turn.

Every mutation appends a change-log entry. Persistence (saving the session's
evolved environment so it survives a restart) is delegated to a host-supplied
callback — the executor owns the LIVE state + the log; the host owns durable
storage. This keeps the executor host-agnostic.

Scope is bounded to the AVAILABLE environment: a session can only enable tools
/ skills the host already made available (via the registered providers /
skill registry), edit its prompt, and create/edit session-scoped skills. It
cannot invent arbitrary tools.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# A persistence callback: given the serialised env overlay, durably store it.
# Async; host-supplied via ``Pipeline.attach_runtime(env_persistence=...)``.
EnvPersistence = Callable[[Dict[str, Any]], Awaitable[None]]


@dataclass
class EnvChangeEntry:
    """One change-log entry for an environment mutation."""

    seq: int
    action: str          # set_prompt | append_prompt | enable_tool | disable_tool | enable_skill | disable_skill | create_skill | edit_skill | save
    target: str = ""     # tool/skill name (or "")
    detail: str = ""     # human-readable summary
    ok: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "action": self.action,
            "target": self.target,
            "detail": self.detail,
            "ok": self.ok,
        }


class PipelineEnvironment:
    """Live, session-scoped controller for self-modifying environment.

    Holds references to the running pipeline's mutable surfaces — the tool
    registry, the available tool providers, the (mutable) prompt builder, and
    the skill registry — plus an append-only change log and an optional host
    persistence callback.
    """

    def __init__(
        self,
        *,
        registry: Any,
        providers: Tuple[Any, ...] = (),
        prompt_builder: Optional[Any] = None,
        skill_registry: Optional[Any] = None,
        skill_fork_runner: Optional[Any] = None,
        persistence: Optional[EnvPersistence] = None,
    ) -> None:
        self._registry = registry
        self._providers: Tuple[Any, ...] = tuple(providers or ())
        self._prompt_builder = prompt_builder
        self._skill_registry = skill_registry
        self._skill_fork_runner = skill_fork_runner
        self._persistence = persistence
        self._log: List[EnvChangeEntry] = []
        self._seq = 0
        # Session-authored skills (create_skill/edit_skill) — kept so the
        # overlay can carry them for host persistence + restore on resume.
        self._authored_skills: Dict[str, Dict[str, Any]] = {}

    # ── late binding (pipeline updates these post-build) ──────────────
    def attach_prompt_builder(self, builder: Any) -> None:
        """Re-point at the current system prompt builder. The pipeline calls
        this when a host swaps the builder via attach_runtime/refresh_runtime
        (e.g. installs a MutablePromptBuilder) AFTER the controller was built."""
        self._prompt_builder = builder

    def attach_persistence(self, persistence: Optional[EnvPersistence]) -> None:
        """Set/replace the host persistence callback (``env_save``)."""
        self._persistence = persistence

    # ── change log ────────────────────────────────────────────────────
    def _record(self, action: str, target: str, detail: str, ok: bool = True) -> None:
        self._seq += 1
        self._log.append(
            EnvChangeEntry(seq=self._seq, action=action, target=target, detail=detail, ok=ok)
        )

    def changelog(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return change-log entries (most recent last). ``limit`` keeps the
        tail."""
        entries = self._log[-limit:] if limit else self._log
        return [e.to_dict() for e in entries]

    # ── available / active enumeration ────────────────────────────────
    def _provider_names(self) -> List[str]:
        names: set[str] = set()
        for p in self._providers:
            lister = getattr(p, "list_names", None)
            if lister is None:
                continue
            try:
                names.update(lister() or [])
            except Exception:  # noqa: BLE001
                logger.debug("env: provider list_names failed", exc_info=True)
        return sorted(names)

    def active_tools(self) -> List[str]:
        return sorted(self._registry.list_names())

    def available_tools(self) -> List[str]:
        """Tools the host makes available that are NOT currently active."""
        active = set(self._registry.list_names())
        return [n for n in self._provider_names() if n not in active]

    def active_skills(self) -> List[str]:
        """Skill ids currently surfaced as tools (a SkillTool's name == its
        skill id)."""
        if self._skill_registry is None:
            return []
        ids = set(self._skill_registry.list_ids())
        return sorted(n for n in self._registry.list_names() if n in ids)

    def available_skills(self) -> List[str]:
        if self._skill_registry is None:
            return []
        active = set(self.active_skills())
        return [s for s in self._skill_registry.list_ids() if s not in active]

    # ── view ──────────────────────────────────────────────────────────
    def snapshot(self) -> Dict[str, Any]:
        """A compact view of the current environment (for ``env_view``)."""
        prompt_text = self.get_prompt()
        return {
            "prompt_chars": len(prompt_text),
            "prompt_editable": self._prompt_builder is not None
            and hasattr(self._prompt_builder, "set_base"),
            "active_tools": self.active_tools(),
            "available_tools": self.available_tools(),
            "active_skills": self.active_skills(),
            "available_skills": self.available_skills(),
            "changes": len(self._log),
            "persistable": self._persistence is not None,
        }

    # ── prompt ────────────────────────────────────────────────────────
    def get_prompt(self) -> str:
        b = self._prompt_builder
        if b is None:
            return ""
        for attr in ("current_text", "get_text"):
            fn = getattr(b, attr, None)
            if callable(fn):
                try:
                    return str(fn())
                except Exception:  # noqa: BLE001
                    break
        # Fall back to a plain build() with no state.
        try:
            return str(b.build(None))  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            return ""

    def _require_mutable_prompt(self) -> Optional[str]:
        b = self._prompt_builder
        if b is None or not hasattr(b, "set_base"):
            return (
                "prompt is not editable in this environment (no mutable prompt "
                "builder installed)"
            )
        return None

    def set_prompt(self, text: str) -> Tuple[bool, str]:
        err = self._require_mutable_prompt()
        if err:
            self._record("set_prompt", "", err, ok=False)
            return False, err
        self._prompt_builder.set_base(text)  # type: ignore[union-attr]
        msg = f"system prompt replaced ({len(text)} chars)"
        self._record("set_prompt", "", msg)
        return True, msg

    def append_prompt(self, text: str) -> Tuple[bool, str]:
        err = self._require_mutable_prompt()
        if err:
            self._record("append_prompt", "", err, ok=False)
            return False, err
        self._prompt_builder.append_section(text)  # type: ignore[union-attr]
        msg = f"appended a prompt section ({len(text)} chars)"
        self._record("append_prompt", "", msg)
        return True, msg

    # ── tools ─────────────────────────────────────────────────────────
    def enable_tool(self, name: str) -> Tuple[bool, str]:
        if self._registry.get(name) is not None:
            return True, f"tool '{name}' is already active"
        for p in self._providers:
            getter = getattr(p, "get", None)
            tool = getter(name) if getter else None
            if tool is not None:
                self._registry.register(tool)
                msg = f"enabled tool '{name}'"
                self._record("enable_tool", name, msg)
                return True, msg
        msg = f"tool '{name}' is not in the available set"
        self._record("enable_tool", name, msg, ok=False)
        return False, msg

    def disable_tool(self, name: str) -> Tuple[bool, str]:
        if self._registry.get(name) is None:
            msg = f"tool '{name}' is not active"
            self._record("disable_tool", name, msg, ok=False)
            return False, msg
        # Guard the self-modification tools so a session can't strand itself.
        if name.startswith("env_"):
            msg = f"refusing to disable the environment control tool '{name}'"
            self._record("disable_tool", name, msg, ok=False)
            return False, msg
        self._registry.unregister(name)
        msg = f"disabled tool '{name}'"
        self._record("disable_tool", name, msg)
        return True, msg

    # ── skills (enable/disable existing) ──────────────────────────────
    def enable_skill(self, skill_id: str) -> Tuple[bool, str]:
        if self._skill_registry is None:
            return False, "no skill registry is available"
        if self._registry.get(skill_id) is not None:
            return True, f"skill '{skill_id}' is already active"
        skill = self._skill_registry.get(skill_id)
        if skill is None:
            msg = f"skill '{skill_id}' is not in the available set"
            self._record("enable_skill", skill_id, msg, ok=False)
            return False, msg
        from geny_executor.skills.skill_tool import SkillTool

        self._registry.register(SkillTool(skill, fork_runner=self._skill_fork_runner))
        msg = f"enabled skill '{skill_id}'"
        self._record("enable_skill", skill_id, msg)
        return True, msg

    def disable_skill(self, skill_id: str) -> Tuple[bool, str]:
        if self._registry.get(skill_id) is None:
            msg = f"skill '{skill_id}' is not active"
            self._record("disable_skill", skill_id, msg, ok=False)
            return False, msg
        self._registry.unregister(skill_id)
        msg = f"disabled skill '{skill_id}'"
        self._record("disable_skill", skill_id, msg)
        return True, msg

    # ── skill authoring (create / edit session-scoped skills) ─────────
    def create_skill(
        self,
        skill_id: str,
        description: str,
        body: str,
        *,
        allowed_tools: Any = (),
        execution_mode: str = "inline",
        enable: bool = True,
    ) -> Tuple[bool, str]:
        """Author a new session-scoped skill and (by default) activate it.

        The skill lives in this session's skill registry (in-memory); the
        overlay carries its definition so the host can persist + restore it.
        """
        if self._skill_registry is None:
            return False, "no skill registry is available"
        sid = str(skill_id or "").strip()
        if not sid:
            self._record("create_skill", "", "skill_id is required", ok=False)
            return False, "skill_id is required"
        if self._skill_registry.get(sid) is not None:
            msg = f"skill '{sid}' already exists — use edit_skill"
            self._record("create_skill", sid, msg, ok=False)
            return False, msg
        from geny_executor.skills.types import Skill, SkillMetadata

        tools = tuple(str(t) for t in (allowed_tools or ()))
        meta = SkillMetadata(
            name=sid,
            description=str(description or sid),
            allowed_tools=tools,
            execution_mode=str(execution_mode or "inline"),
        )
        skill = Skill(id=sid, metadata=meta, body=str(body or ""), source=None)
        self._skill_registry.register(skill)
        self._authored_skills[sid] = {
            "id": sid,
            "description": meta.description,
            "body": skill.body,
            "allowed_tools": list(tools),
            "execution_mode": meta.execution_mode,
        }
        msg = f"created skill '{sid}'"
        self._record("create_skill", sid, msg)
        if enable:
            self.enable_skill(sid)
            msg += " and enabled it"
        return True, msg

    def edit_skill(
        self,
        skill_id: str,
        *,
        description: Optional[str] = None,
        body: Optional[str] = None,
        allowed_tools: Any = None,
    ) -> Tuple[bool, str]:
        """Edit an existing skill's description / body / allowed_tools. If the
        skill is active, its surfaced tool is refreshed to the new body."""
        if self._skill_registry is None:
            return False, "no skill registry is available"
        sid = str(skill_id or "").strip()
        skill = self._skill_registry.get(sid)
        if skill is None:
            msg = f"skill '{sid}' not found — use create_skill"
            self._record("edit_skill", sid, msg, ok=False)
            return False, msg
        import dataclasses

        meta = skill.metadata
        meta_changes: Dict[str, Any] = {}
        if description is not None:
            meta_changes["description"] = str(description)
        if allowed_tools is not None:
            meta_changes["allowed_tools"] = tuple(str(t) for t in allowed_tools)
        new_meta = dataclasses.replace(meta, **meta_changes) if meta_changes else meta
        new_body = str(body) if body is not None else skill.body
        new_skill = dataclasses.replace(skill, metadata=new_meta, body=new_body)

        self._skill_registry.unregister(sid)
        self._skill_registry.register(new_skill)
        self._authored_skills[sid] = {
            "id": sid,
            "description": new_meta.description,
            "body": new_body,
            "allowed_tools": list(new_meta.allowed_tools),
            "execution_mode": new_meta.execution_mode,
        }
        # If currently active, re-surface so the new body/tools take effect.
        if self._registry.get(sid) is not None:
            from geny_executor.skills.skill_tool import SkillTool

            self._registry.unregister(sid)
            self._registry.register(
                SkillTool(new_skill, fork_runner=self._skill_fork_runner)
            )
        msg = f"edited skill '{sid}'"
        self._record("edit_skill", sid, msg)
        return True, msg

    # ── persistence (save the evolved env overlay) ────────────────────
    def overlay(self) -> Dict[str, Any]:
        """Serialise the session-scoped environment overlay (what changed)
        for the host to persist + restore."""
        return {
            "prompt": self.get_prompt(),
            "active_tools": self.active_tools(),
            "active_skills": self.active_skills(),
            "authored_skills": list(self._authored_skills.values()),
            "changelog": self.changelog(),
        }

    async def save(self) -> Tuple[bool, str]:
        if self._persistence is None:
            msg = "this environment has no persistence configured (changes are live-only for this session)"
            self._record("save", "", msg, ok=False)
            return False, msg
        try:
            await self._persistence(self.overlay())
        except Exception as exc:  # noqa: BLE001
            logger.warning("env: persistence callback failed: %s", exc, exc_info=True)
            msg = f"save failed: {exc}"
            self._record("save", "", msg, ok=False)
            return False, msg
        msg = "environment saved for this session"
        self._record("save", "", msg)
        return True, msg
