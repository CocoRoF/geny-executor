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

# A pack-persistence callback: given {name, description, tools[spec], skills[spec],
# sandbox}, snapshot the sandbox + durably store it as a reusable Sandbox Tool
# Pack, returning a result dict (e.g. {"pack_id": ...}). Async; host-supplied via
# ``Pipeline.attach_runtime(pack_persistence=...)``. The executor gathers what to
# save (forged tools + authored skills + the live sandbox) and delegates the
# host-specific storage (snapshot + DB row) to this callback.
PackPersistence = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


# ── editable env-config surface ────────────────────────────────────────
# Model generation knobs that ``PipelineConfig.apply_to_state`` stamps onto
# state EVERY turn — so editing them on the live config takes effect next turn.
# The MODEL ID and PROVIDER are deliberately absent: those are "core" and stay
# locked (a session must not silently switch which model it is).
_TUNABLE_MODEL_KEYS: Dict[str, type] = {
    "temperature": float,
    "max_tokens": int,
    "top_p": float,
    "top_k": int,
    "thinking_enabled": bool,
    "thinking_budget_tokens": int,
}
# Top-level PipelineConfig limits re-applied to state every run.
_TUNABLE_PIPELINE_KEYS: Dict[str, type] = {
    "max_iterations": int,
    "cost_budget_usd": float,
    "context_window_budget": int,
    "single_turn": bool,
}
# Never editable at runtime — identity / credentials / wiring.
_CORE_LOCKED_KEYS = frozenset(
    {"model", "provider", "api_key", "base_url", "name", "credentials"}
)
# extras keys that are runtime HANDLES (objects), not editable settings. The
# value-is-a-dict heuristic already excludes most; this names the known ones
# defensively so they never surface as "settings" even if dict-shaped.
_RESERVED_EXTRAS_KEYS = frozenset(
    {
        "workspace_stack", "task_registry", "task_runner", "cron_store",
        "cron_runner", "agent_orchestrator", "subagent_manager", "mcp_manager",
        "mcp_config", "notification_endpoints", "env_extras",
    }
)
_SECRET_NAME_HINTS = ("key", "token", "secret", "password", "passwd", "credential")


def _coerce(value: Any, typ: type) -> Any:
    """Best-effort coerce a JSON value to the config field's type."""
    if typ is bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    if typ is int:
        return int(float(value))
    if typ is float:
        return float(value)
    return value


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
        pack_persistence: Optional[PackPersistence] = None,
        tool_context: Optional[Any] = None,
        config: Optional[Any] = None,
        settings_schemas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._registry = registry
        self._providers: Tuple[Any, ...] = tuple(providers or ())
        self._prompt_builder = prompt_builder
        self._skill_registry = skill_registry
        self._skill_fork_runner = skill_fork_runner
        self._persistence = persistence
        # Live tool-dispatch context — its ``.extras`` carries per-tool
        # settings (e.g. ``extras["web_search"]["brave_api_key"]``). Editing
        # this dict takes effect on the next tool call (the Tool stage reads
        # ``self._context.extras`` live per dispatch).
        self._tool_context = tool_context
        # The running PipelineConfig — model tunables (temperature / max_tokens
        # / thinking) + pipeline limits (max_iterations) are applied to state
        # every turn, so editing them here takes effect next turn.
        self._config = config
        # Optional host-supplied descriptor of the configurable tool settings
        # (groups + fields + which fields are secret) for richer discovery +
        # accurate secret masking. The executor stays host-agnostic without it.
        self._settings_schemas: List[Dict[str, Any]] = list(settings_schemas or [])
        self._log: List[EnvChangeEntry] = []
        self._seq = 0
        # Session-authored skills (create_skill/edit_skill) — kept so the
        # overlay can carry them for host persistence + restore on resume.
        self._authored_skills: Dict[str, Dict[str, Any]] = {}
        # Tools forged this session (forge_tool) — kept so save_pack knows what
        # to persist into a reusable Sandbox Tool Pack.
        self._forged_tools: Dict[str, Any] = {}
        # Host callback that snapshots the sandbox + stores a pack (save_pack).
        self._pack_persistence: Optional[PackPersistence] = pack_persistence

    # ── late binding (pipeline updates these post-build) ──────────────
    def attach_prompt_builder(self, builder: Any) -> None:
        """Re-point at the current system prompt builder. The pipeline calls
        this when a host swaps the builder via attach_runtime/refresh_runtime
        (e.g. installs a MutablePromptBuilder) AFTER the controller was built."""
        self._prompt_builder = builder

    def attach_persistence(self, persistence: Optional[EnvPersistence]) -> None:
        """Set/replace the host persistence callback (``env_save``)."""
        self._persistence = persistence

    def attach_pack_persistence(self, pack_persistence: Optional[PackPersistence]) -> None:
        """Set/replace the host pack-persistence callback (``save_pack``)."""
        self._pack_persistence = pack_persistence

    def attach_tool_context(self, tool_context: Optional[Any]) -> None:
        """Re-point at the live tool-dispatch context (its ``.extras`` holds
        the editable per-tool settings). Called when a host swaps the context
        via attach_runtime/refresh_runtime."""
        self._tool_context = tool_context

    def attach_config(self, config: Optional[Any]) -> None:
        """Re-point at the running PipelineConfig (model tunables + limits)."""
        self._config = config

    def attach_settings_schemas(self, schemas: Optional[List[Dict[str, Any]]]) -> None:
        """Set/replace the host descriptor of configurable tool settings."""
        self._settings_schemas = list(schemas or [])

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
            "setting_groups": self._setting_groups(),
            "config": {**self.get_config().get("model", {}), **self.get_config().get("pipeline", {})},
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
        # Guard the self-modification tool so a session can't strand itself
        # (the built-in dispatcher is named "env"; older builds used "env_*").
        if name == "env" or name.startswith("env_"):
            msg = f"refusing to disable the environment control tool '{name}'"
            self._record("disable_tool", name, msg, ok=False)
            return False, msg
        self._registry.unregister(name)
        msg = f"disabled tool '{name}'"
        self._record("disable_tool", name, msg)
        return True, msg

    def forge_tool(
        self,
        name: str,
        description: str = "",
        entrypoint: str = "",
        *,
        runtime: str = "python3",
        input_schema: Optional[Dict[str, Any]] = None,
        argv: Any = (),
        timeout_s: float = 60.0,
        workdir: str = "/workspace",
        network_egress: bool = False,
        read_only: bool = False,
    ) -> Tuple[bool, str]:
        """Register a NEW sandboxed tool LIVE this turn.

        The tool's implementation is an authored script (``entrypoint``,
        relative to ``workdir``) that the session wrote into its sandbox; the
        tool runs it INSIDE the sandbox (stdin JSON → stdout JSON) on each call.
        Callable from the next turn. Ephemeral (this session) — to keep it, the
        host persists a snapshot + spec as a reusable Sandbox Tool Pack.
        """
        name = str(name or "").strip()
        if not name:
            return False, "a tool name is required"
        if not entrypoint:
            return False, "an entrypoint (path to the tool's script in the sandbox) is required"
        if self._registry.get(name) is not None:
            msg = f"tool '{name}' is already active — choose another name or disable it first"
            self._record("forge_tool", name, msg, ok=False)
            return False, msg
        sandbox = getattr(self._tool_context, "sandbox", None) if self._tool_context else None
        if sandbox is None:
            msg = (
                "no sandbox is attached to this session — forge_tool needs an "
                "isolated workspace to run the tool's code in"
            )
            self._record("forge_tool", name, msg, ok=False)
            return False, msg
        from geny_executor.tools.built_in.sandbox_exec_tool import SandboxExecTool

        tool = SandboxExecTool(
            name=name,
            description=str(description or name),
            input_schema=input_schema,
            entrypoint=str(entrypoint),
            runtime=str(runtime or "python3"),
            argv=argv or (),
            timeout_s=float(timeout_s),
            workdir=str(workdir or "/workspace"),
            sandbox=sandbox,
            network_egress=bool(network_egress),
            read_only=bool(read_only),
        )
        self._registry.register(tool)
        self._forged_tools[name] = tool
        msg = (
            f"forged sandboxed tool '{name}' (runs `{runtime} {entrypoint}` in the "
            f"sandbox) — callable next turn"
        )
        self._record("forge_tool", name, msg)
        return True, msg

    async def save_pack(
        self,
        name: str,
        *,
        description: str = "",
        tools: Any = None,
        skills: Any = None,
    ) -> Tuple[bool, str]:
        """Persist **[this session's sandbox + the tools you forged + the skills
        you authored]** as one reusable **Sandbox Tool Pack**.

        The host snapshots the sandbox workspace (code + artifacts) and stores
        the pack (default disabled until an owner enables it). ``tools`` / ``skills``
        optionally restrict which forged tools / authored skills go in; omit to
        include all. The pack can later be enabled per-environment and restored
        into any fresh workspace.
        """
        name = str(name or "").strip()
        if not name:
            return False, "a pack name is required"
        if self._pack_persistence is None:
            msg = "this host doesn't support saving sandbox tool packs"
            self._record("save_pack", name, msg, ok=False)
            return False, msg
        sandbox = getattr(self._tool_context, "sandbox", None) if self._tool_context else None
        if sandbox is None:
            msg = "no sandbox is attached — there's no workspace to snapshot into a pack"
            self._record("save_pack", name, msg, ok=False)
            return False, msg
        want_tools = {str(t) for t in tools} if tools else None
        tool_specs = [
            t.to_dict()
            for n, t in self._forged_tools.items()
            if (want_tools is None or n in want_tools) and self._registry.get(n) is not None
        ]
        if not tool_specs:
            msg = "no forged tools to save — use forge_tool to build at least one first"
            self._record("save_pack", name, msg, ok=False)
            return False, msg
        want_skills = {str(s) for s in skills} if skills else None
        skill_specs = [
            dict(s)
            for sid, s in self._authored_skills.items()
            if want_skills is None or sid in want_skills
        ]
        payload = {
            "name": name,
            "description": str(description or ""),
            "tools": tool_specs,
            "skills": skill_specs,
            "sandbox": sandbox,
        }
        try:
            result = await self._pack_persistence(payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("env: pack persistence callback failed: %s", exc, exc_info=True)
            msg = f"save_pack failed: {exc}"
            self._record("save_pack", name, msg, ok=False)
            return False, msg
        pid = (result or {}).get("pack_id") if isinstance(result, dict) else None
        msg = (
            f"saved Sandbox Tool Pack '{name}'"
            + (f" (id {pid})" if pid else "")
            + f" with {len(tool_specs)} tool(s) + {len(skill_specs)} skill(s); "
            "enable it for an environment to reuse it"
        )
        self._record("save_pack", name, msg)
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

    # ── tool settings (values tools need — e.g. API keys) ─────────────
    def _extras(self) -> Optional[Dict[str, Any]]:
        ctx = self._tool_context
        if ctx is None:
            return None
        ex = getattr(ctx, "extras", None)
        if ex is None:  # late-attached context with no extras dict yet
            ex = {}
            try:
                ctx.extras = ex  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                return None
        return ex if isinstance(ex, dict) else None

    def _setting_groups(self) -> List[str]:
        """extras keys that are editable setting groups (dict-shaped, not a
        reserved runtime handle, not an internal ``__key__``)."""
        ex = self._extras()
        if not ex:
            return []
        groups = []
        for k, v in ex.items():
            if k in _RESERVED_EXTRAS_KEYS or str(k).startswith("__"):
                continue
            if isinstance(v, dict):
                groups.append(k)
        return sorted(groups)

    def _schema_for(self, group: str) -> Optional[Dict[str, Any]]:
        for sch in self._settings_schemas:
            if sch.get("key") == group:
                return sch
        return None

    def _is_secret(self, group: str, field: str) -> bool:
        sch = self._schema_for(group)
        if sch:
            for f in sch.get("fields", []) or []:
                if f.get("name") == field:
                    return bool(f.get("secret") or f.get("secure"))
        low = field.lower()
        return any(h in low for h in _SECRET_NAME_HINTS)

    @staticmethod
    def _mask(value: Any) -> str:
        s = str(value)
        if not s:
            return ""
        return "•" * 6 + (s[-4:] if len(s) > 4 else "")

    def get_settings(self, *, reveal: bool = False) -> Dict[str, Any]:
        """Current per-tool settings (secrets masked unless ``reveal``), plus
        any host-declared schema so the agent can discover what's configurable."""
        out: Dict[str, Any] = {"groups": {}, "schemas": self._settings_schemas}
        ex = self._extras()
        if ex is None:
            out["note"] = "no settings context is available in this environment"
            return out
        for group in self._setting_groups():
            vals = {}
            for field, raw in (ex.get(group) or {}).items():
                if not reveal and self._is_secret(group, field):
                    vals[field] = self._mask(raw) if raw not in (None, "") else ""
                else:
                    vals[field] = raw
            out["groups"][group] = vals
        return out

    def set_setting(self, key: str, field: str, value: Any) -> Tuple[bool, str]:
        """Set a single tool-setting value (e.g. an API key). Takes effect on
        the next call of the tool that reads ``extras[key][field]``."""
        ex = self._extras()
        if ex is None:
            msg = "no settings context is available in this environment"
            self._record("set_setting", f"{key}.{field}", msg, ok=False)
            return False, msg
        key = str(key).strip()
        field = str(field).strip()
        if not key or not field:
            return False, "both 'key' and 'field' are required"
        if key in _RESERVED_EXTRAS_KEYS or key.startswith("__"):
            msg = f"'{key}' is a protected runtime handle, not an editable setting"
            self._record("set_setting", f"{key}.{field}", msg, ok=False)
            return False, msg
        group = ex.get(key)
        if not isinstance(group, dict):
            group = {}
            ex[key] = group
        group[field] = value
        shown = self._mask(value) if self._is_secret(key, field) else value
        msg = f"set {key}.{field} = {shown}"
        self._record("set_setting", f"{key}.{field}", msg)
        return True, msg

    # ── env config (model tunables + pipeline limits; core stays locked) ─
    def get_config(self) -> Dict[str, Any]:
        """Editable config knobs + their current values, plus the locked core
        (model / provider) so the agent knows what it may and may not change."""
        model_vals: Dict[str, Any] = {}
        pipe_vals: Dict[str, Any] = {}
        core: Dict[str, Any] = {}
        cfg = self._config
        if cfg is not None:
            mc = getattr(cfg, "model", None)
            if mc is not None:
                for k in _TUNABLE_MODEL_KEYS:
                    if hasattr(mc, k):
                        model_vals[k] = getattr(mc, k)
                core["model"] = getattr(mc, "model", None)
            for k in _TUNABLE_PIPELINE_KEYS:
                if hasattr(cfg, k):
                    pipe_vals[k] = getattr(cfg, k)
        return {
            "model": model_vals,
            "pipeline": pipe_vals,
            "locked": sorted(_CORE_LOCKED_KEYS),
            "core": core,
        }

    def set_config(self, key: str, value: Any) -> Tuple[bool, str]:
        """Edit one tunable config knob. Refuses core/locked keys (model,
        provider, credentials). Takes effect next turn."""
        key = str(key).strip()
        if key in _CORE_LOCKED_KEYS:
            msg = f"'{key}' is a core setting and cannot be changed at runtime"
            self._record("set_config", key, msg, ok=False)
            return False, msg
        cfg = self._config
        if cfg is None:
            msg = "no config is available in this environment"
            self._record("set_config", key, msg, ok=False)
            return False, msg
        try:
            if key in _TUNABLE_MODEL_KEYS:
                mc = getattr(cfg, "model", None)
                if mc is None:
                    return False, "no model config available"
                coerced = _coerce(value, _TUNABLE_MODEL_KEYS[key])
                setattr(mc, key, coerced)
            elif key in _TUNABLE_PIPELINE_KEYS:
                coerced = _coerce(value, _TUNABLE_PIPELINE_KEYS[key])
                setattr(cfg, key, coerced)
            else:
                msg = f"'{key}' is not an editable config knob"
                self._record("set_config", key, msg, ok=False)
                return False, msg
        except Exception as exc:  # noqa: BLE001
            msg = f"could not set {key}: {exc}"
            self._record("set_config", key, msg, ok=False)
            return False, msg
        msg = f"set config {key} = {coerced}"
        self._record("set_config", key, msg)
        return True, msg

    def _config_overrides(self) -> Dict[str, Any]:
        """The tunable config values (for the overlay) — restored on resume."""
        snap = self.get_config()
        return {"model": snap.get("model", {}), "pipeline": snap.get("pipeline", {})}

    # ── persistence (save the evolved env overlay) ────────────────────
    def overlay(self) -> Dict[str, Any]:
        """Serialise the session-scoped environment overlay (what changed)
        for the host to persist + restore."""
        return {
            "prompt": self.get_prompt(),
            "active_tools": self.active_tools(),
            "active_skills": self.active_skills(),
            "authored_skills": list(self._authored_skills.values()),
            # Real (unmasked) values so a resume restores working settings —
            # stored in the session's own scoped storage, like the manifest.
            "tool_settings": {g: dict(self._extras().get(g, {})) for g in self._setting_groups()}
            if self._extras() is not None
            else {},
            "config": self._config_overrides(),
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
