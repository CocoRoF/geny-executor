"""Hook configuration types + YAML loader.

Cycle 20260424 executor uplift — Phase 5 Week 9.

A hook is a subprocess the executor invokes when a registered event
fires. The subprocess receives the event payload as JSON on stdin and
may return a JSON ``HookOutcome`` on stdout to influence the in-flight
operation (block a tool, modify its input, suppress its output).

This module defines the in-memory configuration shape and the YAML
loader. The runner that actually spawns subprocesses lives in
:mod:`~geny_executor.hooks.runner`.

Config shape (canonical YAML form, e.g. ``.geny/hooks.yaml``)::

    enabled: true
    hooks:
      pre_tool_use:
        - command: /usr/local/bin/audit-hook
          args: ["--session", "${session_id}"]
          timeout_ms: 5000
          match:
            tool: Bash
      post_tool_use:
        - command: ./scripts/log-hook.sh
          working_dir: /home/me/scripts
          env:
            HOOK_LEVEL: info

Defaults — important to preserve:

* ``enabled`` defaults to ``False``. Hooks are an opt-in security
  surface; a host that loads a config but doesn't flip the switch
  gets pure pass-through.
* Per-entry ``timeout_ms`` defaults to 5000 ms. Anything slower
  blocks the agent's progress, so we cap aggressively. Hosts that
  need long-running side effects should fire-and-forget from inside
  the hook script (write to a queue, return immediately).
* Per-entry ``match`` defaults to ``{}`` (matches every event). The
  matcher language is intentionally narrow today: ``{"tool": "<name>"}``
  filters tool events to a specific tool name. Future expansions
  (regex, scope tagging) extend this without breaking older configs.

See ``executor_uplift/12_detailed_plan.md`` §5.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from geny_executor.hooks.events import HookEvent

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_MS = 5000
HOOKS_OPT_IN_ENV = "GENY_ALLOW_HOOKS"


@dataclass(frozen=True)
class HookConfigEntry:
    """One subprocess hook registration.

    Attributes:
        command: Path or executable name. Resolved via ``$PATH``.
        args: Argument list passed verbatim — no shell interpolation
            (we never spawn a shell).
        timeout_ms: Wall clock timeout for the subprocess in
            milliseconds. Exceeded → the subprocess is killed and
            the runner emits a fail-open passthrough outcome.
        match: Filter expression. Empty dict → fires for every event
            of the configured kind. Currently understood keys:
            ``tool`` (exact tool name match for tool events).
        env: Extra environment variables merged onto the child's env.
            The parent's env is inherited; entries here override.
        working_dir: Working directory for the subprocess. Falls back
            to the parent process's cwd when absent.
    """

    command: str
    args: List[str] = field(default_factory=list)
    timeout_ms: int = DEFAULT_TIMEOUT_MS
    match: Dict[str, Any] = field(default_factory=dict)
    env: Dict[str, str] = field(default_factory=dict)
    working_dir: Optional[str] = None

    def matches(self, event: HookEvent, tool_name: Optional[str]) -> bool:
        """True if this entry should fire for ``(event, tool_name)``.

        ``event`` filtering is performed at the registration level
        (``HookConfig.entries`` is keyed by event). This method only
        evaluates the per-entry ``match`` expression.
        """
        match_tool = self.match.get("tool")
        if match_tool is not None:
            if tool_name != match_tool:
                return False
        return True


@dataclass(frozen=True)
class HookConfig:
    """Top-level hook configuration.

    Attributes:
        enabled: Master switch. When ``False`` (default) the runner
          short-circuits every fire to pass-through, regardless of
          declared entries. Hosts opt in by setting this to ``True``
          OR by setting the ``GENY_ALLOW_HOOKS=1`` env var (the
          latter is checked at runner construction).
        entries: Dict mapping each :class:`HookEvent` kind to the
          list of subprocess entries that should fire for it.
        audit_log_path: Optional file path the runner appends one
          JSON line per hook invocation to. ``None`` → no audit log.
    """

    enabled: bool = False
    entries: Dict[HookEvent, List[HookConfigEntry]] = field(default_factory=dict)
    audit_log_path: Optional[str] = None

    @classmethod
    def disabled(cls) -> "HookConfig":
        """Sentinel "no hooks at all" config."""
        return cls(enabled=False, entries={})

    def entries_for(self, event: HookEvent) -> List[HookConfigEntry]:
        """Return registered entries for ``event``; empty list if none."""
        return list(self.entries.get(event, []))


def _coerce_entry(raw: Any, *, source: str) -> HookConfigEntry:
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: hook entry must be a mapping, got {type(raw).__name__}")
    command = raw.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError(f"{source}: hook entry missing required string 'command'")

    raw_args = raw.get("args", [])
    if raw_args in (None, ""):
        args: List[str] = []
    elif isinstance(raw_args, list):
        if not all(isinstance(a, str) for a in raw_args):
            raise ValueError(f"{source}: 'args' must be a list of strings")
        args = list(raw_args)
    else:
        raise ValueError(f"{source}: 'args' must be a list (got {type(raw_args).__name__})")

    raw_timeout = raw.get("timeout_ms", DEFAULT_TIMEOUT_MS)
    try:
        timeout_ms = int(raw_timeout)
    except (TypeError, ValueError):
        raise ValueError(f"{source}: 'timeout_ms' must be an integer") from None
    if timeout_ms <= 0:
        raise ValueError(f"{source}: 'timeout_ms' must be positive")

    raw_match = raw.get("match", {}) or {}
    if not isinstance(raw_match, dict):
        raise ValueError(f"{source}: 'match' must be a mapping")

    raw_env = raw.get("env", {}) or {}
    if not isinstance(raw_env, dict):
        raise ValueError(f"{source}: 'env' must be a mapping")
    env = {str(k): str(v) for k, v in raw_env.items()}

    working_dir = raw.get("working_dir")
    if working_dir is not None and not isinstance(working_dir, str):
        raise ValueError(f"{source}: 'working_dir' must be a string if provided")

    return HookConfigEntry(
        command=command.strip(),
        args=args,
        timeout_ms=timeout_ms,
        match=dict(raw_match),
        env=env,
        working_dir=working_dir,
    )


def parse_hook_config(raw: Any, *, source: str = "<inline>") -> HookConfig:
    """Validate and parse a hook config dict into a :class:`HookConfig`.

    Accepts the raw output of ``yaml.safe_load`` or any equivalent
    loader. Raises ``ValueError`` with a concrete location-suffixed
    message for any malformed entry.

    Unknown event names log a WARNING and are skipped — better to keep
    a partial config alive than to fail the whole pipeline because a
    config file mentions a hook event the executor version doesn't
    recognise (forward compat).
    """
    if raw is None:
        return HookConfig.disabled()
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: hook config root must be a mapping, got {type(raw).__name__}")

    enabled = bool(raw.get("enabled", False))

    raw_hooks = raw.get("hooks") or {}
    if not isinstance(raw_hooks, dict):
        raise ValueError(f"{source}: 'hooks' must be a mapping")

    entries: Dict[HookEvent, List[HookConfigEntry]] = {}
    valid_events = {e.value: e for e in HookEvent}
    for event_name, raw_list in raw_hooks.items():
        if not isinstance(event_name, str):
            raise ValueError(f"{source}: hook event keys must be strings")
        event = valid_events.get(event_name)
        if event is None:
            logger.warning(
                "%s: unknown hook event %r — skipping (executor may need an upgrade)",
                source,
                event_name,
            )
            continue
        if raw_list in (None, []):
            continue
        if not isinstance(raw_list, list):
            raise ValueError(f"{source}: hook event {event_name!r} value must be a list of entries")
        parsed_entries: List[HookConfigEntry] = []
        for i, raw_entry in enumerate(raw_list):
            entry = _coerce_entry(raw_entry, source=f"{source}.{event_name}[{i}]")
            parsed_entries.append(entry)
        if parsed_entries:
            entries[event] = parsed_entries

    audit_log_path = raw.get("audit_log_path")
    if audit_log_path is not None and not isinstance(audit_log_path, str):
        raise ValueError(f"{source}: 'audit_log_path' must be a string if provided")

    return HookConfig(
        enabled=enabled,
        entries=entries,
        audit_log_path=audit_log_path,
    )


def load_hooks_config(path: Path) -> HookConfig:
    """Load and parse a hook config YAML file.

    Returns :meth:`HookConfig.disabled` when the file does not exist —
    a missing config is the same as "no hooks", not an error.

    Raises:
        ValueError: For malformed entries (propagated from
            :func:`parse_hook_config`).
        OSError: For unreadable files (permission, broken symlink,
            etc). Hosts can decide whether to swallow or surface.
    """
    if not path.exists():
        return HookConfig.disabled()
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return HookConfig.disabled()
    raw = yaml.safe_load(text)
    return parse_hook_config(raw, source=str(path))


def hooks_opt_in_from_env(env: Optional[Dict[str, str]] = None) -> bool:
    """Return True when the host has opted into running hooks.

    Reads ``GENY_ALLOW_HOOKS`` from the supplied env mapping (defaults
    to ``os.environ``). Truthy values: ``1``, ``true``, ``yes``, ``on``
    (case-insensitive). Anything else — including unset — is False.

    The opt-in env var is the second of two switches: the runner only
    fires hooks when **both** the env opt-in and ``HookConfig.enabled``
    are true. Belt-and-braces because a misconfigured config that
    enables hooks would otherwise be a security regression for hosts
    that hadn't intended to run subprocesses.
    """
    if env is None:
        env = dict(os.environ)
    raw = env.get(HOOKS_OPT_IN_ENV, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}
