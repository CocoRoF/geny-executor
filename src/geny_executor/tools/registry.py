"""Tool registry — registration, lookup, preset management.

Core vs deferred tools (2.42.0)
-------------------------------

Every registration carries a ``core`` flag:

* **core** tools are *exposed* — their schemas ship to the LLM on every
  request (``to_api_format(exposed_only=True)`` includes them).
* **deferred** (non-core) tools are registered and dispatchable, but
  their schemas are NOT sent upfront. The LLM discovers them at runtime
  through the ``ToolSearch`` built-in, which calls :meth:`activate` —
  from the next loop iteration the activated tool's schema appears in
  ``state.tools`` (Stage 3 rebuilds on the version bump).

``register`` defaults to ``core=True`` so hosts that build registries
by hand keep the pre-2.42 behaviour (everything exposed). The manifest
build path (:meth:`Pipeline.from_manifest`) applies the real policy:
framework built-ins default to core, external / provider / MCP tools
default to deferred, and ``manifest.tools.core_overrides`` flips
individual names either way.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from geny_executor.tools.base import Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry for tools.

    Supports:
      - Registration by name (with a per-tool ``core`` flag)
      - Preset-based filtering
      - API format export (full catalogue or exposed-only)
      - Runtime activation of deferred tools (ToolSearch discovery)
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}
        # name → core flag. Core tools are always exposed to the LLM;
        # non-core ("deferred") tools only after activate().
        self._core: Dict[str, bool] = {}
        # Deferred tools promoted at runtime (ToolSearch hit or explicit
        # host call). Session-scoped: lives and dies with this registry.
        self._activated: Set[str] = set()
        # Monotonic counter bumped on every register/unregister. Stage 3 (System)
        # compares it against the version baked into the last ``state.tools``
        # snapshot; a mismatch triggers a per-turn rebuild so a tool
        # enabled/disabled mid-session (self-modifying environment) takes effect
        # on the next turn. Also catches MCP re-seeds. activate()/set_core()
        # bump it too, so a ToolSearch discovery reaches the model on the very
        # next loop iteration.
        self._version: int = 0

    @property
    def version(self) -> int:
        """Monotonic mutation counter (bumped on register/unregister)."""
        return self._version

    def register(self, tool: Tool, *, core: bool = True) -> ToolRegistry:
        """Register a tool. Chaining supported.

        ``core=True`` (default) exposes the tool's schema to the LLM on
        every request; ``core=False`` registers it as *deferred* — the
        schema stays out of ``state.tools`` until :meth:`activate`
        promotes it (normally via a ``ToolSearch`` hit).

        Registering a tool under a name that already exists overrides the
        previous entry *and* emits a warning — this is almost always a
        bug (two providers claiming the same slot). MCP-sourced tools
        get their own ``mcp__{server}__{tool}`` namespace, so collisions
        here indicate a built-in / adhoc conflict or a duplicate
        registration pass.
        """
        existing = self._tools.get(tool.name)
        if existing is not None and existing is not tool:
            logger.warning(
                "tool name collision: '%s' re-registered (old=%s, new=%s) — the new entry wins",
                tool.name,
                type(existing).__name__,
                type(tool).__name__,
            )
        self._tools[tool.name] = tool
        self._core[tool.name] = bool(core)
        # A re-registration resets runtime activation state — the new
        # entry's exposure is whatever its ``core`` flag says.
        self._activated.discard(tool.name)
        self._version += 1
        return self

    def unregister(self, name: str) -> None:
        """Remove a tool by name."""
        if self._tools.pop(name, None) is not None:
            self._core.pop(name, None)
            self._activated.discard(name)
            self._version += 1

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    # ── Core / deferred exposure ─────────────────────────────

    def is_core(self, name: str) -> bool:
        """True when *name* is registered with the core flag."""
        return bool(self._core.get(name, False))

    def set_core(self, name: str, core: bool) -> bool:
        """Flip a registered tool's core flag. Returns False for unknown names.

        Bumps :attr:`version` on an actual change so Stage 3 rebuilds
        ``state.tools`` on the next iteration.
        """
        if name not in self._tools:
            return False
        core = bool(core)
        if self._core.get(name) != core:
            self._core[name] = core
            self._version += 1
        return True

    def activate(self, name: str) -> bool:
        """Promote a deferred tool so its schema is exposed to the LLM.

        No-op (but still True) for core or already-activated tools.
        Returns False for unknown names. Bumps :attr:`version` on an
        actual promotion.
        """
        if name not in self._tools:
            return False
        if self._core.get(name, False) or name in self._activated:
            return True
        self._activated.add(name)
        self._version += 1
        return True

    def deactivate(self, name: str) -> bool:
        """Demote a runtime-activated tool back to deferred.

        Core tools cannot be deactivated this way (use
        :meth:`set_core`). Returns False for unknown names.
        """
        if name not in self._tools:
            return False
        if name in self._activated:
            self._activated.discard(name)
            self._version += 1
        return True

    def is_exposed(self, name: str) -> bool:
        """True when *name*'s schema ships to the LLM (core or activated)."""
        if name not in self._tools:
            return False
        return self._core.get(name, False) or name in self._activated

    def list_exposed(self) -> List[Tool]:
        """Tools whose schemas ship to the LLM this turn."""
        return [t for n, t in self._tools.items() if self.is_exposed(n)]

    def list_deferred(self) -> List[Tool]:
        """Registered tools NOT currently exposed (ToolSearch-discoverable)."""
        return [t for n, t in self._tools.items() if not self.is_exposed(n)]

    def core_names(self) -> List[str]:
        """Names registered with the core flag."""
        return [n for n in self._tools if self._core.get(n, False)]

    def activated_names(self) -> List[str]:
        """Deferred names promoted at runtime (insertion order not guaranteed)."""
        return [n for n in self._tools if n in self._activated]

    # ── Listing / export ─────────────────────────────────────

    def list_all(self) -> List[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def list_names(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def filter(
        self,
        include: Optional[Set[str]] = None,
        exclude: Optional[Set[str]] = None,
    ) -> List[Tool]:
        """Filter tools by name sets."""
        tools = self.list_all()
        if include is not None:
            tools = [t for t in tools if t.name in include]
        if exclude is not None:
            tools = [t for t in tools if t.name not in exclude]
        return tools

    def to_api_format(
        self,
        include: Optional[Set[str]] = None,
        exclude: Optional[Set[str]] = None,
        *,
        exposed_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Export tools in Anthropic API format.

        ``exposed_only=True`` (the Stage 3 path) restricts the export to
        core + runtime-activated tools — deferred tools stay out of the
        request payload until a ``ToolSearch`` hit promotes them. The
        default (``False``) keeps the full-catalogue behaviour for
        hosts/tests that build registries by hand.
        """
        tools = self.filter(include=include, exclude=exclude)
        if exposed_only:
            tools = [t for t in tools if self.is_exposed(t.name)]
        return [t.to_api_format() for t in tools]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
