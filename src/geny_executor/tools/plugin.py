"""Tool-plugin discovery via the ``geny_executor.tools`` entry-point group.

External / host packages register custom :class:`~geny_executor.tools.base.Tool`
implementations declaratively, without editing executor code, by publishing
``[project.entry-points."geny_executor.tools"]`` entries. This mirrors the
preset discovery system in :mod:`geny_executor.core.presets`.

The discovery is **opt-in and non-breaking**: nothing here is invoked
automatically when a session starts. A host calls :meth:`ToolPluginRegistry.
discover` (or the module-level :func:`discover_tool_plugins`) and then
:meth:`ToolPluginRegistry.register_into` / :func:`register_tool_plugins` to
fold discovered tools into a :class:`~geny_executor.tools.registry.ToolRegistry`.

Each entry-point may resolve to any of:

* a :class:`Tool` subclass (instantiated with a no-arg constructor, exactly
  like ``BUILT_IN_TOOL_CLASSES`` entries),
* a list / tuple of :class:`Tool` subclasses,
* a zero-arg callable factory returning a :class:`Tool`, a list of
  :class:`Tool` instances, or a list of subclasses,
* a dict like ``{"tools": [...], "description": str}`` where ``tools`` holds
  any mix of subclasses / instances.

A broken plugin (import error, bad factory, etc.) is logged and skipped so a
single faulty package can never take down the host process.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import RLock
from typing import Any, List

from geny_executor.tools.base import Tool
from geny_executor.tools.built_in import BUILT_IN_TOOL_CLASSES
from geny_executor.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

#: Entry-point group scanned for third-party tools. A package publishes
#: ``[project.entry-points."geny_executor.tools"]`` entries where each value
#: resolves to a ``Tool`` subclass / list / factory / dict (see module docs).
TOOL_ENTRY_POINT_GROUP = "geny_executor.tools"


@dataclass
class _ToolPluginRecord:
    """Internal record bound to a discovered plugin tool instance."""

    name: str
    tool: Tool
    source: str  # entry-point name the tool was discovered under
    description: str = ""


def _coerce_to_tools(target: Any, ep_name: str) -> List[Tool]:
    """Normalize a loaded entry-point target into a list of ``Tool`` instances.

    Accepts a ``Tool`` subclass, a ``Tool`` instance, a list/tuple of either,
    a zero-arg factory callable, or a ``{"tools": [...], ...}`` dict. Unknown
    shapes yield an empty list (caller logs + skips).
    """
    # (d) dict like {"tools": [...], "description": str}
    if isinstance(target, dict):
        raw = target.get("tools", [])
        items = list(raw) if isinstance(raw, (list, tuple)) else [raw]
        return _flatten_tool_items(items, ep_name)

    # (c) zero-arg callable factory (but NOT a Tool subclass — those are types
    # and handled below). Resolve it, then recurse on its result.
    if callable(target) and not (isinstance(target, type) and issubclass(target, Tool)):
        produced = target()
        return _coerce_to_tools(produced, ep_name)

    # (b) list / tuple of subclasses or instances
    if isinstance(target, (list, tuple)):
        return _flatten_tool_items(list(target), ep_name)

    # (a) single Tool subclass or instance
    return _flatten_tool_items([target], ep_name)


def _flatten_tool_items(items: List[Any], ep_name: str) -> List[Tool]:
    """Turn a flat list of subclasses / instances into ``Tool`` instances."""
    tools: List[Tool] = []
    for item in items:
        if isinstance(item, type) and issubclass(item, Tool):
            tools.append(item())  # no-arg constructor, like BUILT_IN_TOOL_CLASSES
        elif isinstance(item, Tool):
            tools.append(item)
        else:
            logger.warning(
                "Tool entry-point %s yielded a non-Tool item %r — skipping",
                ep_name,
                type(item).__name__,
            )
    return tools


class ToolPluginRegistry:
    """Global registry for ``Tool`` plugins contributed via entry-points.

    Thread-safe and cached: :meth:`discover` only scans once until
    :meth:`clear` is called or ``force=True`` is passed, mirroring
    :class:`geny_executor.core.presets.PresetRegistry`.
    """

    _lock = RLock()
    _records: List[_ToolPluginRecord] = []
    _discovered: bool = False

    @classmethod
    def list(cls) -> List[_ToolPluginRecord]:
        with cls._lock:
            return list(cls._records)

    @classmethod
    def tools(cls) -> List[Tool]:
        """Return the discovered ``Tool`` instances."""
        with cls._lock:
            return [rec.tool for rec in cls._records]

    @classmethod
    def clear(cls) -> None:
        """Drop all discovered records and reset the cache (useful in tests)."""
        with cls._lock:
            cls._records = []
            cls._discovered = False

    @classmethod
    def discover(cls, *, force: bool = False) -> int:
        """Scan ``geny_executor.tools`` entry-points and cache plugin tools.

        Safe to call multiple times — results are cached until :meth:`clear`
        or ``force=True``. Returns the number of ``Tool`` instances collected
        in this call. Import / load / instantiation failures are logged and
        skipped rather than raised, so a broken plugin cannot take down the
        host process.
        """
        with cls._lock:
            if cls._discovered and not force:
                return 0
            cls._discovered = True
            cls._records = []

        try:
            from importlib.metadata import entry_points
        except ImportError:  # pragma: no cover — Python 3.8 fallback
            return 0

        try:
            eps = entry_points()
            group_eps = (
                eps.select(group=TOOL_ENTRY_POINT_GROUP)
                if hasattr(eps, "select")
                else eps.get(TOOL_ENTRY_POINT_GROUP, [])  # type: ignore[arg-type]  # pre-3.10 dict API
            )
        except Exception as exc:  # pragma: no cover — metadata backend variance
            logger.warning("Failed to enumerate tool entry-points: %s", exc)
            return 0

        collected: List[_ToolPluginRecord] = []
        for ep in group_eps:
            try:
                target: Any = ep.load()
            except Exception as exc:
                logger.warning("Failed to load tool entry-point %s: %s", ep.name, exc)
                continue

            description = ""
            if isinstance(target, dict):
                description = str(target.get("description", ""))

            try:
                tools = _coerce_to_tools(target, ep.name)
            except Exception as exc:
                logger.warning("Failed to build tools from entry-point %s: %s", ep.name, exc)
                continue

            if not tools:
                logger.warning("Tool entry-point %s did not yield any Tool instances", ep.name)
                continue

            for tool in tools:
                collected.append(
                    _ToolPluginRecord(
                        name=tool.name,
                        tool=tool,
                        source=ep.name,
                        description=description,
                    )
                )

        with cls._lock:
            cls._records = collected

        return len(collected)

    @classmethod
    def register_into(cls, registry: ToolRegistry, *, force: bool = False) -> List[str]:
        """Register all discovered plugin tools into *registry*.

        Calls :meth:`discover` (respecting the cache unless ``force=True``)
        first, then registers each discovered tool. A discovered tool whose
        name collides with an already-registered tool *or* with a built-in
        tool name is skipped and logged — plugins never overwrite a built-in.

        Returns the list of tool names actually registered.
        """
        cls.discover(force=force)

        registered: List[str] = []
        for rec in cls.list():
            name = rec.name
            if name in BUILT_IN_TOOL_CLASSES:
                logger.warning(
                    "Tool plugin %r (from entry-point %s) collides with a "
                    "built-in tool name — skipping",
                    name,
                    rec.source,
                )
                continue
            if name in registry:
                logger.warning(
                    "Tool plugin %r (from entry-point %s) collides with an "
                    "already-registered tool — skipping",
                    name,
                    rec.source,
                )
                continue
            registry.register(rec.tool)
            registered.append(name)

        return registered


def discover_tool_plugins(*, force: bool = False) -> List[Tool]:
    """Discover plugin tools and return the collected ``Tool`` instances.

    Thin functional wrapper over :meth:`ToolPluginRegistry.discover`.
    """
    ToolPluginRegistry.discover(force=force)
    return ToolPluginRegistry.tools()


def register_tool_plugins(registry: ToolRegistry, *, force: bool = False) -> List[str]:
    """Discover plugin tools and register them into *registry*.

    Thin functional wrapper over :meth:`ToolPluginRegistry.register_into`.
    Returns the names actually registered (collisions skipped).
    """
    return ToolPluginRegistry.register_into(registry, force=force)
