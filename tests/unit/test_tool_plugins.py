"""Tests for tool-plugin entry-point discovery (geny_executor.tools group).

These mirror the preset-discovery style: fake entry-points are injected by
monkeypatching ``importlib.metadata.entry_points`` (the symbol imported lazily
inside ``ToolPluginRegistry.discover``). Each fake entry-point exercises one of
the supported target shapes plus failure / collision handling.
"""

from __future__ import annotations

import importlib.metadata
from typing import Any, Dict, List

import pytest

from geny_executor.tools import (
    TOOL_ENTRY_POINT_GROUP,
    ToolPluginRegistry,
    ToolRegistry,
    discover_tool_plugins,
    register_tool_plugins,
)
from geny_executor.tools.base import Tool, ToolContext, ToolResult


# ─────────────────────────────────────────────────────────────────
# Fake tool implementations
# ─────────────────────────────────────────────────────────────────


def _make_tool_class(tool_name: str) -> type:
    class _FakeTool(Tool):
        @property
        def name(self) -> str:
            return tool_name

        @property
        def description(self) -> str:
            return f"fake tool {tool_name}"

        @property
        def input_schema(self) -> Dict[str, Any]:
            return {"type": "object", "properties": {}}

        async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
            return ToolResult(content=f"ran {tool_name}")

    _FakeTool.__name__ = f"FakeTool_{tool_name}"
    return _FakeTool


AlphaTool = _make_tool_class("alpha")
BetaTool = _make_tool_class("beta")
GammaTool = _make_tool_class("gamma")
DeltaTool = _make_tool_class("delta")
EpsilonTool = _make_tool_class("epsilon")
# Collides with a built-in name → must be skipped on register.
ReadCollisionTool = _make_tool_class("Read")


def _gamma_factory() -> Tool:
    """Zero-arg factory returning a single Tool instance."""
    return GammaTool()


def _delta_epsilon_factory() -> List[Tool]:
    """Zero-arg factory returning a list of Tool instances."""
    return [DeltaTool(), EpsilonTool()]


# ─────────────────────────────────────────────────────────────────
# Fake entry-point plumbing
# ─────────────────────────────────────────────────────────────────


class _FakeEntryPoint:
    def __init__(self, name: str, loader: Any, *, raises: bool = False) -> None:
        self.name = name
        self.group = TOOL_ENTRY_POINT_GROUP
        self._loader = loader
        self._raises = raises

    def load(self) -> Any:
        if self._raises:
            raise ImportError(f"boom loading {self.name}")
        return self._loader


class _FakeEntryPoints:
    """Mimics the 3.10+ ``EntryPoints`` object with a ``.select`` method."""

    def __init__(self, eps: List[_FakeEntryPoint]) -> None:
        self._eps = eps

    def select(self, *, group: str) -> List[_FakeEntryPoint]:
        return [ep for ep in self._eps if ep.group == group]


def _install_fake_entry_points(monkeypatch, eps: List[_FakeEntryPoint]) -> None:
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda: _FakeEntryPoints(eps),
    )


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset the global plugin registry around every test."""
    ToolPluginRegistry.clear()
    yield
    ToolPluginRegistry.clear()


# ─────────────────────────────────────────────────────────────────
# Tests — shapes
# ─────────────────────────────────────────────────────────────────


def test_discover_tool_subclass_entry_point(monkeypatch):
    """(a) entry-point loads to a single Tool subclass → instantiated."""
    _install_fake_entry_points(monkeypatch, [_FakeEntryPoint("alpha-ep", AlphaTool)])

    count = ToolPluginRegistry.discover(force=True)
    assert count == 1

    tools = ToolPluginRegistry.tools()
    assert [t.name for t in tools] == ["alpha"]
    assert isinstance(tools[0], Tool)


def test_discover_list_entry_point(monkeypatch):
    """(b) entry-point loads to a list/tuple of Tool subclasses."""
    _install_fake_entry_points(
        monkeypatch,
        [_FakeEntryPoint("multi-ep", [AlphaTool, BetaTool])],
    )

    count = ToolPluginRegistry.discover(force=True)
    assert count == 2
    assert sorted(t.name for t in ToolPluginRegistry.tools()) == ["alpha", "beta"]


def test_discover_factory_entry_point(monkeypatch):
    """(c) zero-arg factory callable returning a Tool and a list of Tools."""
    _install_fake_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint("gamma-ep", _gamma_factory),
            _FakeEntryPoint("delta-eps-ep", _delta_epsilon_factory),
        ],
    )

    count = ToolPluginRegistry.discover(force=True)
    assert count == 3
    assert sorted(t.name for t in ToolPluginRegistry.tools()) == [
        "delta",
        "epsilon",
        "gamma",
    ]


def test_discover_dict_entry_point(monkeypatch):
    """(d) dict shape {"tools": [...], "description": str}."""
    payload = {
        "tools": [AlphaTool, BetaTool()],  # mix of subclass + instance
        "description": "bundle of fakes",
    }
    _install_fake_entry_points(monkeypatch, [_FakeEntryPoint("bundle-ep", payload)])

    count = ToolPluginRegistry.discover(force=True)
    assert count == 2

    records = ToolPluginRegistry.list()
    assert all(rec.description == "bundle of fakes" for rec in records)
    assert sorted(rec.name for rec in records) == ["alpha", "beta"]


# ─────────────────────────────────────────────────────────────────
# Tests — resilience
# ─────────────────────────────────────────────────────────────────


def test_broken_entry_point_is_skipped(monkeypatch):
    """A load-raising entry-point is skipped; siblings still register."""
    _install_fake_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint("broken-ep", None, raises=True),
            _FakeEntryPoint("good-ep", AlphaTool),
        ],
    )

    count = ToolPluginRegistry.discover(force=True)
    assert count == 1
    assert [t.name for t in ToolPluginRegistry.tools()] == ["alpha"]


def test_discover_caches_until_force(monkeypatch):
    """Second discover() without force is a no-op (cache hit)."""
    _install_fake_entry_points(monkeypatch, [_FakeEntryPoint("alpha-ep", AlphaTool)])

    assert ToolPluginRegistry.discover(force=True) == 1
    # Cached — returns 0 added, records unchanged.
    assert ToolPluginRegistry.discover() == 0
    assert len(ToolPluginRegistry.tools()) == 1


# ─────────────────────────────────────────────────────────────────
# Tests — registration / collisions
# ─────────────────────────────────────────────────────────────────


def test_register_into_skips_builtin_collision(monkeypatch):
    """A plugin tool named like a built-in must NOT be registered."""
    _install_fake_entry_points(
        monkeypatch,
        [
            _FakeEntryPoint("read-collision-ep", ReadCollisionTool),
            _FakeEntryPoint("alpha-ep", AlphaTool),
        ],
    )

    registry = ToolRegistry()
    registered = ToolPluginRegistry.register_into(registry, force=True)

    assert registered == ["alpha"]
    assert "alpha" in registry
    assert "Read" not in registry  # built-in slot protected


def test_register_into_skips_existing_collision(monkeypatch):
    """A plugin tool colliding with an already-registered tool is skipped."""
    _install_fake_entry_points(monkeypatch, [_FakeEntryPoint("alpha-ep", AlphaTool)])

    registry = ToolRegistry()
    # Pre-register a different object under the same name.
    pre_existing = AlphaTool()
    registry.register(pre_existing)

    registered = ToolPluginRegistry.register_into(registry, force=True)

    assert registered == []  # collision → skipped
    assert registry.get("alpha") is pre_existing  # untouched


def test_module_level_helpers(monkeypatch):
    """The functional wrappers behave like the classmethods."""
    _install_fake_entry_points(monkeypatch, [_FakeEntryPoint("alpha-ep", AlphaTool)])

    tools = discover_tool_plugins(force=True)
    assert [t.name for t in tools] == ["alpha"]

    registry = ToolRegistry()
    registered = register_tool_plugins(registry, force=True)
    assert registered == ["alpha"]
    assert "alpha" in registry
