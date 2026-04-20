"""Auto-registration of framework-shipped built-in tools.

Cycle 20260420_7 / PR-2: ``Pipeline.from_manifest_async`` (and
``from_manifest``) now resolve ``manifest.tools.built_in`` names
against :data:`geny_executor.tools.built_in.BUILT_IN_TOOL_CLASSES`,
so that consumers can opt into the framework's shipped Read / Write /
Edit / Bash / Glob / Grep without reimplementing them.

The matrix:

- ``built_in=["*"]`` → all classes in the map register.
- ``built_in=["Write"]`` → only the named class registers.
- ``built_in=[]`` / missing → no framework tools (preserves the old
  "external only" behaviour).
- ``built_in=["Unknown"]`` → warn, skip, no crash.
- ``built_in=["Write"]`` + external provider supplying "Write" →
  the external tool wins. Built-ins register first so consumers
  get a working default, but a host that wants to harden or replace
  a framework tool (e.g. a sandboxed Bash variant) can shadow it
  via ``AdhocToolProvider``.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from geny_executor.core.environment import EnvironmentManifest, ToolsSnapshot
from geny_executor.core.pipeline import Pipeline
from geny_executor.tools.base import Tool, ToolContext, ToolResult
from geny_executor.tools.built_in import BUILT_IN_TOOL_CLASSES


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────


class _NamedTool(Tool):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"external:{self._name}"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(
        self, input: Dict[str, Any], context: ToolContext
    ) -> ToolResult:
        return ToolResult(content=self._name)


class _DictProvider:
    def __init__(self, tools: Dict[str, Tool]) -> None:
        self._tools = tools

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)


def _manifest(*, built_in: List[str] = (), external: List[str] = ()) -> EnvironmentManifest:
    return EnvironmentManifest(
        stages=[],
        tools=ToolsSnapshot(
            built_in=list(built_in),
            external=list(external),
        ),
    )


# ─────────────────────────────────────────────────────────────────
# Map shape
# ─────────────────────────────────────────────────────────────────


def test_built_in_tool_classes_map_covers_all_exports() -> None:
    """Every class exported from ``built_in`` should appear in the map.
    Prevents drift between ``__all__`` and the auto-register source."""
    from geny_executor.tools import built_in as built_in_mod

    for cls_name in (
        "ReadTool",
        "WriteTool",
        "EditTool",
        "BashTool",
        "GlobTool",
        "GrepTool",
    ):
        assert hasattr(built_in_mod, cls_name)

    registered_classes = set(BUILT_IN_TOOL_CLASSES.values())
    assert len(registered_classes) == len(BUILT_IN_TOOL_CLASSES), (
        "BUILT_IN_TOOL_CLASSES must map each name to a distinct class"
    )


# ─────────────────────────────────────────────────────────────────
# Pipeline.from_manifest — auto-registration
# ─────────────────────────────────────────────────────────────────


def test_star_registers_every_built_in() -> None:
    manifest = _manifest(built_in=["*"])
    pipeline = Pipeline.from_manifest(manifest)
    registry = pipeline.tool_registry

    for name in BUILT_IN_TOOL_CLASSES:
        tool = registry.get(name)
        assert tool is not None, f"'{name}' missing after '*' expansion"
        # Instance of the correct class (not a stub)
        assert isinstance(tool, BUILT_IN_TOOL_CLASSES[name])


def test_named_list_registers_only_listed() -> None:
    manifest = _manifest(built_in=["Write", "Read"])
    pipeline = Pipeline.from_manifest(manifest)
    registry = pipeline.tool_registry

    assert registry.get("Write") is not None
    assert registry.get("Read") is not None
    # Not requested → absent
    assert registry.get("Edit") is None
    assert registry.get("Bash") is None


def test_empty_list_registers_nothing() -> None:
    """Backward compatibility: manifests authored before this feature
    carry ``built_in=[]`` and must continue to produce an empty
    registry from the built-in side."""
    manifest = _manifest(built_in=[], external=[])
    pipeline = Pipeline.from_manifest(manifest)
    for name in BUILT_IN_TOOL_CLASSES:
        assert pipeline.tool_registry.get(name) is None


def test_unknown_name_warns_and_skips(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="geny_executor.core.pipeline")
    manifest = _manifest(built_in=["Write", "Nonexistent"])
    pipeline = Pipeline.from_manifest(manifest)

    # Valid name still registered
    assert pipeline.tool_registry.get("Write") is not None
    assert pipeline.tool_registry.get("Nonexistent") is None
    # Warning emitted
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("Nonexistent" in r.message for r in warnings), (
        f"expected a warning mentioning 'Nonexistent'; got {warnings!r}"
    )


def test_external_overrides_built_in_on_name_collision() -> None:
    """A host that wants to replace a framework tool (e.g. a
    security-hardened ``Bash``) supplies an ``AdhocToolProvider``
    declaring the same name. Built-ins register first so consumers
    always get a working default; external registrations then
    shadow by name — matching ``ToolRegistry.register``'s existing
    last-write-wins semantics. This keeps the override path a
    single-line provider tweak instead of a fork."""
    manifest = _manifest(built_in=["Write"], external=["Write"])
    provider = _DictProvider({"Write": _NamedTool("Write")})
    pipeline = Pipeline.from_manifest(manifest, adhoc_providers=[provider])

    write = pipeline.tool_registry.get("Write")
    assert write is not None
    # The external _NamedTool won the collision — the host override
    # path works. Consumers that want the built-in simply omit the
    # external "Write" entry from their manifest.
    assert not isinstance(write, BUILT_IN_TOOL_CLASSES["Write"])
    assert write.description.startswith("external:")


def test_built_in_and_external_coexist_under_different_names() -> None:
    manifest = _manifest(built_in=["Write"], external=["news_search"])
    provider = _DictProvider({"news_search": _NamedTool("news_search")})
    pipeline = Pipeline.from_manifest(manifest, adhoc_providers=[provider])

    assert isinstance(
        pipeline.tool_registry.get("Write"), BUILT_IN_TOOL_CLASSES["Write"]
    )
    assert pipeline.tool_registry.get("news_search") is not None


# ─────────────────────────────────────────────────────────────────
# Write tool actually writes — end-to-end sandbox check
# ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_registered_write_tool_writes_file_in_working_dir(tmp_path) -> None:
    """Proves the registered built-in is a functional instance, not a
    placeholder. The sandbox is ``working_dir``; ``resolve_and_validate``
    rejects paths outside it."""
    manifest = _manifest(built_in=["Write"])
    pipeline = Pipeline.from_manifest(manifest)
    write_tool = pipeline.tool_registry.get("Write")
    assert write_tool is not None

    target = tmp_path / "hello.txt"
    ctx = ToolContext(
        session_id="test-session",
        working_dir=str(tmp_path),
        allowed_paths=[str(tmp_path)],
    )
    result = await write_tool.execute(
        {"file_path": str(target), "content": "안녕 test.txt"},
        ctx,
    )
    assert result.is_error is False, result.content
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "안녕 test.txt"


@pytest.mark.asyncio
async def test_write_tool_refuses_path_outside_working_dir(tmp_path) -> None:
    manifest = _manifest(built_in=["Write"])
    pipeline = Pipeline.from_manifest(manifest)
    write_tool = pipeline.tool_registry.get("Write")

    outside = tmp_path.parent / "escape.txt"
    ctx = ToolContext(
        session_id="test-session",
        working_dir=str(tmp_path),
        allowed_paths=[str(tmp_path)],
    )
    result = await write_tool.execute(
        {"file_path": str(outside), "content": "should fail"},
        ctx,
    )
    assert result.is_error is True, (
        "Write must reject paths outside working_dir"
    )
    assert not outside.exists()
