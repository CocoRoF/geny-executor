"""External-tool contract: required vs optional + ToolResolutionReport.

Audit §3.5: ``manifest.tools.external`` entries were plain strings and
registration "warned and prayed" — a deployment could lose every
declared tool and the only evidence was a log line. 2.2.0 adds:

  - dict entries ``{"name": ..., "required": bool}`` alongside the
    back-compat plain strings (strings stay optional-with-warning);
  - required + unresolved + strict → ``ConfigError`` at build time;
  - ``pipeline.tool_resolution_report`` recording resolved /
    unresolved / shadowed / required_unresolved for every build.

Dict entries are handled at the registration site (pipeline.py) —
``ToolsSnapshot.external`` is typed ``List[str]`` but round-trips
foreign values untouched, so no schema change was needed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pytest

from geny_executor.core.environment import EnvironmentManifest, ToolsSnapshot
from geny_executor.core.pipeline import Pipeline, ToolResolutionReport
from geny_executor.llm_client.credentials import ConfigError
from geny_executor.tools.base import Tool, ToolContext, ToolResult


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

    async def execute(self, input: Dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult(content=self._name)


class _DictProvider:
    def __init__(self, tools: Dict[str, Tool]) -> None:
        self._tools = tools

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)


def _manifest(*, built_in: List[str] = (), external: List[Any] = ()) -> EnvironmentManifest:
    return EnvironmentManifest(
        stages=[],
        tools=ToolsSnapshot(
            built_in=list(built_in),
            external=list(external),
        ),
    )


# ── back-compat: string entries unchanged ───────────────────────────


def test_string_entry_resolved_registers_and_reports() -> None:
    manifest = _manifest(external=["alpha"])
    provider = _DictProvider({"alpha": _NamedTool("alpha")})
    pipeline = Pipeline.from_manifest(manifest, adhoc_providers=[provider])

    assert pipeline.tool_registry.get("alpha") is not None
    report = pipeline.tool_resolution_report
    assert report is not None
    assert "alpha" in report.resolved
    assert report.unresolved == []
    assert report.required_unresolved == []


def test_string_entry_unresolved_warns_and_builds(caplog) -> None:
    """Plain strings stay optional: unresolved is a warning + report
    entry, never a build failure — even in strict mode."""
    caplog.set_level(logging.WARNING, logger="geny_executor.core.pipeline")
    manifest = _manifest(external=["ghost"])
    pipeline = Pipeline.from_manifest(
        manifest, adhoc_providers=[_DictProvider({})], strict=True
    )

    assert pipeline.tool_registry.get("ghost") is None
    assert "ghost" in pipeline.tool_resolution_report.unresolved
    assert pipeline.tool_resolution_report.required_unresolved == []
    assert any("ghost" in r.message for r in caplog.records)


# ── dict entries ────────────────────────────────────────────────────


def test_dict_entry_optional_unresolved_warns_and_builds(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="geny_executor.core.pipeline")
    manifest = _manifest(external=[{"name": "ghost", "required": False}])
    pipeline = Pipeline.from_manifest(
        manifest, adhoc_providers=[_DictProvider({})], strict=True
    )

    assert "ghost" in pipeline.tool_resolution_report.unresolved
    assert pipeline.tool_resolution_report.required_unresolved == []


def test_dict_entry_required_resolved_registers() -> None:
    manifest = _manifest(external=[{"name": "alpha", "required": True}])
    provider = _DictProvider({"alpha": _NamedTool("alpha")})
    pipeline = Pipeline.from_manifest(
        manifest, adhoc_providers=[provider], strict=True
    )

    assert pipeline.tool_registry.get("alpha") is not None
    assert "alpha" in pipeline.tool_resolution_report.resolved


def test_dict_entry_required_unresolved_strict_raises() -> None:
    manifest = _manifest(external=[{"name": "must_have", "required": True}])
    with pytest.raises(ConfigError, match="must_have"):
        Pipeline.from_manifest(
            manifest, adhoc_providers=[_DictProvider({})], strict=True
        )


def test_dict_entry_required_unresolved_no_providers_strict_raises() -> None:
    """The no-providers-at-all early path must enforce the same
    contract — a missing adhoc_providers kwarg is the most common way
    to lose every tool in a deployment."""
    manifest = _manifest(external=[{"name": "must_have", "required": True}])
    with pytest.raises(ConfigError, match="must_have"):
        Pipeline.from_manifest(manifest, strict=True)


def test_dict_entry_required_unresolved_lenient_warns_and_reports(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="geny_executor.core.pipeline")
    manifest = _manifest(external=[{"name": "must_have", "required": True}])
    pipeline = Pipeline.from_manifest(
        manifest, adhoc_providers=[_DictProvider({})], strict=False
    )

    report = pipeline.tool_resolution_report
    assert "must_have" in report.unresolved
    assert "must_have" in report.required_unresolved
    assert any(
        "REQUIRED" in r.message and "must_have" in r.message for r in caplog.records
    )


def test_mixed_string_and_dict_entries() -> None:
    manifest = _manifest(
        external=["alpha", {"name": "beta", "required": True}, "ghost"]
    )
    provider = _DictProvider(
        {"alpha": _NamedTool("alpha"), "beta": _NamedTool("beta")}
    )
    pipeline = Pipeline.from_manifest(
        manifest, adhoc_providers=[provider], strict=True
    )

    report = pipeline.tool_resolution_report
    assert report.resolved == ["alpha", "beta"]
    assert report.unresolved == ["ghost"]
    assert report.required_unresolved == []


def test_malformed_entries_warn_and_skip(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="geny_executor.core.pipeline")
    manifest = _manifest(external=[{"required": True}, 42])
    pipeline = Pipeline.from_manifest(
        manifest, adhoc_providers=[_DictProvider({})], strict=True
    )

    # Nameless / non-mapping entries cannot be required — they warn
    # and vanish rather than failing a build over a value we cannot
    # even attribute to a tool.
    assert pipeline.tool_resolution_report.unresolved == []
    assert len([r for r in caplog.records if "skipping" in r.message]) == 2


# ── report contents across built-in + external ──────────────────────


def test_report_covers_built_ins() -> None:
    manifest = _manifest(built_in=["Write", "Nonexistent"])
    pipeline = Pipeline.from_manifest(manifest)

    report = pipeline.tool_resolution_report
    assert "Write" in report.resolved
    assert "Nonexistent" in report.unresolved


def test_external_shadowing_built_in_is_reported() -> None:
    manifest = _manifest(built_in=["Write"], external=["Write"])
    provider = _DictProvider({"Write": _NamedTool("Write")})
    pipeline = Pipeline.from_manifest(manifest, adhoc_providers=[provider])

    report = pipeline.tool_resolution_report
    # Built-in registered, then the external override displaced it:
    # the name is live (resolved twice) AND flagged as shadowed.
    assert report.resolved.count("Write") == 2
    assert "Write" in report.shadowed
    # The live tool is the external one (last-write-wins).
    assert pipeline.tool_registry.get("Write").description == "external:Write"


def test_hand_constructed_pipeline_has_no_report() -> None:
    assert Pipeline().tool_resolution_report is None


def test_report_dataclass_defaults_empty() -> None:
    report = ToolResolutionReport()
    assert report.resolved == []
    assert report.unresolved == []
    assert report.shadowed == []
    assert report.required_unresolved == []
