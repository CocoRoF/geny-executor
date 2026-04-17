"""Phase 7 — Integration tests.

Tests:
  - Pipeline mutation workflows (multi-step, snapshot, rollback)
  - Environment roundtrip (save→load, export→import, variable resolution)
  - Tool lifecycle (AdhocTool creation, scope resolution)
  - History→Environment cross-module integration
  - Sandbox security enforcement
"""

import sys, os, json, tempfile, shutil, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
from geny_executor.core.snapshot import PipelineSnapshot, StageSnapshot
from geny_executor.core.environment import (
    EnvironmentManager,
    EnvironmentManifest,
    EnvironmentMetadata,
    EnvironmentResolver,
    EnvironmentSanitizer,
    EnvironmentSummary,
    ToolsSnapshot,
)
from geny_executor.core.diff import DiffEntry, EnvironmentDiff
from geny_executor.core.presets import PresetInfo, PresetManager
from geny_executor.history.service import HistoryService
from geny_executor.history.monitor import PerformanceMonitor
from geny_executor.history.cost import CostAnalyzer
from geny_executor.history.ab_test import ABTestRunner
from geny_executor.history.models import StageTimingRecord, ToolCallRecord
from geny_executor.tools.adhoc import AdhocToolDefinition, AdhocToolFactory, TemplateToolConfig
from geny_executor.tools.scope import ToolScope, ToolScopeRule


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def env_mgr(tmp_dir):
    return EnvironmentManager(os.path.join(tmp_dir, "environments"))


@pytest.fixture
def history_svc(tmp_dir):
    db = os.path.join(tmp_dir, "history.db")
    blobs = os.path.join(tmp_dir, "blobs")
    s = HistoryService(db_path=db, blob_path=blobs)
    yield s
    s.close()


def _make_snapshot(name="test-pipeline", n_stages=3, model="claude-sonnet-4-20250514"):
    """Helper to create a basic pipeline snapshot."""
    stages = []
    for i in range(1, n_stages + 1):
        stages.append(StageSnapshot(
            order=i,
            name=f"stage_{i}",
            is_active=True,
            strategies={"main": f"Strategy{i}"},
            strategy_configs={"main": {"param": i}},
            stage_config={"timeout": 30},
        ))
    return PipelineSnapshot(
        pipeline_name=name,
        stages=stages,
        pipeline_config={"max_iterations": 10},
        model_config={"model": model, "temperature": 0.7},
    )


# ═══════════════════════════════════════════════════════════
# Environment Roundtrip Integration
# ═══════════════════════════════════════════════════════════


class TestEnvironmentRoundtrip:
    """Save → Load → Export → Import → Compare workflows."""

    def test_save_and_load_preserves_data(self, env_mgr):
        snap = _make_snapshot()
        env_id = env_mgr.save(snap, name="Test Env", description="Test", tags=["test"])
        loaded = env_mgr.load(env_id)

        assert loaded.metadata.name == "Test Env"
        assert loaded.metadata.description == "Test"
        assert "test" in loaded.metadata.tags
        assert len(loaded.stages) == 3
        assert loaded.model["model"] == "claude-sonnet-4-20250514"

    def test_export_import_roundtrip(self, env_mgr):
        snap = _make_snapshot(name="export-test")
        env_id = env_mgr.save(snap, name="Export Test", tags=["v1"])

        # Export
        json_str = env_mgr.export_json(env_id)
        data = json.loads(json_str)
        assert data["metadata"]["name"] == "Export Test"

        # Import
        imported_id = env_mgr.import_json(json_str, override_name="Imported")
        assert imported_id != env_id

        imported = env_mgr.load(imported_id)
        assert imported.metadata.name == "Imported"
        assert len(imported.stages) == 3

    def test_save_load_to_snapshot_roundtrip(self, env_mgr):
        """Snapshot → Manifest → back to Snapshot."""
        snap = _make_snapshot()
        env_id = env_mgr.save(snap, name="Roundtrip")
        loaded = env_mgr.load(env_id)
        restored_snap = loaded.to_snapshot()

        assert restored_snap.pipeline_name == snap.pipeline_name
        assert len(restored_snap.stages) == len(snap.stages)
        for orig, rest in zip(snap.stages, restored_snap.stages):
            assert orig.order == rest.order
            assert orig.name == rest.name
            assert orig.is_active == rest.is_active

    def test_diff_between_environments(self, env_mgr):
        snap1 = _make_snapshot(model="claude-sonnet-4-20250514")
        snap2 = _make_snapshot(model="claude-opus-4-20250514")
        # Modify stage 2 in snap2
        snap2.stages[1].strategies["main"] = "DifferentStrategy"
        snap2.stages[1].is_active = False

        id1 = env_mgr.save(snap1, name="Env A")
        id2 = env_mgr.save(snap2, name="Env B")

        diff = env_mgr.diff(id1, id2)
        assert isinstance(diff, EnvironmentDiff)
        assert len(diff.entries) > 0

        # Model should differ
        model_diffs = [e for e in diff.entries if "model" in e.path.lower()]
        assert len(model_diffs) > 0

    def test_variable_resolution(self, env_mgr):
        snap = _make_snapshot()
        env_id = env_mgr.save(snap, name="Var Test", tags=["vars"],
                              tools=ToolsSnapshot(
                                  mcp_servers=[{
                                      "name": "github",
                                      "env": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"}
                                  }]
                              ))
        # Check required variables
        required = env_mgr.get_required_variables(env_id)
        assert "GITHUB_TOKEN" in required

        # Resolve
        resolved = env_mgr.resolve_and_load(env_id, {"GITHUB_TOKEN": "ghp_test123"})
        assert resolved is not None

    def test_sanitize_removes_secrets(self, env_mgr):
        snap = _make_snapshot()
        snap.pipeline_config["api_key"] = "sk-secret123"
        snap.model_config["auth_token"] = "token-abc"

        env_id = env_mgr.save(snap, name="Secret Env")
        loaded = env_mgr.load(env_id)
        sanitized = EnvironmentSanitizer.sanitize(loaded.to_dict())

        # Secrets should be masked
        assert "sk-secret123" not in json.dumps(sanitized)
        assert "token-abc" not in json.dumps(sanitized)

    def test_list_and_delete(self, env_mgr):
        snap = _make_snapshot()
        env_mgr.save(snap, name="Env 1")
        env_mgr.save(snap, name="Env 2")
        env_mgr.save(snap, name="Env 3")

        all_envs = env_mgr.list_all()
        assert len(all_envs) == 3

        # Delete one
        env_mgr.delete(all_envs[0].id)
        all_envs = env_mgr.list_all()
        assert len(all_envs) == 2


# ═══════════════════════════════════════════════════════════
# Preset → Environment Integration
# ═══════════════════════════════════════════════════════════


class TestPresetEnvironmentIntegration:
    def test_save_as_preset_and_list(self, env_mgr):
        preset_mgr = PresetManager(env_mgr)
        snap = _make_snapshot()
        env_id = env_mgr.save(snap, name="My Custom Preset", tags=["custom"])

        # Mark as preset
        preset_mgr.save_as_preset(env_id)

        all_presets = preset_mgr.list_all()
        user_presets = [p for p in all_presets if p.preset_type == "user"]
        assert any(p.environment_id == env_id for p in user_presets)

    def test_remove_preset_flag(self, env_mgr):
        preset_mgr = PresetManager(env_mgr)
        snap = _make_snapshot()
        env_id = env_mgr.save(snap, name="Temp Preset")
        preset_mgr.save_as_preset(env_id)
        preset_mgr.remove_preset_flag(env_id)

        all_presets = preset_mgr.list_all()
        user_presets = [p for p in all_presets if p.preset_type == "user"]
        assert not any(p.environment_id == env_id for p in user_presets)


# ═══════════════════════════════════════════════════════════
# Tool Lifecycle Integration
# ═══════════════════════════════════════════════════════════


class TestToolLifecycle:
    def test_adhoc_tool_from_definition(self):
        """Create AdhocTool from definition and serialize."""
        defn = AdhocToolDefinition(
            name="greeter",
            description="Greet a person",
            input_schema={"type": "object", "properties": {"name": {"type": "string"}}},
            executor_type="template",
            template_config=TemplateToolConfig(template="Hello, $name!"),
        )
        tool = AdhocToolFactory.create(defn)
        assert tool.name == "greeter"

        # Roundtrip via dict
        d = defn.to_dict()
        defn2 = AdhocToolDefinition.from_dict(d)
        assert defn2.name == "greeter"
        assert defn2.executor_type == "template"

    def test_scope_resolve_include_exclude(self):
        """ToolScope include/exclude filtering."""
        scope = ToolScope(
            include={"web_search", "calculator", "code_exec"},
            exclude={"code_exec"},
        )
        all_tools = ["web_search", "calculator", "code_exec", "file_read"]

        # Need a mock PipelineState-like object
        class FakeState:
            iteration = 0
            total_cost = 0.0
        state = FakeState()

        resolved = scope.resolve(all_tools, state)
        assert "web_search" in resolved
        assert "calculator" in resolved
        assert "code_exec" not in resolved
        assert "file_read" not in resolved

    def test_scope_with_rules(self):
        """ToolScope rules dynamically add/remove tools."""
        scope = ToolScope(
            include={"web_search", "calculator"},
            rules=[
                ToolScopeRule(
                    tool_name="expensive_api",
                    action="add",
                    condition_type="iteration",
                    condition_value=3,  # int → actual >= value
                ),
            ],
        )
        all_tools = ["web_search", "calculator", "expensive_api"]

        class FakeState:
            iteration = 1
            total_cost_usd = 0.0
        state = FakeState()

        # Iteration 1: expensive_api not added
        resolved = scope.resolve(all_tools, state)
        assert "expensive_api" not in resolved

        # Iteration 3: rule triggers
        state.iteration = 3
        resolved = scope.resolve(all_tools, state)
        assert "expensive_api" in resolved


# ═══════════════════════════════════════════════════════════
# History → Environment Cross-Module
# ═══════════════════════════════════════════════════════════


class TestHistoryEnvironmentCrossModule:
    """History service recording linked to environments."""

    def test_execution_linked_to_environment(self, history_svc, env_mgr):
        snap = _make_snapshot()
        env_id = env_mgr.save(snap, name="Production Env")

        exec_id = history_svc.start_execution(
            "sess-1", "claude-sonnet-4-20250514", "test query",
            environment_id=env_id,
        )
        history_svc.finish_execution(exec_id, "completed", result_text="done")

        detail = history_svc.get_execution_detail(exec_id)
        assert detail["environment_id"] == env_id

    def test_ab_test_with_real_environments(self, history_svc, env_mgr):
        # Create two environments
        snap_a = _make_snapshot(model="claude-sonnet-4-20250514")
        snap_b = _make_snapshot(model="claude-opus-4-20250514")
        env_a = env_mgr.save(snap_a, name="Env A (Sonnet)")
        env_b = env_mgr.save(snap_b, name="Env B (Opus)")

        # Create A/B test
        runner = ABTestRunner(history_svc)
        result = runner.create_test(env_a, env_b, "Compare these models")

        # Complete both sides
        runner.complete_side(result.env_a.execution_id,
            result_text="Sonnet result",
            usage={"total_tokens": 500, "cost_usd": 0.01, "iterations": 1, "tool_calls": 0},
            duration_ms=1000, iterations=1, tool_calls_count=0,
        )
        runner.complete_side(result.env_b.execution_id,
            result_text="Opus result",
            usage={"total_tokens": 800, "cost_usd": 0.05, "iterations": 1, "tool_calls": 0},
            duration_ms=2000, iterations=1, tool_calls_count=0,
        )

        # Compare
        comp = runner.get_comparison(result.env_a.execution_id, result.env_b.execution_id)
        assert comp is not None
        assert comp["diff"]["cost_diff"] != 0

        # Verify each exec is linked to correct environment
        detail_a = history_svc.get_execution_detail(result.env_a.execution_id)
        detail_b = history_svc.get_execution_detail(result.env_b.execution_id)
        assert detail_a["environment_id"] == env_a
        assert detail_b["environment_id"] == env_b

    def test_history_stats_after_multi_execution(self, history_svc):
        """Run several executions and verify analytics."""
        for i in range(5):
            eid = history_svc.start_execution("sess-1", "claude-sonnet-4-20250514", f"query {i}")
            history_svc.record_stage_timing(eid, StageTimingRecord(
                iteration=0, stage_order=1, stage_name="Input",
                started_at=f"2025-01-01T0{i}:00:00Z",
                finished_at=f"2025-01-01T0{i}:00:00.100Z",
                duration_ms=100, input_tokens=50*(i+1),
            ))
            history_svc.finish_execution(eid, "completed", usage={
                "total_tokens": 100*(i+1), "cost_usd": 0.01*(i+1),
                "input_tokens": 50*(i+1), "output_tokens": 50*(i+1),
            })

        stats = history_svc.get_stats("sess-1")
        assert stats["total"] == 5
        assert stats["completed"] == 5
        assert stats["total_cost"] > 0

        # Cost analyzer
        analyzer = CostAnalyzer(history_svc)
        summary = analyzer.get_session_cost_summary("sess-1")
        assert summary.total_executions == 5

        # Performance monitor
        monitor = PerformanceMonitor(history_svc)
        stage_stats = monitor.get_stage_stats("sess-1")
        assert 1 in stage_stats
        assert stage_stats[1].count == 5


# ═══════════════════════════════════════════════════════════
# Snapshot Serialization Roundtrip
# ═══════════════════════════════════════════════════════════


class TestSnapshotSerialization:
    def test_json_roundtrip(self):
        snap = _make_snapshot(n_stages=5)
        json_str = snap.to_json()
        restored = PipelineSnapshot.from_json(json_str)

        assert restored.pipeline_name == snap.pipeline_name
        assert len(restored.stages) == 5
        assert restored.model_config == snap.model_config
        assert restored.pipeline_config == snap.pipeline_config

    def test_dict_roundtrip(self):
        snap = _make_snapshot()
        d = snap.to_dict()
        restored = PipelineSnapshot.from_dict(d)

        assert restored.pipeline_name == snap.pipeline_name
        for orig, rest in zip(snap.stages, restored.stages):
            assert orig.strategies == rest.strategies
            assert orig.strategy_configs == rest.strategy_configs

    def test_empty_snapshot(self):
        snap = PipelineSnapshot(pipeline_name="empty")
        d = snap.to_dict()
        restored = PipelineSnapshot.from_dict(d)
        assert restored.pipeline_name == "empty"
        assert len(restored.stages) == 0
