"""Phase 5 — Environment System tests.

Tests:
  - EnvironmentManifest serialization round-trip
  - EnvironmentResolver variable expansion
  - EnvironmentDiff deep comparison
  - EnvironmentManager CRUD + import/export
  - EnvironmentSanitizer sensitive data removal
  - PresetManager built-in + user presets
"""

import sys, os, json, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
from geny_executor.core.environment import (
    EnvironmentManifest,
    EnvironmentManager,
    EnvironmentMetadata,
    EnvironmentResolver,
    EnvironmentSanitizer,
    EnvironmentSummary,
    ToolsSnapshot,
)
from geny_executor.core.diff import DiffEntry, EnvironmentDiff
from geny_executor.core.snapshot import PipelineSnapshot, StageSnapshot
from geny_executor.core.presets import PresetInfo, PresetManager


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture
def tmp_storage(tmp_path):
    """Temporary storage directory for EnvironmentManager."""
    return str(tmp_path / "environments")


@pytest.fixture
def manager(tmp_storage):
    return EnvironmentManager(storage_path=tmp_storage)


@pytest.fixture
def sample_snapshot():
    return PipelineSnapshot(
        pipeline_name="test-pipeline",
        stages=[
            StageSnapshot(
                order=1,
                name="input",
                is_active=True,
                strategies={"InputValidator": "StrictValidator"},
                strategy_configs={"InputValidator": {"max_length": 50000}},
                stage_config={},
            ),
            StageSnapshot(
                order=6,
                name="api",
                is_active=True,
                strategies={"APIProvider": "AnthropicProvider"},
                strategy_configs={"APIProvider": {}},
                stage_config={},
            ),
            StageSnapshot(
                order=9,
                name="parse",
                is_active=True,
                strategies={},
                strategy_configs={},
                stage_config={"strict_json": False},
            ),
        ],
        pipeline_config={
            "max_iterations": 30,
            "cost_budget_usd": 2.0,
            "stream": True,
        },
        model_config={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 8192,
            "temperature": 0.0,
            "thinking_enabled": True,
        },
        description="Test snapshot",
    )


# ═══════════════════════════════════════════════════════════
#  EnvironmentManifest
# ═══════════════════════════════════════════════════════════

class TestEnvironmentManifest:

    def test_from_snapshot(self, sample_snapshot):
        manifest = EnvironmentManifest.from_snapshot(
            sample_snapshot, "Test Env", "Description", ["test", "dev"]
        )
        assert manifest.metadata.name == "Test Env"
        assert manifest.metadata.description == "Description"
        assert manifest.metadata.tags == ["test", "dev"]
        assert manifest.metadata.id.startswith("env_")
        assert len(manifest.stages) == 3
        assert manifest.model["model"] == "claude-sonnet-4-20250514"

    def test_roundtrip_dict(self, sample_snapshot):
        manifest = EnvironmentManifest.from_snapshot(sample_snapshot, "RT Test")
        d = manifest.to_dict()
        restored = EnvironmentManifest.from_dict(d)

        assert restored.version == manifest.version
        assert restored.metadata.name == "RT Test"
        assert len(restored.stages) == 3
        assert restored.model == manifest.model
        assert restored.pipeline == manifest.pipeline

    def test_roundtrip_json(self, sample_snapshot):
        manifest = EnvironmentManifest.from_snapshot(sample_snapshot, "JSON Test")
        json_str = json.dumps(manifest.to_dict(), ensure_ascii=False)
        parsed = json.loads(json_str)
        restored = EnvironmentManifest.from_dict(parsed)
        assert restored.metadata.name == "JSON Test"

    def test_to_snapshot(self, sample_snapshot):
        manifest = EnvironmentManifest.from_snapshot(sample_snapshot, "Snap Test")
        snapshot = manifest.to_snapshot()
        assert snapshot.pipeline_name == "test-pipeline"
        assert len(snapshot.stages) == 3
        assert snapshot.stages[0].name == "input"
        assert snapshot.stages[0].is_active is True

    def test_update(self, sample_snapshot):
        manifest = EnvironmentManifest.from_snapshot(sample_snapshot, "Old Name")
        manifest.update({"metadata": {"name": "New Name", "tags": ["updated"]}})
        assert manifest.metadata.name == "New Name"
        assert manifest.metadata.tags == ["updated"]

    def test_update_model(self, sample_snapshot):
        manifest = EnvironmentManifest.from_snapshot(sample_snapshot, "Model Test")
        manifest.update({"model": {"temperature": 0.5}})
        assert manifest.model["temperature"] == 0.5
        # Original fields preserved
        assert manifest.model["model"] == "claude-sonnet-4-20250514"


class TestToolsSnapshot:

    def test_roundtrip(self):
        tools = ToolsSnapshot(
            built_in=["Read", "Glob"],
            adhoc=[{"name": "lint_check", "executor_type": "script"}],
            mcp_servers=[{"name": "github", "transport": "stdio"}],
            scope={"global": {"include": None, "exclude": ["Write"]}},
        )
        d = tools.to_dict()
        restored = ToolsSnapshot.from_dict(d)
        assert restored.built_in == ["Read", "Glob"]
        assert len(restored.adhoc) == 1
        assert restored.mcp_servers[0]["name"] == "github"


# ═══════════════════════════════════════════════════════════
#  EnvironmentResolver
# ═══════════════════════════════════════════════════════════

class TestEnvironmentResolver:

    def test_resolve_simple(self):
        data = {"api_key": "${API_KEY}", "name": "test"}
        resolved = EnvironmentResolver.resolve(data, {"API_KEY": "sk-123"})
        assert resolved["api_key"] == "sk-123"
        assert resolved["name"] == "test"

    def test_resolve_nested(self):
        data = {
            "tools": {
                "mcp_servers": [
                    {"env": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"}}
                ]
            }
        }
        resolved = EnvironmentResolver.resolve(data, {"GITHUB_TOKEN": "ghp_abc"})
        assert resolved["tools"]["mcp_servers"][0]["env"]["GITHUB_TOKEN"] == "ghp_abc"

    def test_unresolved_kept(self):
        data = {"key": "${UNKNOWN_VAR}"}
        resolved = EnvironmentResolver.resolve(data, {})
        assert resolved["key"] == "${UNKNOWN_VAR}"

    def test_multiple_in_string(self):
        data = {"url": "https://${HOST}:${PORT}/api"}
        resolved = EnvironmentResolver.resolve(
            data, {"HOST": "localhost", "PORT": "8080"}
        )
        assert resolved["url"] == "https://localhost:8080/api"

    def test_extract_variables(self):
        data = {
            "api_key": "${API_KEY}",
            "servers": [{"token": "${GITHUB_TOKEN}"}],
            "name": "no-vars",
        }
        variables = EnvironmentResolver.extract_variables(data)
        assert variables == {"API_KEY", "GITHUB_TOKEN"}

    def test_extract_empty(self):
        data = {"name": "simple", "count": 42}
        variables = EnvironmentResolver.extract_variables(data)
        assert variables == set()


# ═══════════════════════════════════════════════════════════
#  EnvironmentDiff
# ═══════════════════════════════════════════════════════════

class TestEnvironmentDiff:

    def test_identical(self):
        a = {"model": {"temperature": 0.5}, "pipeline": {"stream": True}}
        diff = EnvironmentDiff.compute(a, a)
        assert diff.identical
        assert diff.summary == {"added": 0, "removed": 0, "changed": 0}

    def test_added(self):
        a = {"model": {"temperature": 0.5}}
        b = {"model": {"temperature": 0.5, "top_p": 0.9}}
        diff = EnvironmentDiff.compute(a, b)
        assert not diff.identical
        assert diff.summary["added"] == 1
        assert diff.entries[0].path == "model.top_p"
        assert diff.entries[0].new_value == 0.9

    def test_removed(self):
        a = {"model": {"temperature": 0.5, "top_p": 0.9}}
        b = {"model": {"temperature": 0.5}}
        diff = EnvironmentDiff.compute(a, b)
        assert diff.summary["removed"] == 1

    def test_changed(self):
        a = {"model": {"temperature": 0.5}}
        b = {"model": {"temperature": 0.8}}
        diff = EnvironmentDiff.compute(a, b)
        assert diff.summary["changed"] == 1
        assert diff.entries[0].old_value == 0.5
        assert diff.entries[0].new_value == 0.8

    def test_ignore_metadata_ids(self):
        a = {"metadata": {"id": "env_1", "name": "A", "created_at": "t1"}}
        b = {"metadata": {"id": "env_2", "name": "B", "created_at": "t2"}}
        diff = EnvironmentDiff.compute(a, b)
        # id and created_at should be ignored
        paths = {e.path for e in diff.entries}
        assert "metadata.id" not in paths
        assert "metadata.created_at" not in paths
        assert "metadata.name" in paths

    def test_nested_list_dict_diff(self):
        a = {"stages": [{"order": 1, "name": "input", "active": True}]}
        b = {"stages": [{"order": 1, "name": "input", "active": False}]}
        diff = EnvironmentDiff.compute(a, b)
        assert diff.summary["changed"] == 1
        assert "stages[0].active" in diff.entries[0].path

    def test_list_length_diff(self):
        a = {"tags": ["a", "b"]}
        b = {"tags": ["a", "b", "c"]}
        diff = EnvironmentDiff.compute(a, b)
        assert diff.summary["changed"] == 1

    def test_human_readable(self):
        e = DiffEntry("model.temperature", "changed", 0.5, 0.8)
        assert "→" in e.human_readable()

    def test_filter_by_type(self):
        diff = EnvironmentDiff(entries=[
            DiffEntry("a", "added", new_value=1),
            DiffEntry("b", "removed", old_value=2),
            DiffEntry("c", "changed", 3, 4),
        ])
        added = diff.filter_by_type("added")
        assert len(added.entries) == 1

    def test_filter_by_prefix(self):
        diff = EnvironmentDiff(entries=[
            DiffEntry("model.temperature", "changed", 0.5, 0.8),
            DiffEntry("pipeline.stream", "changed", True, False),
        ])
        model_only = diff.filter_by_prefix("model")
        assert len(model_only.entries) == 1

    def test_serialization_roundtrip(self):
        diff = EnvironmentDiff(entries=[
            DiffEntry("model.temperature", "changed", 0.5, 0.8),
            DiffEntry("pipeline.stream", "added", new_value=True),
        ])
        d = diff.to_dict()
        restored = EnvironmentDiff.from_dict(d)
        assert len(restored.entries) == 2
        assert restored.entries[0].change_type == "changed"


# ═══════════════════════════════════════════════════════════
#  EnvironmentManager
# ═══════════════════════════════════════════════════════════

class TestEnvironmentManager:

    def test_save_and_load(self, manager, sample_snapshot):
        env_id = manager.save(sample_snapshot, "Test Env", "Desc", ["tag1"])
        loaded = manager.load(env_id)
        assert loaded.metadata.name == "Test Env"
        assert loaded.metadata.description == "Desc"
        assert loaded.metadata.tags == ["tag1"]
        assert len(loaded.stages) == 3

    def test_list_all(self, manager, sample_snapshot):
        manager.save(sample_snapshot, "Env A")
        manager.save(sample_snapshot, "Env B")
        envs = manager.list_all()
        assert len(envs) == 2
        names = {e.name for e in envs}
        assert "Env A" in names
        assert "Env B" in names

    def test_delete(self, manager, sample_snapshot):
        env_id = manager.save(sample_snapshot, "To Delete")
        assert manager.delete(env_id)
        assert not manager.delete(env_id)  # already deleted
        with pytest.raises(FileNotFoundError):
            manager.load(env_id)

    def test_update(self, manager, sample_snapshot):
        env_id = manager.save(sample_snapshot, "Old Name")
        updated = manager.update(env_id, {"metadata": {"name": "New Name"}})
        assert updated.metadata.name == "New Name"
        # Reload to verify persistence
        manager._cache.clear()
        reloaded = manager.load(env_id)
        assert reloaded.metadata.name == "New Name"

    def test_export_import_json(self, manager, sample_snapshot):
        env_id = manager.save(sample_snapshot, "Export Test")
        json_str = manager.export_json(env_id)
        parsed = json.loads(json_str)
        assert parsed["metadata"]["name"] == "Export Test"

        new_id = manager.import_json(json_str, override_name="Imported")
        imported = manager.load(new_id)
        assert imported.metadata.name == "Imported"
        assert new_id != env_id  # new ID assigned

    def test_diff(self, manager, sample_snapshot):
        id_a = manager.save(sample_snapshot, "Env A")
        # Create a variant
        snapshot_b = PipelineSnapshot(
            pipeline_name="test-pipeline",
            stages=sample_snapshot.stages,
            pipeline_config={"max_iterations": 50, "stream": True},
            model_config=sample_snapshot.model_config,
        )
        id_b = manager.save(snapshot_b, "Env B")
        diff = manager.diff(id_a, id_b)
        assert not diff.identical
        # cost_budget_usd removed (was in A, not in B), max_iterations changed
        paths = {e.path for e in diff.entries}
        assert "pipeline.max_iterations" in paths

    def test_load_not_found(self, manager):
        with pytest.raises(FileNotFoundError):
            manager.load("env_nonexistent")

    def test_resolve_and_load(self, manager, sample_snapshot):
        # Save an environment with variable references
        env_id = manager.save(sample_snapshot, "Var Env")
        # Manually inject a variable reference
        manifest = manager.load(env_id)
        manifest.model["api_key"] = "${MY_API_KEY}"
        import pathlib
        path = pathlib.Path(manager._storage) / f"{env_id}.json"
        path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2))
        manager._cache.clear()

        resolved = manager.resolve_and_load(env_id, {"MY_API_KEY": "sk-test"})
        assert resolved.model["api_key"] == "sk-test"

    def test_get_required_variables(self, manager, sample_snapshot):
        env_id = manager.save(sample_snapshot, "Var Env")
        manifest = manager.load(env_id)
        manifest.tools = ToolsSnapshot(
            mcp_servers=[{"env": {"TOKEN": "${GITHUB_TOKEN}"}}]
        )
        import pathlib
        path = pathlib.Path(manager._storage) / f"{env_id}.json"
        path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2))
        manager._cache.clear()

        vars_needed = manager.get_required_variables(env_id)
        assert "GITHUB_TOKEN" in vars_needed


# ═══════════════════════════════════════════════════════════
#  EnvironmentSanitizer
# ═══════════════════════════════════════════════════════════

class TestEnvironmentSanitizer:

    def test_sanitize_api_key(self):
        data = {"model": {"api_key": "sk-real-secret"}}
        sanitized = EnvironmentSanitizer.sanitize(data)
        assert sanitized["model"]["api_key"] == "${API_KEY}"
        # Original not mutated
        assert data["model"]["api_key"] == "sk-real-secret"

    def test_sanitize_nested_token(self):
        data = {
            "tools": {
                "mcp_servers": [
                    {"name": "github", "env": {"GITHUB_TOKEN": "ghp_real"}}
                ]
            }
        }
        sanitized = EnvironmentSanitizer.sanitize(data)
        assert sanitized["tools"]["mcp_servers"][0]["env"]["GITHUB_TOKEN"] == "${GITHUB_TOKEN}"

    def test_sanitize_multiple_sensitive_keys(self):
        data = {
            "api_key": "sk-123",
            "password": "hunter2",
            "secret_token": "secret",
            "name": "safe",
        }
        sanitized = EnvironmentSanitizer.sanitize(data)
        assert sanitized["api_key"] == "${API_KEY}"
        assert sanitized["password"] == "${PASSWORD}"
        assert sanitized["secret_token"] == "${SECRET_TOKEN}"
        assert sanitized["name"] == "safe"

    def test_sanitize_preserves_structure(self):
        data = {
            "model": {"temperature": 0.5},
            "pipeline": {"stream": True},
            "stages": [{"name": "input", "active": True}],
        }
        sanitized = EnvironmentSanitizer.sanitize(data)
        assert sanitized == data  # No sensitive keys → identical


# ═══════════════════════════════════════════════════════════
#  PresetManager
# ═══════════════════════════════════════════════════════════

class TestPresetManager:

    def test_list_built_in(self, manager):
        pm = PresetManager(manager)
        presets = pm.list_all()
        names = {p.name for p in presets}
        assert "minimal" in names
        assert "chat" in names
        assert "agent" in names
        assert "evaluator" in names
        assert "geny_vtuber" in names
        assert all(p.preset_type == "built_in" for p in presets)

    def test_save_as_preset(self, manager, sample_snapshot):
        pm = PresetManager(manager)
        env_id = manager.save(sample_snapshot, "My Preset", tags=["dev"])
        pm.save_as_preset(env_id)

        manifest = manager.load(env_id)
        assert "preset" in manifest.metadata.tags

        # Appears in list_all
        presets = pm.list_all()
        user_presets = [p for p in presets if p.preset_type == "user"]
        assert len(user_presets) == 1
        assert user_presets[0].environment_id == env_id

    def test_remove_preset_flag(self, manager, sample_snapshot):
        pm = PresetManager(manager)
        env_id = manager.save(sample_snapshot, "Removable", tags=["preset"])
        pm.remove_preset_flag(env_id)
        manifest = manager.load(env_id)
        assert "preset" not in manifest.metadata.tags

    def test_save_as_preset_idempotent(self, manager, sample_snapshot):
        pm = PresetManager(manager)
        env_id = manager.save(sample_snapshot, "Idem", tags=["preset"])
        pm.save_as_preset(env_id)  # already has the tag
        manifest = manager.load(env_id)
        assert manifest.metadata.tags.count("preset") == 1  # not duplicated
