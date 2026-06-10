"""2.2.0 Wave 3 — manifest ``subagents`` + ``memory`` sections.

Audit §1-1's last first-class gaps: sub-agent environments and the
memory provider were host-code-only. Both become optional manifest
sections — absent → empty (full back-compat), serialized via
``to_dict``/``from_dict``, checked by ``validate_manifest`` with the
new ``subagent.*`` / ``memory.*`` issue codes.
"""

from __future__ import annotations

import logging

import pytest

from geny_executor import validate_manifest
from geny_executor.core.environment import (
    EnvironmentManifest,
    EnvironmentMetadata,
    ToolsSnapshot,
)

from tests._fixtures.manifest_entries import required_stage_entries


def _manifest(*, subagents=None, memory=None) -> EnvironmentManifest:
    """Minimal valid manifest plus the Wave 3 sections under test."""
    return EnvironmentManifest(
        metadata=EnvironmentMetadata(id="env_w3", name="wave3"),
        stages=required_stage_entries(),
        tools=ToolsSnapshot(),
        subagents=list(subagents or []),
        memory=dict(memory or {}),
    )


def _codes(issues, severity=None):
    return [i.code for i in issues if severity is None or i.severity == severity]


def _by_code(issues, code):
    found = [i for i in issues if i.code == code]
    assert found, f"expected an issue with code {code!r}; got {[i.code for i in issues]}"
    return found[0]


_SUBAGENT_ENTRY = {
    "agent_type": "researcher",
    "description": "Looks things up",
    "provider": "openai",
    "model_override": "gpt-5",
    "allowed_tools": ["Read", "WebSearch"],
    "env_id": None,
    "manifest": None,
}

_MEMORY_BLOCK = {"provider": "file", "config": {"root": "/tmp/mem"}}


# ── Serialization round-trip ─────────────────────────────────


class TestSerialization:
    def test_roundtrip_preserves_both_sections(self):
        m = _manifest(subagents=[_SUBAGENT_ENTRY], memory=_MEMORY_BLOCK)
        reloaded = EnvironmentManifest.from_dict(m.to_dict())
        assert reloaded.subagents == [_SUBAGENT_ENTRY]
        assert reloaded.memory == _MEMORY_BLOCK
        assert reloaded.to_dict() == m.to_dict()

    def test_absent_sections_default_empty(self):
        """Pre-Wave-3 payloads carry neither key — full back-compat."""
        data = _manifest().to_dict()
        del data["subagents"]
        del data["memory"]
        reloaded = EnvironmentManifest.from_dict(data)
        assert reloaded.subagents == []
        assert reloaded.memory == {}

    def test_none_payload_values_coerce_to_empty(self):
        data = _manifest().to_dict()
        data["subagents"] = None
        data["memory"] = None
        reloaded = EnvironmentManifest.from_dict(data)
        assert reloaded.subagents == []
        assert reloaded.memory == {}

    def test_sections_are_known_keys_not_warned(self, caplog):
        """The from_dict hygiene pass must not flag the new sections."""
        data = _manifest(subagents=[_SUBAGENT_ENTRY], memory=_MEMORY_BLOCK).to_dict()
        with caplog.at_level(logging.WARNING, logger="geny_executor.core.environment"):
            EnvironmentManifest.from_dict(data)
        assert "unknown" not in caplog.text

    def test_v1_migration_still_lands_empty_sections(self):
        """Legacy payloads chain through v1→v2→v3 and gain the defaults."""
        legacy = {"version": "1.0", "stages": [], "metadata": {"id": "env_old"}}
        m = EnvironmentManifest.from_dict(legacy)
        assert m.subagents == []
        assert m.memory == {}


# ── validate_manifest: subagents ─────────────────────────────


class TestValidateSubagents:
    def test_clean_entry_no_issues(self):
        entry = dict(_SUBAGENT_ENTRY, provider="anthropic")
        issues = validate_manifest(_manifest(subagents=[entry]))
        assert [c for c in _codes(issues) if c.startswith("subagent.")] == []

    def test_duplicate_agent_type_is_error(self):
        issues = validate_manifest(
            _manifest(subagents=[dict(_SUBAGENT_ENTRY), dict(_SUBAGENT_ENTRY)])
        )
        issue = _by_code(issues, "subagent.duplicate_type")
        assert issue.severity == "error"
        assert "researcher" in issue.message

    def test_missing_agent_type_is_error(self):
        issues = validate_manifest(_manifest(subagents=[{"description": "nameless"}]))
        assert _by_code(issues, "subagent.missing_type").severity == "error"

    def test_non_dict_entry_is_error(self):
        issues = validate_manifest(_manifest(subagents=["researcher"]))
        assert _by_code(issues, "subagent.malformed_entry").severity == "error"

    def test_env_id_and_manifest_both_set_is_warning(self):
        entry = dict(_SUBAGENT_ENTRY, env_id="env_x", manifest={"version": "3.0"})
        issues = validate_manifest(_manifest(subagents=[entry]))
        issue = _by_code(issues, "subagent.dual_source")
        assert issue.severity == "warning"
        assert "inline manifest wins" in issue.message

    def test_unknown_provider_is_warning_not_error(self):
        """Hosts register custom providers late — must stay a warning."""
        entry = dict(_SUBAGENT_ENTRY, provider="acme_llm")
        issues = validate_manifest(_manifest(subagents=[entry]))
        issue = _by_code(issues, "subagent.unknown_provider")
        assert issue.severity == "warning"
        assert "acme_llm" in issue.message

    def test_provider_none_means_inherit_and_is_clean(self):
        entry = dict(_SUBAGENT_ENTRY, provider=None)
        issues = validate_manifest(_manifest(subagents=[entry]))
        assert "subagent.unknown_provider" not in _codes(issues)


# ── validate_manifest: memory ────────────────────────────────


class TestValidateMemory:
    def test_clean_file_block_no_issues(self):
        issues = validate_manifest(_manifest(memory=_MEMORY_BLOCK))
        assert [c for c in _codes(issues) if c.startswith("memory.")] == []

    def test_empty_block_no_issues(self):
        issues = validate_manifest(_manifest(memory={}))
        assert [c for c in _codes(issues) if c.startswith("memory.")] == []

    def test_unknown_provider_is_error(self):
        issues = validate_manifest(_manifest(memory={"provider": "redis"}))
        issue = _by_code(issues, "memory.unknown_provider")
        assert issue.severity == "error"
        assert "redis" in issue.message

    def test_missing_provider_is_error(self):
        issues = validate_manifest(_manifest(memory={"config": {"root": "/x"}}))
        assert _by_code(issues, "memory.missing_provider").severity == "error"

    def test_unaccepted_config_key_is_warning(self):
        block = {"provider": "file", "config": {"root": "/x", "shards": 4}}
        issues = validate_manifest(_manifest(memory=block))
        issue = _by_code(issues, "memory.unknown_config_key")
        assert issue.severity == "warning"
        assert "shards" in issue.message

    def test_stray_top_level_key_is_warning(self):
        """A 'root' beside 'provider' is the forgot-to-nest mistake."""
        block = {"provider": "file", "root": "/x", "config": {"root": "/x"}}
        issues = validate_manifest(_manifest(memory=block))
        issue = _by_code(issues, "memory.unknown_key")
        assert issue.severity == "warning"
        assert "root" in issue.message

    @pytest.mark.parametrize("provider", ["ephemeral", "file", "sql", "composite"])
    def test_all_builtin_provider_names_recognised(self, provider):
        issues = validate_manifest(_manifest(memory={"provider": provider}))
        assert "memory.unknown_provider" not in _codes(issues)
