"""Unit tests for the MCP credential store (S8.1)."""

from __future__ import annotations

import json
import os
import stat

import pytest

from geny_executor.tools.mcp import (
    CredentialStore,
    FileCredentialStore,
    MemoryCredentialStore,
    mcp_credential_key,
)


# ── mcp_credential_key ──────────────────────────────────────────────────


class TestKeyHelper:
    def test_canonical_prefix(self):
        assert mcp_credential_key("gdrive") == "mcp:gdrive"

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            mcp_credential_key("")


# ── MemoryCredentialStore ──────────────────────────────────────────────


class TestMemoryStore:
    def test_set_and_get(self):
        s = MemoryCredentialStore()
        s.set("k", "v")
        assert s.get("k") == "v"

    def test_get_missing_returns_none(self):
        assert MemoryCredentialStore().get("k") is None

    def test_set_overwrites(self):
        s = MemoryCredentialStore()
        s.set("k", "v1")
        s.set("k", "v2")
        assert s.get("k") == "v2"

    def test_delete_existing_returns_true(self):
        s = MemoryCredentialStore()
        s.set("k", "v")
        assert s.delete("k") is True
        assert s.get("k") is None

    def test_delete_missing_returns_false(self):
        assert MemoryCredentialStore().delete("k") is False

    def test_keys_sorted(self):
        s = MemoryCredentialStore()
        s.set("c", "1")
        s.set("a", "1")
        s.set("b", "1")
        assert s.keys() == ["a", "b", "c"]

    def test_blank_key_rejected(self):
        s = MemoryCredentialStore()
        with pytest.raises(ValueError):
            s.set("", "v")
        with pytest.raises(ValueError):
            s.get("")
        with pytest.raises(ValueError):
            s.delete("")

    def test_non_string_value_rejected(self):
        s = MemoryCredentialStore()
        with pytest.raises(TypeError):
            s.set("k", 123)  # type: ignore[arg-type]

    def test_satisfies_protocol(self):
        assert isinstance(MemoryCredentialStore(), CredentialStore)


# ── FileCredentialStore ────────────────────────────────────────────────


class TestFileStore:
    def test_get_on_missing_file_returns_none(self, tmp_path):
        s = FileCredentialStore(tmp_path / "creds.json")
        assert s.get("k") is None
        assert s.keys() == []

    def test_set_creates_file_with_mode_0600(self, tmp_path):
        path = tmp_path / "creds.json"
        s = FileCredentialStore(path)
        s.set("k", "v")
        assert path.exists()
        # Mode check (POSIX). On Windows this would skip, but the
        # CI runs on Linux so we assert directly.
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o600

    def test_round_trip(self, tmp_path):
        path = tmp_path / "creds.json"
        s = FileCredentialStore(path)
        s.set("a", "1")
        s.set("b", "2")
        # Re-open from disk to confirm persistence.
        s2 = FileCredentialStore(path)
        assert s2.get("a") == "1"
        assert s2.get("b") == "2"
        assert s2.keys() == ["a", "b"]

    def test_overwrite(self, tmp_path):
        s = FileCredentialStore(tmp_path / "creds.json")
        s.set("k", "v1")
        s.set("k", "v2")
        assert s.get("k") == "v2"

    def test_delete_existing(self, tmp_path):
        s = FileCredentialStore(tmp_path / "creds.json")
        s.set("k", "v")
        assert s.delete("k") is True
        assert s.get("k") is None

    def test_delete_missing_returns_false(self, tmp_path):
        s = FileCredentialStore(tmp_path / "creds.json")
        s.set("other", "v")
        assert s.delete("k") is False
        # Other entries unaffected.
        assert s.get("other") == "v"

    def test_creates_parent_directory(self, tmp_path):
        path = tmp_path / "nested" / "subdir" / "creds.json"
        s = FileCredentialStore(path)
        s.set("k", "v")
        assert path.exists()
        assert s.get("k") == "v"

    def test_corrupt_file_raises(self, tmp_path):
        path = tmp_path / "creds.json"
        path.write_text("not json {", encoding="utf-8")
        s = FileCredentialStore(path)
        with pytest.raises(ValueError, match="corrupt"):
            s.get("k")

    def test_non_object_json_rejected(self, tmp_path):
        path = tmp_path / "creds.json"
        path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
        s = FileCredentialStore(path)
        with pytest.raises(ValueError, match="JSON object"):
            s.get("k")

    def test_empty_file_treated_as_empty(self, tmp_path):
        path = tmp_path / "creds.json"
        path.write_text("", encoding="utf-8")
        s = FileCredentialStore(path)
        assert s.get("k") is None
        assert s.keys() == []

    def test_atomic_replace_does_not_leave_temp(self, tmp_path):
        path = tmp_path / "creds.json"
        s = FileCredentialStore(path)
        s.set("a", "1")
        s.set("b", "2")
        s.set("c", "3")
        leftover = [p.name for p in tmp_path.iterdir() if p.name.startswith(".cred-")]
        assert leftover == []

    def test_path_property(self, tmp_path):
        path = tmp_path / "creds.json"
        assert FileCredentialStore(path).path == path

    def test_satisfies_protocol(self, tmp_path):
        assert isinstance(FileCredentialStore(tmp_path / "x.json"), CredentialStore)

    def test_blank_key_rejected(self, tmp_path):
        s = FileCredentialStore(tmp_path / "creds.json")
        with pytest.raises(ValueError):
            s.set("", "v")

    def test_concurrent_set_persists_last(self, tmp_path):
        """No real race-coverage, but verifies file lock semantics under sequential reuse."""
        s = FileCredentialStore(tmp_path / "creds.json")
        for i in range(20):
            s.set("k", str(i))
        assert s.get("k") == "19"

    def test_existing_file_mode_preserved_on_rewrite(self, tmp_path):
        """Even pre-existing files end up at 0600 after a rewrite."""
        path = tmp_path / "creds.json"
        # Write a wide-open file first.
        path.write_text("{}", encoding="utf-8")
        os.chmod(path, 0o644)
        s = FileCredentialStore(path)
        s.set("k", "v")
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o600
