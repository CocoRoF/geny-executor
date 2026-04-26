"""SettingsLoader + section registry tests (PR-B.3.1)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

from geny_executor.settings import (
    SettingsLoader,
    get_default_loader,
    register_section,
    reset_default_loader,
)
from geny_executor.settings.section_registry import (
    list_section_names,
    reset_section_registry,
)


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture(autouse=True)
def _isolate_default():
    reset_default_loader()
    reset_section_registry()
    yield
    reset_default_loader()
    reset_section_registry()


# ── Loader basics ────────────────────────────────────────────────────


class TestLoad:
    def test_empty_paths_returns_empty(self):
        loader = SettingsLoader(paths=[])
        assert loader.load() == {}

    def test_single_file(self, tmp_path: Path):
        p = tmp_path / "settings.json"
        _write_json(p, {"model": {"default": "claude-haiku"}})
        loader = SettingsLoader(paths=[p])
        assert loader.get_section("model")["default"] == "claude-haiku"

    def test_missing_file_skipped(self, tmp_path: Path):
        loader = SettingsLoader(paths=[tmp_path / "ghost.json"])
        assert loader.load() == {}

    def test_invalid_json_skipped(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text("not json", encoding="utf-8")
        loader = SettingsLoader(paths=[p])
        assert loader.load() == {}

    def test_root_not_object_skipped(self, tmp_path: Path):
        p = tmp_path / "list.json"
        _write_json(p, ["not", "an", "object"])
        loader = SettingsLoader(paths=[p])
        assert loader.load() == {}


# ── Cascade + deep merge ─────────────────────────────────────────────


class TestCascade:
    def test_later_overlays_earlier(self, tmp_path: Path):
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        _write_json(a, {"model": {"default": "haiku", "temp": 0.7}})
        _write_json(b, {"model": {"default": "sonnet"}})
        loader = SettingsLoader(paths=[a, b])
        section = loader.get_section("model")
        assert section["default"] == "sonnet"   # overlay wins
        assert section["temp"] == 0.7           # base preserved

    def test_lists_replace_not_concat(self, tmp_path: Path):
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        _write_json(a, {"perms": {"deny": ["x", "y"]}})
        _write_json(b, {"perms": {"deny": ["z"]}})
        loader = SettingsLoader(paths=[a, b])
        assert loader.get_section("perms")["deny"] == ["z"]


# ── add_path / reload ────────────────────────────────────────────────


class TestMutation:
    def test_add_path_after_load(self, tmp_path: Path):
        a = tmp_path / "a.json"
        _write_json(a, {"x": 1})
        loader = SettingsLoader(paths=[a])
        loader.load()
        b = tmp_path / "b.json"
        _write_json(b, {"y": 2})
        loader.add_path(b)
        assert loader.get_section("x") == 1
        assert loader.get_section("y") == 2

    def test_reload_picks_up_changes(self, tmp_path: Path):
        p = tmp_path / "settings.json"
        _write_json(p, {"k": "v1"})
        loader = SettingsLoader(paths=[p])
        assert loader.get_section("k") == "v1"
        _write_json(p, {"k": "v2"})
        loader.reload()
        assert loader.get_section("k") == "v2"


# ── Section registry ─────────────────────────────────────────────────


class _ModelSection:
    def __init__(self, default: str = "claude-haiku", temp: float = 0.7):
        self.default = default
        self.temp = temp


class TestSectionRegistry:
    def test_register_and_lookup(self):
        register_section("model", _ModelSection)
        assert "model" in list_section_names()

    def test_get_section_validates_via_schema(self, tmp_path: Path):
        register_section("model", _ModelSection)
        p = tmp_path / "settings.json"
        _write_json(p, {"model": {"default": "claude-sonnet", "temp": 0.5}})
        loader = SettingsLoader(paths=[p])
        section = loader.get_section("model")
        assert isinstance(section, _ModelSection)
        assert section.default == "claude-sonnet"

    def test_invalid_section_returns_default(self, tmp_path: Path):
        register_section("model", _ModelSection)
        p = tmp_path / "settings.json"
        # Constructor only accepts default+temp; extra key crashes.
        _write_json(p, {"model": {"default": "x", "wrong_field": 1}})
        loader = SettingsLoader(paths=[p])
        # Schema rejected → default returned (None unless we pass one).
        assert loader.get_section("model", default="fallback") == "fallback"

    def test_section_without_schema_returns_raw_dict(self, tmp_path: Path):
        p = tmp_path / "settings.json"
        _write_json(p, {"untyped": {"raw": True}})
        loader = SettingsLoader(paths=[p])
        assert loader.get_section("untyped") == {"raw": True}

    def test_missing_section_returns_default(self):
        loader = SettingsLoader(paths=[])
        assert loader.get_section("ghost", default="fallback") == "fallback"


# ── Default loader singleton ─────────────────────────────────────────


class TestDefaultLoader:
    def test_singleton(self):
        a = get_default_loader()
        b = get_default_loader()
        assert a is b

    def test_reset_returns_fresh(self):
        a = get_default_loader()
        b = reset_default_loader()
        assert b is not a
        assert get_default_loader() is b
