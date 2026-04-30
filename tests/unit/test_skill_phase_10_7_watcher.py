"""Phase 10.7 — SkillRegistryWatcher hot-reload."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List

import pytest

from geny_executor.skills.loader import SkillLoadReport, load_skills_dir
from geny_executor.skills.registry import SkillRegistry
from geny_executor.skills.watcher import SkillRegistryWatcher


def _write_skill(root: Path, skill_id: str, *, body: str = "body\n") -> Path:
    skill_dir = root / skill_id
    skill_dir.mkdir(parents=True, exist_ok=True)
    md = skill_dir / "SKILL.md"
    md.write_text(
        f"---\nname: {skill_id}\ndescription: t\n---\n\n{body}",
        encoding="utf-8",
    )
    return md


# ── Manual reload (no thread, deterministic) ─────────────────────────


def test_reload_now_picks_up_new_skill(tmp_path: Path) -> None:
    """Calling ``reload_now()`` synchronously rebuilds the registry
    from current disk state. We use this path for deterministic
    tests instead of waiting on the polling thread."""
    _write_skill(tmp_path, "alpha")
    registry = SkillRegistry()
    # Seed the registry with the initial state.
    registry.register_many(load_skills_dir(tmp_path).loaded)
    assert "alpha" in registry

    # Add a new skill on disk.
    _write_skill(tmp_path, "beta")

    watcher = SkillRegistryWatcher(registry, roots=[tmp_path])
    report = watcher.reload_now()
    assert report is not None
    assert {s.id for s in report.loaded} == {"alpha", "beta"}
    assert "beta" in registry


def test_reload_now_picks_up_removed_skill(tmp_path: Path) -> None:
    _write_skill(tmp_path, "alpha")
    _write_skill(tmp_path, "beta")
    registry = SkillRegistry()
    registry.register_many(load_skills_dir(tmp_path).loaded)
    assert {"alpha", "beta"} == set(registry.list_ids())

    # Remove beta.
    (tmp_path / "beta" / "SKILL.md").unlink()

    watcher = SkillRegistryWatcher(registry, roots=[tmp_path])
    watcher.reload_now()
    assert set(registry.list_ids()) == {"alpha"}


def test_reload_now_picks_up_modified_body(tmp_path: Path) -> None:
    md = _write_skill(tmp_path, "edit-me", body="old body\n")
    registry = SkillRegistry()
    registry.register_many(load_skills_dir(tmp_path).loaded)
    assert registry.get("edit-me").body.strip() == "old body"

    md.write_text(
        "---\nname: edit-me\ndescription: t\n---\n\nnew body\n",
        encoding="utf-8",
    )

    watcher = SkillRegistryWatcher(registry, roots=[tmp_path])
    watcher.reload_now()
    assert registry.get("edit-me").body.strip() == "new body"


def test_reload_now_handles_empty_root(tmp_path: Path) -> None:
    """Watcher treats a missing or empty root as 'nothing to load'
    rather than raising."""
    registry = SkillRegistry()
    watcher = SkillRegistryWatcher(registry, roots=[tmp_path / "nope"])
    report = watcher.reload_now()
    assert report is not None
    assert report.loaded == []
    assert len(registry) == 0


def test_on_change_callback_fires_on_reload(tmp_path: Path) -> None:
    _write_skill(tmp_path, "x")
    registry = SkillRegistry()
    registry.register_many(load_skills_dir(tmp_path).loaded)

    calls: List[SkillLoadReport] = []
    watcher = SkillRegistryWatcher(
        registry,
        roots=[tmp_path],
        on_change=lambda r: calls.append(r),
    )
    watcher.reload_now()
    assert len(calls) == 1
    assert {s.id for s in calls[0].loaded} == {"x"}


# ── Background polling (start/stop, real thread) ─────────────────────


def test_start_stop_idempotent(tmp_path: Path) -> None:
    registry = SkillRegistry()
    watcher = SkillRegistryWatcher(
        registry, roots=[tmp_path], poll_interval_s=0.1
    )
    watcher.start()
    assert watcher.is_running
    # Calling start again is a no-op — same thread.
    watcher.start()
    assert watcher.is_running
    watcher.stop()
    assert not watcher.is_running
    # Stop is idempotent too.
    watcher.stop()


def test_polling_picks_up_change_in_background(tmp_path: Path) -> None:
    """Drive the actual thread to make sure the loop fires, not just
    the manual reload helper."""
    _write_skill(tmp_path, "before")
    registry = SkillRegistry()
    registry.register_many(load_skills_dir(tmp_path).loaded)

    changed: List[bool] = []
    watcher = SkillRegistryWatcher(
        registry,
        roots=[tmp_path],
        poll_interval_s=0.05,
        debounce_s=0.0,
        on_change=lambda _r: changed.append(True),
    )
    watcher.start()
    try:
        # Add a new skill on disk after the watcher took its initial
        # snapshot. The next poll should detect + reload.
        time.sleep(0.05)
        _write_skill(tmp_path, "after")
        # Force a stat-distinguishable mtime — some filesystems have
        # second-level resolution.
        os.utime(tmp_path / "after" / "SKILL.md", None)

        # Wait for the change to land. Up to 2s.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if "after" in registry:
                break
            time.sleep(0.05)
    finally:
        watcher.stop()

    assert "after" in registry
    assert len(changed) >= 1


def test_debounce_collapses_rapid_changes(tmp_path: Path) -> None:
    """Rapid edits within the debounce window should produce one
    reload, not one per change."""
    _write_skill(tmp_path, "x")
    registry = SkillRegistry()
    registry.register_many(load_skills_dir(tmp_path).loaded)

    fires: List[int] = []
    watcher = SkillRegistryWatcher(
        registry,
        roots=[tmp_path],
        poll_interval_s=0.03,
        debounce_s=0.5,
        on_change=lambda _r: fires.append(time.monotonic_ns()),
    )
    watcher.start()
    try:
        # Three writes spaced 0.05s apart — well inside the 0.5s
        # debounce window.
        for i in range(3):
            time.sleep(0.05)
            (tmp_path / "x" / "SKILL.md").write_text(
                f"---\nname: x\ndescription: t\n---\n\nrev{i}\n",
                encoding="utf-8",
            )
        # Wait for debounce to clear + one poll past it.
        time.sleep(0.7)
    finally:
        watcher.stop()

    # Expect exactly 1 reload — possibly 2 in pathological timing,
    # but never the 3 we'd see without debounce.
    assert 1 <= len(fires) <= 2, f"expected debounce to collapse to 1-2, got {len(fires)}"


def test_error_callback_receives_exceptions(tmp_path: Path) -> None:
    """If a callback the watcher invokes raises, ``on_error`` gets
    the exception. The watcher itself stays alive."""
    _write_skill(tmp_path, "x")
    registry = SkillRegistry()
    registry.register_many(load_skills_dir(tmp_path).loaded)

    errors: List[Exception] = []

    def explosive_on_change(_report: SkillLoadReport) -> None:
        raise RuntimeError("callback boom")

    watcher = SkillRegistryWatcher(
        registry,
        roots=[tmp_path],
        on_change=explosive_on_change,
        on_error=lambda exc: errors.append(exc),
    )
    watcher.reload_now()

    assert len(errors) == 1
    assert "callback boom" in str(errors[0])


# ── Multiple roots ──────────────────────────────────────────────────


def test_watches_multiple_roots(tmp_path: Path) -> None:
    user_root = tmp_path / "user"
    project_root = tmp_path / "project"
    user_root.mkdir()
    project_root.mkdir()
    _write_skill(user_root, "user-skill")
    _write_skill(project_root, "project-skill")

    registry = SkillRegistry()
    watcher = SkillRegistryWatcher(
        registry, roots=[user_root, project_root]
    )
    watcher.reload_now()
    assert set(registry.list_ids()) == {"user-skill", "project-skill"}


def test_root_collision_first_wins(tmp_path: Path) -> None:
    """If the same skill id lives in two watched roots, the first
    root wins (matches SkillRegistry's first-wins policy — second
    register raises and the watcher catches it via on_error)."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _write_skill(a, "dup", body="from-a\n")
    _write_skill(b, "dup", body="from-b\n")

    errors: List[Exception] = []
    registry = SkillRegistry()
    watcher = SkillRegistryWatcher(
        registry,
        roots=[a, b],
        on_error=lambda exc: errors.append(exc),
    )
    watcher.reload_now()

    # The collision propagates as an on_error so the operator sees
    # which directory clashed.
    assert any("dup" in str(e) for e in errors)
