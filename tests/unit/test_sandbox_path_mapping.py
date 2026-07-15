"""Host ↔ container path translation for workspace-unified sandboxes.

Regression for the split-brain failure (2026-07-15): a HOST-absolute
working_dir passed verbatim to ``docker exec -w`` chdir-killed every
sandboxed Bash/Read call ("Bash has a broken working directory") because
the path did not exist inside the container.
"""

from __future__ import annotations

from types import SimpleNamespace

from geny_executor.tools._sandbox import (
    map_into_container,
    resolve_container_workdir,
)

HOST_ROOT = "/data/geny_agent_sessions/abc-123"


def _mapped_handle():
    """A handle for a sandbox whose /workspace binds HOST_ROOT/workspace."""

    def map_path(p: str):
        base = HOST_ROOT + "/workspace"
        if p == base:
            return "/workspace"
        if p.startswith(base + "/"):
            return "/workspace/" + p[len(base) + 1 :]
        return None

    return SimpleNamespace(
        container_name="ws", container_workdir="/workspace", map_path=map_path
    )


def _legacy_handle():
    return SimpleNamespace(container_name="ws")


class TestResolveWorkdir:
    def test_mapped_host_workdir(self):
        h = _mapped_handle()
        assert (
            resolve_container_workdir(h, HOST_ROOT + "/workspace/uploads")
            == "/workspace/uploads"
        )

    def test_container_side_workdir_passes_through(self):
        h = _mapped_handle()
        assert resolve_container_workdir(h, "/workspace/x") == "/workspace/x"

    def test_unmappable_host_workdir_degrades_to_root(self):
        """THE regression: host path + no mapping must NOT reach docker -w."""
        h = _legacy_handle()
        assert resolve_container_workdir(h, HOST_ROOT) == "/workspace"

    def test_none_workdir_defaults(self):
        assert resolve_container_workdir(_legacy_handle(), None) == "/workspace"

    def test_custom_container_workdir(self):
        h = SimpleNamespace(container_name="ws", container_workdir="/srv/app")
        assert resolve_container_workdir(h, None) == "/srv/app"
        assert resolve_container_workdir(h, "/srv/app/sub") == "/srv/app/sub"


class TestMapIntoContainer:
    def test_host_absolute_file_maps(self):
        h = _mapped_handle()
        assert (
            map_into_container(h, HOST_ROOT + "/workspace/uploads/a.pptx", HOST_ROOT + "/workspace")
            == "/workspace/uploads/a.pptx"
        )

    def test_relative_joins_mapped_workdir(self):
        h = _mapped_handle()
        assert (
            map_into_container(h, "outputs/deck.pptx", HOST_ROOT + "/workspace")
            == "/workspace/outputs/deck.pptx"
        )

    def test_legacy_behaviour_unchanged(self):
        h = _legacy_handle()
        assert map_into_container(h, "a.txt", "/workspace") == "/workspace/a.txt"

    def test_escape_still_refused(self):
        h = _mapped_handle()
        import pytest

        with pytest.raises(PermissionError):
            map_into_container(h, "../../etc/passwd", "/workspace")
