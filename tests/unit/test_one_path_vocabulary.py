"""A tool answers in the address the session uses.

Two providers on one session drifted into two vocabularies: one wrote
`/workspace/x.txt`, the other sent the host path. Both landed correctly — the
mapper handles either — but the result echoed back whatever was sent, so the
model that used the host path kept using it and taught the rest of the
conversation to. The tools speak the container's address, which is the one
the prompt names and the only one every tool accepts.
"""

from geny_executor.tools._sandbox import spoken_path


class Handle:
    """A sandbox whose bind maps <host>/agents/s1 → /workspace."""

    container_workdir = "/workspace"

    @staticmethod
    def map_path(p: str):
        host = "/data/geny_agent_sessions/_cloud/u/workspace/agents/s1"
        if p == host:
            return "/workspace"
        if p.startswith(host + "/"):
            return "/workspace/" + p[len(host) + 1 :]
        return None


def test_a_host_path_is_answered_in_the_container_address():
    said = spoken_path(
        Handle(),
        "/data/geny_agent_sessions/_cloud/u/workspace/agents/s1/notes.md",
        "/workspace",
    )
    assert said == "/workspace/notes.md"


def test_a_relative_path_is_answered_in_full():
    assert spoken_path(Handle(), "notes.md", "/workspace") == "/workspace/notes.md"


def test_the_container_address_is_answered_unchanged():
    assert spoken_path(Handle(), "/workspace/notes.md", "/workspace") == "/workspace/notes.md"


def test_an_unmappable_path_is_echoed_rather_than_raising():
    """A result that says nothing is worse than one that says something odd."""

    class Hostile:
        container_workdir = "/workspace"

        @staticmethod
        def map_path(_p):
            raise RuntimeError("no mapping table")

    assert spoken_path(Hostile(), "/elsewhere/x.txt", "/workspace") == "/elsewhere/x.txt"


def test_no_sandbox_at_all_is_survivable():
    assert spoken_path(None, "notes.md", "/workspace") == "/workspace/notes.md"
