"""``NoteGraph`` query helpers (EXEC-4)."""

from __future__ import annotations

from geny_executor.memory.provider import (
    Importance,
    NoteGraph,
    NoteMeta,
    NoteRef,
    Scope,
)


def _ref(name: str) -> NoteRef:
    return NoteRef(filename=name, scope=Scope.SESSION, category="topics", backend="test")


def _meta(name: str, *, tags=None) -> NoteMeta:
    return NoteMeta(
        ref=_ref(name),
        title=name,
        importance=Importance.MEDIUM,
        tags=list(tags or []),
        category="topics",
    )


def _graph(edges, nodes=None):
    return NoteGraph(
        nodes=[_meta(n) for n in (nodes or [])],
        edges=list(edges),
    )


def test_neighbours_returns_one_hop_targets():
    g = _graph([("a.md", "b.md"), ("a.md", "c.md"), ("b.md", "c.md")])
    assert sorted(g.neighbours("a.md")) == ["b.md", "c.md"]
    assert g.neighbours("c.md") == []


def test_k_hop_walks_levels_in_bfs_order():
    g = _graph(
        [
            ("a.md", "b.md"),
            ("a.md", "c.md"),
            ("b.md", "d.md"),
            ("c.md", "d.md"),
            ("d.md", "e.md"),
        ]
    )
    assert g.k_hop("a.md", 1) == ["b.md", "c.md"]
    # k=2 reaches d (and not e) — d is shared by b and c.
    assert sorted(g.k_hop("a.md", 2)) == ["b.md", "c.md", "d.md"]
    # k=3 reaches e too
    assert sorted(g.k_hop("a.md", 3)) == ["b.md", "c.md", "d.md", "e.md"]


def test_k_hop_zero_returns_empty():
    g = _graph([("a.md", "b.md")])
    assert g.k_hop("a.md", 0) == []


def test_connected_component_treats_edges_as_undirected():
    g = _graph(
        [
            ("a.md", "b.md"),
            ("c.md", "b.md"),  # back-edge into b
            ("d.md", "e.md"),  # disjoint
        ]
    )
    cc = g.connected_component("a.md")
    assert cc == {"a.md", "b.md", "c.md"}
    cc2 = g.connected_component("d.md")
    assert cc2 == {"d.md", "e.md"}


def test_connected_component_isolated_node_known_to_graph():
    g = _graph([], nodes=["lonely.md"])
    assert g.connected_component("lonely.md") == {"lonely.md"}


def test_connected_component_unknown_node_returns_empty():
    g = _graph([("a.md", "b.md")])
    assert g.connected_component("z.md") == set()


def test_linked_chain_finds_shortest_path():
    g = _graph(
        [
            ("a.md", "b.md"),
            ("a.md", "c.md"),
            ("c.md", "d.md"),
            ("b.md", "d.md"),
            ("d.md", "e.md"),
        ]
    )
    chain = g.linked_chain("a.md", "e.md")
    assert chain is not None
    assert chain[0] == "a.md" and chain[-1] == "e.md"
    # Both candidate routes (a→b→d→e and a→c→d→e) are length 4; either is acceptable.
    assert len(chain) == 4


def test_linked_chain_returns_self_when_endpoints_match():
    g = _graph([("a.md", "b.md")])
    assert g.linked_chain("a.md", "a.md") == ["a.md"]


def test_linked_chain_no_path_returns_none():
    g = _graph([("a.md", "b.md"), ("c.md", "d.md")])
    assert g.linked_chain("a.md", "d.md") is None


def test_notes_with_tag_case_insensitive():
    g = NoteGraph(
        nodes=[
            _meta("a.md", tags=["Roadmap", "Q3"]),
            _meta("b.md", tags=["q3"]),
            _meta("c.md", tags=["other"]),
        ],
        edges=[],
    )
    assert sorted(g.notes_with_tag("q3")) == ["a.md", "b.md"]
    assert g.notes_with_tag("missing") == []
    assert g.notes_with_tag("") == []
