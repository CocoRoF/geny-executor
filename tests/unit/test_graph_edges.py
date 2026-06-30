"""Unit tests for the knowledge-graph edge derivation (pure, no I/O).

Covers the three edge layers and the de-clumping policy:
  - wikilink edges (and their priority over tag/semantic on the same pair)
  - IDF-weighted, de-clumped tag edges (denylist, df cutoff, fanout cap)
  - lexical TF-IDF k-NN ``semantic`` edges (the populator for link/tag-less vaults)
"""

from __future__ import annotations

from types import SimpleNamespace

from geny_executor.memory.providers.file.graph_edges import (
    TAG_FANOUT_MAX,
    derive_graph_edges,
)


def _note(fn, title="", body="", tags=None, links=None):
    return SimpleNamespace(
        ref=SimpleNamespace(filename=fn),
        title=title,
        body=body,
        tags=list(tags or []),
        links_out=list(links or []),
    )


def _pairset(edges, etype=None):
    return {
        frozenset((e["source"], e["target"]))
        for e in edges
        if etype is None or e["type"] == etype
    }


def test_wikilink_edges_resolved_and_typed():
    notes = [
        _note("a.md", "A", "x", links=["b.md", "missing.md"]),
        _note("b.md", "B", "y"),
    ]
    edges = derive_graph_edges(notes, enable_semantic=False)
    wl = [e for e in edges if e["type"] == "wikilink"]
    assert len(wl) == 1  # a->b kept, a->missing dropped (target absent)
    assert wl[0]["source"] == "a.md" and wl[0]["target"] == "b.md"
    assert wl[0]["weight"] == 1.0


def test_meta_tag_denylist_and_universal_dropped():
    # every note shares #conversation (denylist) + #execution-summary (denylist)
    notes = [_note(f"n{i}.md", body=f"body {i}", tags=["conversation"]) for i in range(5)]
    edges = derive_graph_edges(notes, enable_semantic=False)
    assert _pairset(edges, "tag") == set()


def test_real_tag_cluster_kept_in_small_vault():
    # #neowiz on 3 of 5 is a real cluster (a bare 0.3*N ratio would wrongly drop it)
    notes = []
    for i in range(5):
        tags = ["conversation"] + (["neowiz"] if i < 3 else [])
        notes.append(_note(f"n{i}.md", body=f"text {i}", tags=tags))
    edges = derive_graph_edges(notes, enable_semantic=False)
    tag_pairs = _pairset(edges, "tag")
    assert tag_pairs == {
        frozenset(("n0.md", "n1.md")),
        frozenset(("n0.md", "n2.md")),
        frozenset(("n1.md", "n2.md")),
    }
    for e in edges:
        if e["type"] == "tag":
            assert e["label"] == "neowiz"
            assert 0.0 < e["weight"] < 0.5  # IDF-scaled below the 0.5 base


def test_tag_fanout_cap_bounds_degree():
    # #proj on 13 of 40 notes: under the df cutoff (kept) but without a fanout
    # cap it would be C(13,2)=78 edges with per-node degree 12.
    notes = [_note(f"k{i}.md", body=f"b{i}", tags=(["proj"] if i < 13 else []))
             for i in range(40)]
    edges = derive_graph_edges(notes, enable_semantic=False)
    deg: dict = {}
    for e in edges:
        if e["type"] == "tag":
            deg[e["source"]] = deg.get(e["source"], 0) + 1
            deg[e["target"]] = deg.get(e["target"], 0) + 1
    assert deg, "expected some tag edges"
    assert max(deg.values()) <= TAG_FANOUT_MAX


def test_semantic_knn_connects_similar_notes():
    notes = [
        _note("e1.md", "Exec 1", "neowiz email IMAP polling incoming messages parsing"),
        _note("e2.md", "Exec 2", "neowiz email IMAP headers sender filters parsing messages"),
        _note("e3.md", "Exec 3", "avatar Live2D Pixi Spine rendering pipeline"),
        _note("e4.md", "Exec 4", "avatar Spine Pixi texture atlas animation blending"),
    ]
    sem = _pairset(derive_graph_edges(notes), "semantic")
    assert frozenset(("e1.md", "e2.md")) in sem  # email cluster
    assert frozenset(("e3.md", "e4.md")) in sem  # avatar cluster


def test_wikilink_beats_semantic_on_same_pair():
    notes = [
        _note("a.md", "A", "postgres index tuning vacuum analyze", links=["b.md"]),
        _note("b.md", "B", "postgres index tuning vacuum analyze query plan"),
    ]
    edges = derive_graph_edges(notes)
    pair = [e for e in edges if frozenset((e["source"], e["target"])) == frozenset(("a.md", "b.md"))]
    assert len(pair) == 1 and pair[0]["type"] == "wikilink"


def test_empty_and_single_note_safe():
    assert derive_graph_edges([]) == []
    assert derive_graph_edges([_note("solo.md", "S", "lonely note")]) == []
