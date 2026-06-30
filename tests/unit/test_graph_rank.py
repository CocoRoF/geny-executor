"""Unit tests for Personalized PageRank over the memory graph (pure)."""

from __future__ import annotations

from geny_executor.memory.graph_rank import personalized_pagerank


def _edges(*pairs):
    return [{"source": a, "target": b, "weight": w} for a, b, w in pairs]


def test_chain_mass_decays_with_distance():
    edges = _edges(("a", "b", 1.0), ("b", "c", 1.0), ("c", "d", 1.0))
    r = personalized_pagerank(edges, {"a": 1.0}, alpha=0.5)
    assert r["a"] > r["b"] > r["c"] > r["d"]


def test_seed_stays_query_local_to_its_cluster():
    edges = _edges(("x1", "x2", 1.0), ("x2", "x3", 1.0), ("y1", "y2", 1.0))
    r = personalized_pagerank(edges, {"x1": 1.0}, alpha=0.5)
    assert min(r["x2"], r["x3"]) > max(r.get("y1", 0.0), r.get("y2", 0.0))


def test_higher_alpha_is_more_local():
    edges = _edges(("a", "b", 1.0), ("b", "c", 1.0))
    near = personalized_pagerank(edges, {"a": 1.0}, alpha=0.8)
    far = personalized_pagerank(edges, {"a": 1.0}, alpha=0.2)
    # heavier restart (0.8) keeps more mass on the seed than a light restart
    assert near["a"] > far["a"]


def test_weight_matters():
    # a connects to b (weak) and c (strong); c should receive more mass
    edges = _edges(("a", "b", 0.2), ("a", "c", 2.0))
    r = personalized_pagerank(edges, {"a": 1.0}, alpha=0.5)
    assert r["c"] > r["b"]


def test_empty_and_no_seed_safe():
    # a lone seed with no edges still ranks itself (restart mass)
    assert "a" in personalized_pagerank([], {"a": 1.0})
    assert personalized_pagerank(_edges(("a", "b", 1.0)), {}) == {}
    assert personalized_pagerank(_edges(("a", "b", 1.0)), {"a": 0.0}) == {}


def test_self_loops_and_bad_weights_ignored():
    edges = _edges(("a", "a", 1.0), ("a", "b", 0.0), ("a", "c", 1.0))
    r = personalized_pagerank(edges, {"a": 1.0}, alpha=0.5)
    # only a—c is a valid edge; b never linked so absent or zero
    assert r["c"] > 0.0
    assert r.get("b", 0.0) == 0.0
