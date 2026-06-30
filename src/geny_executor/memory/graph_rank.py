"""Personalized PageRank over a memory knowledge-graph edge list.

Pure, dependency-free (no numpy) sparse power-iteration. Used to rank notes
by graph proximity to a set of seed notes (the vector/keyword hits), so
retrieval can surface graph-connected context — the "graph influences search"
half of the memory-graph design.

This is the standard Random-Walk-with-Restart / Personalized PageRank:

    r_{t+1} = alpha * s  +  (1 - alpha) * P^T r_t,     r_0 = s

where ``s`` is the L1-normalised personalization (seed) vector and ``P`` is the
row-normalised transition matrix of the (undirected, weighted) graph. The
restart term ``alpha * s`` is load-bearing: without it the walk converges to
query-independent eigenvector centrality (the same hubs for every query).
``alpha`` defaults to 0.5 (HippoRAG convention — heavier restart than classic
PageRank's 0.15 keeps the walk local to the seeds and converges in ~10-20
iterations).

Sparse adjacency-list iteration is O(iters * |E|) — cheap even for thousands
of edges, and never materialises an N*N matrix.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable


def personalized_pagerank(
    edges: Iterable[Dict[str, Any]],
    seeds: Dict[str, float],
    *,
    alpha: float = 0.5,
    max_iter: int = 20,
    tol: float = 1e-4,
) -> Dict[str, float]:
    """Return ``{node: ppr_score}`` for every node reachable in the graph.

    ``edges`` is an iterable of ``{"source", "target", "weight"}`` (undirected;
    each edge contributes both directions). ``seeds`` maps note id → base weight
    (e.g. a hit's relevance score); it is L1-normalised into the personalization
    vector. Returns ``{}`` when there is no usable seed mass.
    """
    adj: Dict[str, list] = {}
    deg: Dict[str, float] = {}
    nodes: set = set()
    for e in edges:
        u = e.get("source")
        v = e.get("target")
        try:
            w = float(e.get("weight", 1.0) or 0.0)
        except (TypeError, ValueError):
            w = 0.0
        if not u or not v or u == v or w <= 0.0:
            continue
        adj.setdefault(u, []).append((v, w))
        adj.setdefault(v, []).append((u, w))
        deg[u] = deg.get(u, 0.0) + w
        deg[v] = deg.get(v, 0.0) + w
        nodes.add(u)
        nodes.add(v)
    for n in seeds:
        nodes.add(n)
    if not nodes:
        return {}

    total = sum(max(0.0, float(x)) for x in seeds.values()) or 0.0
    if total <= 0.0:
        return {}
    s = {n: max(0.0, float(seeds.get(n, 0.0))) / total for n in nodes}

    r = dict(s)
    for _ in range(max(1, int(max_iter))):
        r_new = {n: alpha * s[n] for n in nodes}
        for u in nodes:
            ru = r[u]
            du = deg.get(u, 0.0)
            if ru == 0.0 or du == 0.0:
                continue
            share = (1.0 - alpha) * ru / du
            for v, w in adj.get(u, ()):  # type: ignore[union-attr]
                r_new[v] += share * w
        delta = 0.0
        for n in nodes:
            delta += abs(r_new[n] - r[n])
        r = r_new
        if delta < tol:
            break
    return r
