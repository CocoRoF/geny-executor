"""Derive a rich, de-clumped edge set for the memory knowledge graph.

Pure functions over ``Note``-like objects — no I/O, no embeddings, no LLM,
no third-party deps. This is the executor-owned graph substrate so every host
renders the same graph and the same edges can later drive graph-aware
retrieval (Personalized PageRank).

Edge types (in priority order; a pair gets at most one edge):
  - ``wikilink``  explicit ``[[links]]`` between notes        (weight 1.0)
  - ``tag``       shared tag, IDF-weighted + de-clumped        (weight 0.5·idf)
  - ``semantic``  lexical TF-IDF cosine k-NN over title+body   (weight = cosine)

The ``semantic`` layer is the populator for vaults that contain no
user-authored wikilinks and only meta tags (e.g. auto-archived execution
notes) — it connects notes by content similarity. It is lexical (TF-IDF),
not embedding-based, so it works in every scope regardless of whether vector
indexing is enabled, at zero token cost. Embedding k-NN can replace/augment
it later without changing this contract.

De-clumping policy (mirrors the host fallback so on/off-executor renders match):
meta-tag denylist + per-tag document-frequency cutoff + per-node fanout cap,
plus a per-term DF cutoff on the TF-IDF side that drops boilerplate terms
appearing in most notes (the thing that would otherwise connect everything).
"""

from __future__ import annotations

import heapq
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional

# Meta tags on (nearly) every archived note — never form tag edges.
META_TAG_DENYLIST = {
    "conversation", "user_chat", "assistant_chat", "agent_dm", "dm", "dms",
    "compaction", "system-artifact", "system", "auto", "automated",
    "log", "logs", "chat", "session", "archive", "execution", "execution-summary",
    "insight", "insights", "memory", "note", "daily", "digest",
}
TAG_DF_RATIO_MAX = 0.33
TAG_DF_ABS_FLOOR = 12
TAG_FANOUT_MAX = 6

# Lexical TF-IDF k-NN params.
KNN_TOP_K = 6
KNN_MIN_SIM = 0.10
# A term appearing in more than this fraction of notes is boilerplate — it
# would link everything to everything, so it is dropped from the TF-IDF space.
TERM_DF_RATIO_MAX = 0.40
TERM_DF_ABS_FLOOR = 20

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{1,}")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "of", "to",
    "in", "on", "at", "by", "for", "with", "as", "is", "are", "was", "were",
    "be", "been", "being", "this", "that", "these", "those", "it", "its",
    "i", "you", "he", "she", "we", "they", "them", "his", "her", "their",
    "from", "into", "out", "up", "down", "over", "under", "no", "not", "do",
    "does", "did", "done", "have", "has", "had", "will", "would", "can",
    "could", "should", "may", "might", "must", "so", "than", "too", "very",
    "just", "about", "there", "here", "what", "which", "who", "when", "where",
    "how", "all", "any", "each", "more", "most", "some", "such", "only", "own",
    "same", "also", "via", "per", "etc",
}


def _tokenize(text: str) -> List[str]:
    return [
        t for t in (m.group(0).lower() for m in _TOKEN_RE.finditer(text or ""))
        if t not in _STOPWORDS
    ]


def _semantic_edges(items: List[Dict[str, Any]], *, top_k: int, min_sim: float):
    """Yield (a_fn, b_fn, cosine) lexical-similarity neighbours via a sparse
    inverted-index cosine over normalised TF-IDF vectors. O(sum df²) over
    *kept* terms — bounded because boilerplate terms are dropped."""
    n_docs = max(1, len(items))
    tfs = [Counter(_tokenize(it["title"] + " " + it["body"])) for it in items]

    df: Counter = Counter()
    for tf in tfs:
        for term in tf:
            df[term] += 1
    term_df_cut = max(TERM_DF_ABS_FLOOR, int(TERM_DF_RATIO_MAX * n_docs))
    idf = {
        term: math.log(n_docs / (1 + d)) + 1.0
        for term, d in df.items()
        if 2 <= d <= term_df_cut
    }

    vecs: List[Dict[str, float]] = []
    for tf in tfs:
        v: Dict[str, float] = {}
        for term, c in tf.items():
            w = idf.get(term)
            if w is not None:
                v[term] = (1.0 + math.log(c)) * w
        norm = math.sqrt(sum(x * x for x in v.values()))
        vecs.append({t: x / norm for t, x in v.items()} if norm else {})

    inv: Dict[str, List[tuple]] = {}
    for i, v in enumerate(vecs):
        for term, w in v.items():
            inv.setdefault(term, []).append((i, w))

    for i in range(len(items)):
        scores: Dict[int, float] = {}
        for term, wi in vecs[i].items():
            for j, wj in inv.get(term, ()):  # type: ignore[misc]
                if j != i:
                    scores[j] = scores.get(j, 0.0) + wi * wj
        if not scores:
            continue
        for j, sim in heapq.nlargest(top_k, scores.items(), key=lambda kv: kv[1]):
            if sim >= min_sim:
                yield items[i]["fn"], items[j]["fn"], sim


def derive_graph_edges(
    notes,
    *,
    top_k: int = KNN_TOP_K,
    min_sim: float = KNN_MIN_SIM,
    enable_semantic: bool = True,
) -> List[Dict[str, Any]]:
    """Return a de-duplicated edge list for the graph.

    ``notes`` is any iterable of objects exposing ``.ref.filename``, ``.title``,
    ``.body``, ``.tags`` and ``.links_out`` (the executor ``Note``). Returns
    ``[{source, target, type, weight, label?}]`` with at most one edge per
    unordered pair (wikilink > tag > semantic).
    """
    items: List[Dict[str, Any]] = []
    for n in notes:
        items.append({
            "fn": n.ref.filename,
            "title": n.title or "",
            "body": n.body or "",
            "tags": [str(t) for t in (getattr(n, "tags", None) or [])],
            "links_out": [str(t) for t in (getattr(n, "links_out", None) or [])],
        })
    files = {it["fn"] for it in items}
    n_docs = max(1, len(items))
    edges: List[Dict[str, Any]] = []
    seen: set = set()

    def _key(a: str, b: str):
        return (a, b) if a <= b else (b, a)

    def _add(a: str, b: str, etype: str, weight: float, label: Optional[str] = None) -> None:
        if a == b or a not in files or b not in files:
            return
        k = _key(a, b)
        if k in seen:
            return
        seen.add(k)
        edge: Dict[str, Any] = {"source": a, "target": b, "type": etype, "weight": round(float(weight), 3)}
        if label is not None:
            edge["label"] = label
        edges.append(edge)

    # 1. wikilink (directed source→target, but deduped by unordered pair)
    for it in items:
        for tgt in it["links_out"]:
            _add(it["fn"], tgt, "wikilink", 1.0)

    # 2. de-clumped IDF tag edges
    tag_to_files: Dict[str, List[str]] = {}
    for it in items:
        for tag in it["tags"]:
            tag_to_files.setdefault(tag, []).append(it["fn"])
    node_tag_degree: Dict[str, int] = {}
    df_max = max(TAG_DF_ABS_FLOOR, int(TAG_DF_RATIO_MAX * n_docs))
    for tag, fns in tag_to_files.items():
        d = len(fns)
        if d < 2 or d >= n_docs or d > df_max:
            continue
        if tag.lower().lstrip("#") in META_TAG_DENYLIST:
            continue
        weight = 0.5 * (math.log((1 + n_docs) / (1 + d)))
        if weight <= 0:
            continue
        for i in range(len(fns)):
            a = fns[i]
            if node_tag_degree.get(a, 0) >= TAG_FANOUT_MAX:
                continue
            for j in range(i + 1, len(fns)):
                b = fns[j]
                if node_tag_degree.get(b, 0) >= TAG_FANOUT_MAX:
                    continue
                if _key(a, b) in seen:
                    continue
                _add(a, b, "tag", weight, tag)
                node_tag_degree[a] = node_tag_degree.get(a, 0) + 1
                node_tag_degree[b] = node_tag_degree.get(b, 0) + 1
                if node_tag_degree[a] >= TAG_FANOUT_MAX:
                    break

    # 3. lexical TF-IDF k-NN (the populator for link/tag-less vaults)
    if enable_semantic and n_docs >= 2:
        for a, b, sim in _semantic_edges(items, top_k=top_k, min_sim=min_sim):
            _add(a, b, "semantic", sim)

    return edges
