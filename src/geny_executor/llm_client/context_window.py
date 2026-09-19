"""How much context a model actually has — and how to find out without guessing.

``PipelineConfig.context_window_budget`` sizes two things that decide whether a
long conversation survives: Stage 2's proactive compaction (it fires at a
fraction of the budget) and Stage 4's ``TokenBudgetGuard`` (it compacts when
the next request would not leave enough headroom). Both are only as good as
the number they are given.

The shipped default is 200_000 — right for the frontier vendors and wrong by a
factor of six for a local server launched with a 32k window, where it means
compaction never fires and the request overflows before anything notices.

So the number is **resolved**, in this order, and a guess is never one of the
steps:

1. **Declared.** The operator says. Nobody else knows what sits behind a
   company gateway's address.
2. **Discovered.** The endpoint says. This is the important one, because the
   backends whose window we cannot know from a model id are exactly the ones
   that state it: an aggregator routing to hundreds of models
   (``context_length``), a vLLM server (``max_model_len`` — what it was
   *launched* with, which is usually below what the weights support), Ollama
   (``/api/show``), Google (``inputTokenLimit``).
3. **Known.** A short table of families whose window is a published, stable
   property of the model rather than of the deployment. It stays short on
   purpose: an entry here is a promise, and a wrong promise is worse than no
   entry, because the two failure directions are not symmetric — too high
   overflows the request, too low quietly throws away context the model could
   have used.
4. **Nothing.** ``None``, and the caller keeps its default. Deliberately not a
   fallback guess: a caller that knows it is unknown can say so to the user,
   which a number cannot.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

#: The pipeline's shipped default, re-exported so a host's fallback and the
#: library's default cannot drift apart.
DEFAULT_CONTEXT_WINDOW = 200_000

#: Model-id prefix → input window, for families where the window belongs to
#: the MODEL and not to the deployment.
#:
#: Read in longest-prefix-first order, so a more specific entry can override a
#: family. Everything absent from this table is measured or left unknown —
#: including every open-weight family (llama / qwen / mistral / …), whose real
#: window is whatever the server was started with and therefore knowable only
#: by asking it.
KNOWN_CONTEXT_WINDOWS: dict[str, int] = {
    # Anthropic has shipped a 200k input window across the Claude line for
    # long enough that it is a property of the family. Listed even though it
    # equals today's default: when the default moves, this floor must not.
    "claude": 200_000,
    # Google states it on /v1beta/models, so this is the offline fallback for
    # a host that has not run discovery yet.
    "gemini-2.5": 1_048_576,
    "gemini-1.5": 1_048_576,
    "gemini": 1_048_576,
}

#: CLI-style aliases a host may store instead of a canonical id. The Claude
#: Code account kinds use these, and they resolve to the same family window.
_ALIASES: dict[str, str] = {
    "sonnet": "claude",
    "opus": "claude",
    "haiku": "claude",
    "fable": "claude",
}


def known_context_window(model: str, *, provider: str = "") -> Optional[int]:
    """The published window for *model*, or ``None`` when we do not know.

    ``provider`` is accepted for future per-provider disambiguation (the same
    id can be served with different windows by different backends) and is
    currently only used to skip the table for the providers whose window is a
    deployment property — asking the table about a vLLM model would answer
    about the weights, not about the server.
    """
    name = (model or "").strip().lower()
    if not name:
        return None
    if provider in _DEPLOYMENT_DEFINED_PROVIDERS:
        # Whatever this id looks like, the window here belongs to the server,
        # not to the model. Better no answer than a confident wrong one.
        return None

    alias = _ALIASES.get(name)
    if alias:
        return KNOWN_CONTEXT_WINDOWS.get(alias)

    # An aggregator prefixes the vendor: ``anthropic/claude-sonnet-5``.
    if "/" in name:
        name = name.split("/", 1)[1]

    best: Optional[int] = None
    best_len = -1
    for prefix, window in KNOWN_CONTEXT_WINDOWS.items():
        if name.startswith(prefix) and len(prefix) > best_len:
            best, best_len = window, len(prefix)
    return best


#: Providers where the window is a property of the running server, so the
#: model id says nothing useful and only a probe can answer.
_DEPLOYMENT_DEFINED_PROVIDERS = frozenset({"vllm", "ollama", "lmstudio"})


def resolve_context_window(
    *,
    declared: Optional[int] = None,
    discovered: Optional[int] = None,
    model: str = "",
    provider: str = "",
) -> Optional[int]:
    """Resolve one hop's window by the precedence in this module's docstring.

    Returns ``None`` when no source knows, which the caller should surface
    rather than paper over — "we are assuming 200k for this model" is
    something an operator can act on, and a silent 200k is not.
    """
    for candidate in (declared, discovered):
        if candidate and candidate > 0:
            return int(candidate)
    return known_context_window(model, provider=provider)


def binding_context_window(windows: Iterable[Optional[int]]) -> Optional[int]:
    """The window a ROUTE can hold: the smallest of its hops.

    A route is one conversation that must be able to continue on any of its
    hops — that is the whole point of having fallbacks. So the history it may
    accumulate is bounded by the smallest window in it, not by the primary's:
    sizing to the primary means the failover, which happens exactly when
    things are already going wrong, walks into an overflow it cannot recover
    from.

    Hops that do not know their window are skipped rather than treated as
    unbounded — an unknown hop should not be able to raise the budget. All
    unknown → ``None``.
    """
    known = [int(w) for w in windows if w and w > 0]
    return min(known) if known else None


__all__ = [
    "DEFAULT_CONTEXT_WINDOW",
    "KNOWN_CONTEXT_WINDOWS",
    "known_context_window",
    "resolve_context_window",
    "binding_context_window",
]
