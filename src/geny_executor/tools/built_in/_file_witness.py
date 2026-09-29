"""Which files the agent has seen — so ``Write`` never blindly replaces one.

The reference harnesses' basic contract is "read a file before you change
it". This one had none: ``Write`` could truncate an existing file — a
document the user uploaded, an earlier session's work — without the agent
ever having seen what was in it, and the log showed only a successful write.
That hole cannot be counted after the fact; only the contract closes it.

The ledger is one ``state.shared`` key holding paths only (both the form the
model gave and the resolved one). Checking contents on every write would be
a remote round trip on the sandbox path; what this prevents — erasing what
you have not seen — is caught by the path alone, the same way on both paths.
Hosts that start each turn with a fresh state carry the key over themselves.
"""

from __future__ import annotations

import os
import posixpath
from typing import Any, Dict, List

#: ``state.shared`` key. The ``executor.`` prefix is required — Stage 10
#: drops mutations outside the known namespaces (XGEN shipped this as
#: ``file.witnessed`` and the ledger was never written once).
WITNESSED_KEY = "executor.file_witnessed"

#: Oldest entries are forgotten past this; a forgotten file just needs
#: another read.
MAX_ENTRIES = 500


def path_forms(file_path: str, context: Any) -> List[str]:
    """The path as given and as resolved against the working directory."""
    if not file_path:
        return []
    forms = [file_path]
    if getattr(context, "sandbox", None) is not None:
        wd = getattr(context, "working_dir", None) or "/workspace"
        forms.append(posixpath.normpath(posixpath.join(wd, file_path)))
    else:
        wd = getattr(context, "working_dir", None) or os.getcwd()
        forms.append(os.path.realpath(os.path.join(wd, os.path.expanduser(file_path))))
    return list(dict.fromkeys(forms))


def _book(state_view: Any) -> List[str]:
    shared = getattr(state_view, "shared", None)
    seen = shared.get(WITNESSED_KEY) if isinstance(shared, dict) else None
    return [p for p in seen if isinstance(p, str)] if isinstance(seen, list) else []


def witnessed_mutation(paths: List[str]) -> Dict[str, Any]:
    """The ``state_mutations`` entry that adds ``paths`` to the ledger.

    Only the new paths travel; Stage 10 merges them into the ledger
    (``merge_witnessed``) so parallel reads do not overwrite each other.
    """
    return {WITNESSED_KEY: list(dict.fromkeys(p for p in paths if p))}


def merge_witnessed(current: Any, added: Any) -> List[str]:
    book = [p for p in current if isinstance(p, str)] if isinstance(current, list) else []
    new = [p for p in added if isinstance(p, str)] if isinstance(added, list) else []
    book = [p for p in book if p not in new] + new
    return book[-MAX_ENTRIES:]


def is_witnessed(state_view: Any, paths: List[str]) -> bool:
    book = set(_book(state_view))
    return any(p in book for p in paths)


def refusal(path: str) -> str:
    """What the model is told — the next move has to be obvious."""
    return (
        f"{path} already exists and you have not read it in this session. Nothing was "
        f"written — a blind Write would discard whatever is in it. Read it first, then use "
        f"Edit to change part of it, or Write again to replace it knowingly."
    )


__all__ = [
    "MAX_ENTRIES",
    "WITNESSED_KEY",
    "is_witnessed",
    "merge_witnessed",
    "path_forms",
    "refusal",
    "witnessed_mutation",
]
