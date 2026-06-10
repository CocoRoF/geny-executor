"""2.2.0 Wave 2 — ``HostSelections.resolve`` contract tests.

Audit §3.5 flagged ``HostSelections.resolve`` as a decoy: zero library
call sites while the class docstring claimed the runtime intersected
selections at session boot. The 2.2.0 resolution is honesty, not
wiring: the docstring now states that *hosts* apply selections (the
name/id schemes belong to host registries — e.g. Geny's permission ids
are minted in ``service/permission/install.py``), and ``resolve`` is
the supported helper for that filtering. These tests pin the helper's
contract so host implementations (Geny mirrors the semantics) have a
library-side source of truth.
"""

from __future__ import annotations

from geny_executor.core.environment import HostSelections


class TestResolve:
    def test_wildcard_returns_every_available_name(self):
        assert HostSelections.resolve(["*"], ["a", "b", "c"]) == ["a", "b", "c"]

    def test_wildcard_returns_a_copy(self):
        available = ["a", "b"]
        out = HostSelections.resolve(["*"], available)
        out.append("c")
        assert available == ["a", "b"]

    def test_empty_selection_is_explicit_opt_out(self):
        assert HostSelections.resolve([], ["a", "b"]) == []

    def test_literal_selection_intersects(self):
        assert HostSelections.resolve(["a", "c"], ["a", "b", "c"]) == ["a", "c"]

    def test_unregistered_names_dropped_silently(self):
        """The manifest may outlive a host registration — stale names
        must not crash the runtime."""
        assert HostSelections.resolve(["a", "ghost"], ["a", "b"]) == ["a"]

    def test_selection_order_is_preserved(self):
        assert HostSelections.resolve(["c", "a"], ["a", "b", "c"]) == ["c", "a"]

    def test_star_mixed_with_names_is_literal_not_wildcard(self):
        """Only the exact ``["*"]`` sentinel is a wildcard; a list that
        *contains* ``"*"`` is treated literally (and ``"*"`` is dropped
        unless a host registered that name). Pinned so the subtle case
        has one documented answer."""
        assert HostSelections.resolve(["*", "a"], ["a", "b"]) == ["a"]

    def test_no_available_names(self):
        assert HostSelections.resolve(["*"], []) == []
        assert HostSelections.resolve(["a"], []) == []


class TestFromDictDefaults:
    def test_missing_payload_is_all_on(self):
        sel = HostSelections.from_dict(None)
        assert sel.hooks == ["*"]
        assert sel.skills == ["*"]
        assert sel.permissions == ["*"]

    def test_explicit_empty_lists_survive(self):
        sel = HostSelections.from_dict({"hooks": [], "skills": [], "permissions": []})
        assert sel.hooks == []
        assert sel.skills == []
        assert sel.permissions == []

    def test_partial_payload_defaults_remaining_sections(self):
        sel = HostSelections.from_dict({"hooks": ["h1"]})
        assert sel.hooks == ["h1"]
        assert sel.skills == ["*"]
        assert sel.permissions == ["*"]
