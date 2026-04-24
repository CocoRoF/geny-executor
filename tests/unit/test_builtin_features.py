"""Phase 3 Week 5 — built-in tool feature-flag gate tests."""

from __future__ import annotations

import pytest

from geny_executor.tools.built_in import (
    BUILT_IN_TOOL_CLASSES,
    BUILT_IN_TOOL_FEATURES,
    get_builtin_tools,
)


class TestFeatureMap:
    def test_every_tool_belongs_to_exactly_one_feature(self):
        """Every name in the registry must appear in exactly one feature
        bucket — the feature map is the navigation surface for hosts
        selecting by capability."""
        all_names = set(BUILT_IN_TOOL_CLASSES.keys())
        seen = {}
        for feature, names in BUILT_IN_TOOL_FEATURES.items():
            for name in names:
                assert name in all_names, (
                    f"feature {feature!r} references unknown tool {name!r}"
                )
                assert name not in seen, (
                    f"tool {name!r} listed in both {seen[name]!r} and {feature!r}"
                )
                seen[name] = feature
        assert set(seen) == all_names, (
            f"these tools have no feature mapping: {all_names - set(seen)}"
        )


class TestGetBuiltinTools:
    def test_no_args_returns_all(self):
        tools = get_builtin_tools()
        assert set(tools.keys()) == set(BUILT_IN_TOOL_CLASSES.keys())

    def test_returns_fresh_dict(self):
        a = get_builtin_tools()
        b = get_builtin_tools()
        assert a is not b
        a["Read"] = None  # type: ignore[assignment]
        assert get_builtin_tools()["Read"] is BUILT_IN_TOOL_CLASSES["Read"]

    def test_features_filesystem(self):
        tools = get_builtin_tools(features=["filesystem"])
        assert set(tools.keys()) == {
            "Read",
            "Write",
            "Edit",
            "Glob",
            "Grep",
            "NotebookEdit",
        }

    def test_features_web(self):
        tools = get_builtin_tools(features=["web"])
        assert set(tools.keys()) == {"WebFetch", "WebSearch"}

    def test_features_shell(self):
        tools = get_builtin_tools(features=["shell"])
        assert set(tools.keys()) == {"Bash"}

    def test_multiple_features_union(self):
        tools = get_builtin_tools(features=["filesystem", "web"])
        expected = {
            "Read",
            "Write",
            "Edit",
            "Glob",
            "Grep",
            "NotebookEdit",
            "WebFetch",
            "WebSearch",
        }
        assert set(tools.keys()) == expected

    def test_unknown_feature_raises(self):
        with pytest.raises(KeyError) as exc:
            get_builtin_tools(features=["nonexistent"])
        assert "nonexistent" in str(exc.value)

    def test_names_only(self):
        tools = get_builtin_tools(names=["Read", "WebFetch"])
        assert set(tools.keys()) == {"Read", "WebFetch"}

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError) as exc:
            get_builtin_tools(names=["NotAThing"])
        assert "NotAThing" in str(exc.value)

    def test_features_and_names_combined(self):
        tools = get_builtin_tools(features=["web"], names=["Read"])
        assert set(tools.keys()) == {"Read", "WebFetch", "WebSearch"}

    def test_empty_features_returns_empty(self):
        # Empty iterable is a valid (if useless) request — mustn't fall
        # through to the "no args → all" branch.
        tools = get_builtin_tools(features=[])
        assert tools == {}

    def test_empty_names_returns_empty(self):
        tools = get_builtin_tools(names=[])
        assert tools == {}

    def test_feature_union_deduplicates(self):
        # Even if the caller passes the same feature twice, the result
        # should still be a clean set.
        tools = get_builtin_tools(features=["filesystem", "filesystem"])
        assert set(tools.keys()) == {
            "Read",
            "Write",
            "Edit",
            "Glob",
            "Grep",
            "NotebookEdit",
        }

    def test_returned_values_are_tool_classes(self):
        tools = get_builtin_tools(features=["filesystem"])
        for name, cls in tools.items():
            assert isinstance(cls, type)
            # Instantiate to confirm it's a usable Tool subclass
            instance = cls()
            assert instance.name == name


class TestCatalogSize:
    def test_catalog_has_at_least_phase_3_week_5_tools(self):
        """Plan §3.2 validation: ``len(get_builtin_tools()) >= 15`` by
        end of Phase 3. Week 5 only ships 8 (pre-existing 6 + WebFetch
        + WebSearch), but we already pin the floor here so the assertion
        surfaces any accidental regression."""
        assert len(get_builtin_tools()) >= 8
