"""Permission rule matrix — Phase 1 Week 2 Checkpoint 2 tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest

from geny_executor.permission import (
    PermissionBehavior,
    PermissionDecision,
    PermissionMode,
    PermissionRule,
    PermissionSource,
    SOURCE_PRIORITY,
    evaluate_permission,
    load_permission_rules,
    parse_permission_rules,
)


# ─────────────────────────────────────────────────────────────────
# Fake Tool that mimics Tool.prepare_permission_matcher
# ─────────────────────────────────────────────────────────────────


class _FakeTool:
    def __init__(self, name: str, input_pattern_hook: Callable[[Dict[str, Any]], Callable[[str], bool]] | None = None):
        self._name = name
        self._hook = input_pattern_hook

    @property
    def name(self) -> str:
        return self._name

    async def prepare_permission_matcher(self, inp: Dict[str, Any]) -> Callable[[str], bool]:
        if self._hook is not None:
            return self._hook(inp)
        # Default exact-name matcher
        return lambda pattern: pattern == self._name


def _bash_matcher(inp: Dict[str, Any]) -> Callable[[str], bool]:
    """Fake matcher: applies fnmatch on the stored command.

    Pattern is already the sub-pattern (tool_name is stored separately
    on the rule), so we do NOT expect ``"Bash(…)"`` wrapping here.
    """
    import fnmatch

    cmd = inp.get("command", "")

    def _match(pattern: str) -> bool:
        return fnmatch.fnmatch(cmd, pattern)

    return _match


# ─────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────


class TestTypes:
    def test_rule_matches_name(self) -> None:
        r = PermissionRule(tool_name="Bash", behavior=PermissionBehavior.ALLOW, source=PermissionSource.USER)
        assert r.matches_name("Bash") is True
        assert r.matches_name("Other") is False

    def test_wildcard_rule_matches_any_name(self) -> None:
        r = PermissionRule(tool_name="*", behavior=PermissionBehavior.DENY, source=PermissionSource.PROJECT)
        assert r.matches_name("Bash") is True
        assert r.matches_name("Read") is True

    def test_rule_is_frozen(self) -> None:
        r = PermissionRule(tool_name="Bash", behavior=PermissionBehavior.ALLOW, source=PermissionSource.USER)
        with pytest.raises(Exception):
            r.tool_name = "other"  # type: ignore[misc]

    def test_decision_helpers(self) -> None:
        a = PermissionDecision.allow(reason="ok")
        assert a.behavior is PermissionBehavior.ALLOW
        d = PermissionDecision.deny(reason="nope")
        assert d.behavior is PermissionBehavior.DENY
        k = PermissionDecision.ask(reason="confirm?")
        assert k.behavior is PermissionBehavior.ASK

    def test_source_priority_order(self) -> None:
        # CLI > LOCAL > PROJECT > USER > PRESET
        assert SOURCE_PRIORITY[0] is PermissionSource.CLI_ARG
        assert SOURCE_PRIORITY[-1] is PermissionSource.PRESET_DEFAULT


# ─────────────────────────────────────────────────────────────────
# evaluate_permission
# ─────────────────────────────────────────────────────────────────


class TestEvaluatePermission:
    def test_bypass_mode_always_allows(self) -> None:
        tool = _FakeTool("Bash", _bash_matcher)
        rules = [
            PermissionRule(tool_name="Bash", behavior=PermissionBehavior.DENY, source=PermissionSource.USER),
        ]
        d = asyncio.run(evaluate_permission(
            tool=tool,
            tool_input={"command": "rm -rf /"},
            rules=rules,
            mode=PermissionMode.BYPASS,
        ))
        assert d.behavior is PermissionBehavior.ALLOW

    def test_exact_name_allow_default_mode(self) -> None:
        tool = _FakeTool("Read")
        rules = [
            PermissionRule(tool_name="Read", behavior=PermissionBehavior.ALLOW, source=PermissionSource.USER),
        ]
        d = asyncio.run(evaluate_permission(
            tool=tool,
            tool_input={"path": "/etc/passwd"},
            rules=rules,
            mode=PermissionMode.DEFAULT,
        ))
        assert d.behavior is PermissionBehavior.ALLOW
        assert d.matched_rule is not None

    def test_input_pattern_match(self) -> None:
        tool = _FakeTool("Bash", _bash_matcher)
        rules = [
            PermissionRule(tool_name="Bash", pattern="git *", behavior=PermissionBehavior.ALLOW, source=PermissionSource.USER),
            PermissionRule(tool_name="Bash", pattern="rm -rf *", behavior=PermissionBehavior.DENY, source=PermissionSource.USER),
        ]
        ok = asyncio.run(evaluate_permission(
            tool=tool,
            tool_input={"command": "git status"},
            rules=rules,
        ))
        assert ok.behavior is PermissionBehavior.ALLOW

        bad = asyncio.run(evaluate_permission(
            tool=tool,
            tool_input={"command": "rm -rf /"},
            rules=rules,
        ))
        assert bad.behavior is PermissionBehavior.DENY

    def test_source_priority_cli_beats_user(self) -> None:
        tool = _FakeTool("Bash", _bash_matcher)
        rules = [
            PermissionRule(tool_name="Bash", behavior=PermissionBehavior.DENY, source=PermissionSource.USER, reason="user deny"),
            PermissionRule(tool_name="Bash", behavior=PermissionBehavior.ALLOW, source=PermissionSource.CLI_ARG, reason="cli allow"),
        ]
        d = asyncio.run(evaluate_permission(
            tool=tool,
            tool_input={"command": "echo hi"},
            rules=rules,
        ))
        assert d.behavior is PermissionBehavior.ALLOW
        assert d.matched_rule is not None
        assert d.matched_rule.source is PermissionSource.CLI_ARG

    def test_wildcard_tool_name_matches(self) -> None:
        tool = _FakeTool("Read")
        rules = [
            PermissionRule(tool_name="*", behavior=PermissionBehavior.ALLOW, source=PermissionSource.PRESET_DEFAULT, reason="global allow"),
        ]
        d = asyncio.run(evaluate_permission(
            tool=tool,
            tool_input={"path": "x"},
            rules=rules,
        ))
        assert d.behavior is PermissionBehavior.ALLOW

    def test_plan_mode_escalates_destructive_to_ask(self) -> None:
        tool = _FakeTool("Bash", _bash_matcher)
        rules: List[PermissionRule] = []  # no rules at all
        d = asyncio.run(evaluate_permission(
            tool=tool,
            tool_input={"command": "rm -rf /tmp/x"},
            rules=rules,
            mode=PermissionMode.PLAN,
            capabilities_destructive=True,
        ))
        assert d.behavior is PermissionBehavior.ASK

    def test_plan_mode_explicit_allow_beats_escalation(self) -> None:
        tool = _FakeTool("Bash", _bash_matcher)
        rules = [
            PermissionRule(tool_name="Bash", pattern="rm -rf *", behavior=PermissionBehavior.ALLOW, source=PermissionSource.CLI_ARG),
        ]
        d = asyncio.run(evaluate_permission(
            tool=tool,
            tool_input={"command": "rm -rf /tmp/x"},
            rules=rules,
            mode=PermissionMode.PLAN,
            capabilities_destructive=True,
        ))
        assert d.behavior is PermissionBehavior.ALLOW

    def test_no_rule_no_destructive_defaults_allow(self) -> None:
        tool = _FakeTool("Read")
        d = asyncio.run(evaluate_permission(
            tool=tool,
            tool_input={"path": "x"},
            rules=[],
        ))
        assert d.behavior is PermissionBehavior.ALLOW

    def test_fallback_invoked_when_no_rule(self) -> None:
        tool = _FakeTool("Custom")
        calls = {"n": 0}

        async def _fb(inp: Dict[str, Any]) -> PermissionDecision:
            calls["n"] += 1
            return PermissionDecision.deny(reason="tool-level deny")

        d = asyncio.run(evaluate_permission(
            tool=tool,
            tool_input={},
            rules=[],
            fallback=_fb,
        ))
        assert d.behavior is PermissionBehavior.DENY
        assert calls["n"] == 1

    def test_rule_match_skips_fallback(self) -> None:
        tool = _FakeTool("Custom")
        calls = {"n": 0}

        async def _fb(inp: Dict[str, Any]) -> PermissionDecision:
            calls["n"] += 1
            return PermissionDecision.deny()

        rules = [
            PermissionRule(tool_name="Custom", behavior=PermissionBehavior.ALLOW, source=PermissionSource.USER),
        ]
        d = asyncio.run(evaluate_permission(
            tool=tool,
            tool_input={},
            rules=rules,
            fallback=_fb,
        ))
        assert d.behavior is PermissionBehavior.ALLOW
        assert calls["n"] == 0


# ─────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────


class TestParseRules:
    def test_parse_all_sections(self) -> None:
        data = {
            "allow": [
                {"tool": "Read", "pattern": "*"},
                {"tool": "Bash", "pattern": "git *", "reason": "needed"},
            ],
            "deny": [
                {"tool": "Bash", "pattern": "rm -rf *"},
            ],
            "ask": [
                {"tool": "Edit", "pattern": "*"},
            ],
        }
        rules = parse_permission_rules(data, source=PermissionSource.USER)
        assert len(rules) == 4
        assert rules[0].tool_name == "Read"
        assert rules[0].behavior is PermissionBehavior.ALLOW
        assert rules[1].reason == "needed"
        assert rules[2].tool_name == "Bash"
        assert rules[2].behavior is PermissionBehavior.DENY
        assert rules[3].behavior is PermissionBehavior.ASK
        assert all(r.source is PermissionSource.USER for r in rules)

    def test_parse_empty_sections(self) -> None:
        assert parse_permission_rules({}, source=PermissionSource.USER) == []
        assert parse_permission_rules({"allow": []}, source=PermissionSource.USER) == []

    def test_parse_rejects_bad_shape(self) -> None:
        with pytest.raises(ValueError):
            parse_permission_rules({"allow": "not a list"}, source=PermissionSource.USER)
        with pytest.raises(ValueError):
            parse_permission_rules({"allow": [{"not_tool": "x"}]}, source=PermissionSource.USER)
        with pytest.raises(ValueError):
            parse_permission_rules({"allow": ["not a dict"]}, source=PermissionSource.USER)


class TestLoadRules:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        rules = load_permission_rules(tmp_path / "missing.yaml", source=PermissionSource.USER)
        assert rules == []

    def test_load_json_file(self, tmp_path: Path) -> None:
        p = tmp_path / "perms.json"
        p.write_text(json.dumps({
            "allow": [{"tool": "Read", "pattern": "*"}],
            "deny": [{"tool": "Bash", "pattern": "rm -rf *"}],
        }))
        rules = load_permission_rules(p, source=PermissionSource.PROJECT)
        assert len(rules) == 2
        assert rules[0].source is PermissionSource.PROJECT

    def test_load_yaml_file_if_available(self, tmp_path: Path) -> None:
        try:
            import yaml  # type: ignore  # noqa: F401
        except ImportError:
            pytest.skip("PyYAML not installed")
        p = tmp_path / "perms.yaml"
        p.write_text(
            "allow:\n"
            "  - tool: Read\n"
            "    pattern: '*'\n"
            "deny:\n"
            "  - tool: Bash\n"
            "    pattern: 'rm -rf *'\n"
        )
        rules = load_permission_rules(p, source=PermissionSource.USER)
        assert len(rules) == 2
        assert {r.behavior for r in rules} == {PermissionBehavior.ALLOW, PermissionBehavior.DENY}
