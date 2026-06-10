"""2.2.0 permission posture tests (audit 2026-06-09 §1-5).

``default_posture`` makes deny-by-default reachable via config:
honoured when no rule matches AND when zero rules are bound. The
default stays ``ALLOW`` for 2.x back-compat; 3.0 flips it.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import pytest

from geny_executor.permission import (
    PermissionBehavior,
    PermissionDecision,
    PermissionMode,
    PermissionPolicy,
    PermissionPosture,
    PermissionRule,
    PermissionSource,
    coerce_posture,
    evaluate_permission,
    load_hierarchical_policy,
    load_permission_policy,
    parse_permission_policy,
)


class _FakeTool:
    def __init__(self, name: str = "Bash"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def prepare_permission_matcher(self, inp: Dict[str, Any]) -> Callable[[str], bool]:
        return lambda pattern: True


def _rule(
    tool: str,
    behavior: PermissionBehavior,
    *,
    source: PermissionSource = PermissionSource.PROJECT,
) -> PermissionRule:
    return PermissionRule(tool_name=tool, behavior=behavior, source=source)


# ── Matrix: posture honoured ─────────────────────────────────────────


class TestMatrixPosture:
    @pytest.mark.asyncio
    async def test_no_rules_deny_posture_denies(self):
        decision = await evaluate_permission(
            tool=_FakeTool(),
            tool_input={},
            rules=[],
            default_posture=PermissionPosture.DENY,
        )
        assert decision.behavior is PermissionBehavior.DENY
        assert "posture" in (decision.reason or "")

    @pytest.mark.asyncio
    async def test_no_rules_allow_posture_allows_back_compat(self):
        decision = await evaluate_permission(
            tool=_FakeTool(),
            tool_input={},
            rules=[],
            default_posture=PermissionPosture.ALLOW,
        )
        assert decision.behavior is PermissionBehavior.ALLOW

    @pytest.mark.asyncio
    async def test_default_kwarg_is_allow_for_back_compat(self):
        # Omitting the kwarg must behave exactly as 2.1.x did.
        decision = await evaluate_permission(tool=_FakeTool(), tool_input={}, rules=[])
        assert decision.behavior is PermissionBehavior.ALLOW

    @pytest.mark.asyncio
    async def test_explicit_allow_rule_beats_deny_posture(self):
        # deny posture + no rules = deny everything EXCEPT the allowlist.
        decision = await evaluate_permission(
            tool=_FakeTool("Bash"),
            tool_input={},
            rules=[_rule("Bash", PermissionBehavior.ALLOW)],
            default_posture=PermissionPosture.DENY,
        )
        assert decision.behavior is PermissionBehavior.ALLOW

    @pytest.mark.asyncio
    async def test_unmatched_rule_with_deny_posture_denies(self):
        decision = await evaluate_permission(
            tool=_FakeTool("Bash"),
            tool_input={},
            rules=[_rule("OtherTool", PermissionBehavior.ALLOW)],
            default_posture=PermissionPosture.DENY,
        )
        assert decision.behavior is PermissionBehavior.DENY

    @pytest.mark.asyncio
    async def test_bypass_mode_beats_deny_posture(self):
        # BYPASS is the developer escape hatch; it beats deny rules
        # today and beats the posture for symmetry.
        decision = await evaluate_permission(
            tool=_FakeTool(),
            tool_input={},
            rules=[],
            mode=PermissionMode.BYPASS,
            default_posture=PermissionPosture.DENY,
        )
        assert decision.behavior is PermissionBehavior.ALLOW

    @pytest.mark.asyncio
    async def test_deny_posture_skips_tool_fallback(self):
        # A tool vouching for itself is not an allowlist entry —
        # the fallback must not be consulted under DENY posture.
        called = []

        async def fallback(inp):
            called.append(inp)
            return PermissionDecision.allow(reason="tool says yes")

        decision = await evaluate_permission(
            tool=_FakeTool(),
            tool_input={},
            rules=[],
            fallback=fallback,
            default_posture=PermissionPosture.DENY,
        )
        assert decision.behavior is PermissionBehavior.DENY
        assert called == []

    @pytest.mark.asyncio
    async def test_allow_posture_still_consults_fallback(self):
        async def fallback(inp):
            return PermissionDecision.deny(reason="secret scanner")

        decision = await evaluate_permission(
            tool=_FakeTool(),
            tool_input={},
            rules=[],
            fallback=fallback,
            default_posture=PermissionPosture.ALLOW,
        )
        assert decision.behavior is PermissionBehavior.DENY
        assert decision.reason == "secret scanner"

    @pytest.mark.asyncio
    async def test_plan_escalation_takes_precedence_over_posture(self):
        # PLAN + destructive → ASK (a chance for HITL) even under DENY
        # posture — ASK is strictly more conservative than allow.
        decision = await evaluate_permission(
            tool=_FakeTool(),
            tool_input={},
            rules=[],
            mode=PermissionMode.PLAN,
            capabilities_destructive=True,
            default_posture=PermissionPosture.DENY,
        )
        assert decision.behavior is PermissionBehavior.ASK


# ── coerce_posture ───────────────────────────────────────────────────


class TestCoercePosture:
    def test_none_defaults_to_allow(self):
        assert coerce_posture(None) is PermissionPosture.ALLOW

    def test_enum_passthrough(self):
        assert coerce_posture(PermissionPosture.DENY) is PermissionPosture.DENY

    def test_string_coercion_case_insensitive(self):
        assert coerce_posture("DENY") is PermissionPosture.DENY
        assert coerce_posture(" allow ") is PermissionPosture.ALLOW

    def test_garbage_warns_and_falls_back_to_allow(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            assert coerce_posture("denny") is PermissionPosture.ALLOW
        assert any("permission posture" in r.message for r in caplog.records)

    def test_explicit_default_override(self):
        assert coerce_posture(None, default=PermissionPosture.DENY) is PermissionPosture.DENY


# ── Loader: posture rides the same YAML the rules do ─────────────────


class TestPolicyLoader:
    def test_parse_policy_with_deny_posture(self):
        data = {
            "default_posture": "deny",
            "allow": [{"tool": "Read", "pattern": "*"}],
        }
        policy = parse_permission_policy(data, source=PermissionSource.PROJECT)
        assert isinstance(policy, PermissionPolicy)
        assert policy.default_posture is PermissionPosture.DENY
        assert policy.posture_declared is True
        assert len(policy.rules) == 1
        assert policy.rules[0].tool_name == "Read"

    def test_parse_policy_without_posture_is_allow_undeclared(self):
        policy = parse_permission_policy(
            {"allow": [{"tool": "Read"}]}, source=PermissionSource.USER
        )
        assert policy.default_posture is PermissionPosture.ALLOW
        assert policy.posture_declared is False

    def test_parse_policy_invalid_posture_raises(self):
        # A typo'd 'deny' silently becoming 'allow' would be a masked
        # security downgrade — the loader must refuse loudly.
        with pytest.raises(ValueError, match="default_posture"):
            parse_permission_policy(
                {"default_posture": "denny"}, source=PermissionSource.PROJECT
            )

    def test_load_policy_missing_file_is_empty_allow(self, tmp_path):
        policy = load_permission_policy(
            tmp_path / "nope.yaml", source=PermissionSource.PROJECT
        )
        assert policy.rules == []
        assert policy.default_posture is PermissionPosture.ALLOW
        assert policy.posture_declared is False

    def test_load_policy_yaml_roundtrip(self, tmp_path):
        path = tmp_path / "permissions.yaml"
        path.write_text(
            "default_posture: deny\n"
            "allow:\n"
            "  - { tool: Read, pattern: '*' }\n",
            encoding="utf-8",
        )
        policy = load_permission_policy(path, source=PermissionSource.PROJECT)
        assert policy.default_posture is PermissionPosture.DENY
        assert len(policy.rules) == 1

    def test_hierarchical_posture_silence_does_not_override(self, tmp_path):
        # local file adds a rule but stays silent on posture — the
        # project-level explicit deny must survive.
        local = tmp_path / "local.yaml"
        local.write_text("allow:\n  - { tool: Bash, pattern: 'git *' }\n", encoding="utf-8")
        project = tmp_path / "project.yaml"
        project.write_text("default_posture: deny\n", encoding="utf-8")

        policy = load_hierarchical_policy(local_path=local, project_path=project)
        assert policy.default_posture is PermissionPosture.DENY
        assert policy.posture_declared is True
        assert len(policy.rules) == 1

    def test_hierarchical_higher_priority_declared_posture_wins(self, tmp_path):
        local = tmp_path / "local.yaml"
        local.write_text("default_posture: allow\n", encoding="utf-8")
        project = tmp_path / "project.yaml"
        project.write_text("default_posture: deny\n", encoding="utf-8")

        policy = load_hierarchical_policy(local_path=local, project_path=project)
        assert policy.default_posture is PermissionPosture.ALLOW

    def test_hierarchical_no_declarations_default_allow(self, tmp_path):
        user = tmp_path / "user.yaml"
        user.write_text("allow:\n  - { tool: Read }\n", encoding="utf-8")
        policy = load_hierarchical_policy(user_path=user)
        assert policy.default_posture is PermissionPosture.ALLOW
        assert policy.posture_declared is False
