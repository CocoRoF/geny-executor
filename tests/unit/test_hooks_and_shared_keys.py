"""Tests for HookEvent taxonomy + SharedKeys namespace.

Phase 1 Week 2 Checkpoint 3.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from geny_executor.core.shared_keys import SharedKeys
from geny_executor.hooks import HookEvent, HookEventPayload, HookOutcome


# ─────────────────────────────────────────────────────────────────
# HookEvent enum
# ─────────────────────────────────────────────────────────────────


class TestHookEventEnum:
    def test_values_are_snake_case_strings(self) -> None:
        for member in HookEvent:
            assert isinstance(member.value, str)
            assert member.value == member.value.lower()
            assert " " not in member.value

    def test_expected_members_present(self) -> None:
        expected = {
            "SESSION_START", "SESSION_END",
            "PIPELINE_START", "PIPELINE_END",
            "STAGE_ENTER", "STAGE_EXIT",
            "USER_PROMPT_SUBMIT",
            "PRE_TOOL_USE", "POST_TOOL_USE", "POST_TOOL_FAILURE",
            "PERMISSION_REQUEST", "PERMISSION_DENIED",
            "LOOP_ITERATION_END",
            "CWD_CHANGED",
            "MCP_SERVER_STATE",
            "NOTIFICATION",
        }
        actual = {m.name for m in HookEvent}
        assert expected <= actual

    def test_enum_member_is_string(self) -> None:
        assert HookEvent.PRE_TOOL_USE.value == "pre_tool_use"
        # StrEnum behaviour — comparing to plain str works
        assert HookEvent.PRE_TOOL_USE == "pre_tool_use"


# ─────────────────────────────────────────────────────────────────
# HookEventPayload
# ─────────────────────────────────────────────────────────────────


class TestHookEventPayload:
    def test_minimum_fields_serialize(self) -> None:
        p = HookEventPayload(
            event=HookEvent.SESSION_START,
            session_id="sess-1",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        d = p.to_json_dict()
        assert d["event"] == "session_start"
        assert d["session_id"] == "sess-1"
        assert "timestamp" in d
        assert d["permission_mode"] == "default"
        # Optional fields omitted when None
        assert "stage_order" not in d
        assert "tool_name" not in d

    def test_tool_event_serialization(self) -> None:
        p = HookEventPayload(
            event=HookEvent.PRE_TOOL_USE,
            session_id="s",
            timestamp="2026-04-24T00:00:00Z",
            stage_order=10,
            stage_name="tool",
            tool_name="Bash",
            tool_input={"command": "git status"},
            details={"extra": "data"},
        )
        d = p.to_json_dict()
        assert d["event"] == "pre_tool_use"
        assert d["stage_order"] == 10
        assert d["stage_name"] == "tool"
        assert d["tool_name"] == "Bash"
        assert d["tool_input"] == {"command": "git status"}
        assert d["details"] == {"extra": "data"}

    def test_empty_details_is_skipped(self) -> None:
        p = HookEventPayload(
            event=HookEvent.NOTIFICATION,
            session_id="s",
            timestamp="t",
        )
        assert "details" not in p.to_json_dict()


# ─────────────────────────────────────────────────────────────────
# HookOutcome
# ─────────────────────────────────────────────────────────────────


class TestHookOutcome:
    def test_passthrough_defaults(self) -> None:
        o = HookOutcome.passthrough()
        assert o.continue_ is True
        assert o.blocked is False
        assert o.decision is None

    def test_block_helper(self) -> None:
        o = HookOutcome.block("nope")
        assert o.continue_ is False
        assert o.blocked is True
        assert o.decision == "block"
        assert o.stop_reason == "nope"

    def test_approve_helper(self) -> None:
        o = HookOutcome.approve("ok")
        assert o.continue_ is True
        assert o.decision == "approve"
        assert o.stop_reason == "ok"

    def test_from_response_minimal(self) -> None:
        o = HookOutcome.from_response({"continue": True})
        assert o.continue_ is True
        assert o.decision is None

    def test_from_response_full(self) -> None:
        o = HookOutcome.from_response({
            "continue": False,
            "suppress_output": True,
            "decision": "block",
            "stop_reason": "forbidden",
            "modified_input": {"x": 1},
            "hook_specific_output": {"k": "v"},
        })
        assert o.blocked is True
        assert o.suppress_output is True
        assert o.decision == "block"
        assert o.modified_input == {"x": 1}
        assert o.hook_specific_output == {"k": "v"}

    def test_from_response_tolerates_unknown_keys(self) -> None:
        # Forward compat: hook scripts may emit extra fields
        o = HookOutcome.from_response({"continue": True, "extra": "ignored"})
        assert o.continue_ is True

    def test_from_response_wrong_types_fallback(self) -> None:
        # decision must be a string; bad types → None
        o = HookOutcome.from_response({"decision": 42})
        assert o.decision is None
        # stop_reason same
        o2 = HookOutcome.from_response({"stop_reason": ["list"]})
        assert o2.stop_reason is None

    def test_combine_block_wins_over_passthrough(self) -> None:
        a = HookOutcome.passthrough()
        b = HookOutcome.block("audit rejects")
        merged = a.combine(b)
        assert merged.blocked is True
        assert merged.stop_reason == "audit rejects"

        merged2 = b.combine(a)
        assert merged2.blocked is True

    def test_combine_block_beats_approve(self) -> None:
        a = HookOutcome.approve("ok")
        b = HookOutcome.block("nope")
        merged = a.combine(b)
        assert merged.decision == "block"

    def test_combine_suppress_output_or(self) -> None:
        a = HookOutcome(continue_=True, suppress_output=True)
        b = HookOutcome.passthrough()
        assert a.combine(b).suppress_output is True
        assert b.combine(a).suppress_output is True

    def test_combine_modified_input_last_writer_wins(self) -> None:
        a = HookOutcome(modified_input={"x": 1})
        b = HookOutcome(modified_input={"y": 2})
        assert a.combine(b).modified_input == {"y": 2}
        assert b.combine(a).modified_input == {"x": 1}

    def test_combine_hook_specific_output_merged(self) -> None:
        a = HookOutcome(hook_specific_output={"k1": "a"})
        b = HookOutcome(hook_specific_output={"k2": "b"})
        merged = a.combine(b)
        assert merged.hook_specific_output == {"k1": "a", "k2": "b"}


# ─────────────────────────────────────────────────────────────────
# SharedKeys
# ─────────────────────────────────────────────────────────────────


class TestSharedKeys:
    def test_executor_keys_namespaced(self) -> None:
        assert SharedKeys.TOOL_CALL_ID.startswith("executor.")
        assert SharedKeys.SKILL_CTX.startswith("executor.")
        assert SharedKeys.TOOL_REVIEW_FLAGS.startswith("executor.")
        assert SharedKeys.HITL_REQUEST.startswith("executor.")
        assert SharedKeys.TURN_SUMMARY.startswith("executor.")

    def test_memory_keys_namespaced(self) -> None:
        assert SharedKeys.MEMORY_CONTEXT_CHUNKS.startswith("memory.")
        assert SharedKeys.MEMORY_NEEDS_REFLECTION.startswith("memory.")

    def test_geny_keys_namespaced(self) -> None:
        assert SharedKeys.GENY_CREATURE_STATE.startswith("geny.")
        assert SharedKeys.GENY_MUTATION_BUFFER.startswith("geny.")
        assert SharedKeys.GENY_CREATURE_ROLE.startswith("geny.")

    def test_keys_are_unique(self) -> None:
        # All public string attributes
        vals = [
            v for k, v in vars(SharedKeys).items()
            if isinstance(v, str) and not k.startswith("_")
        ]
        assert len(vals) == len(set(vals)), "duplicate SharedKeys values"

    def test_plugin_key_format(self) -> None:
        assert SharedKeys.plugin_key("myplugin", "state") == "plugin.myplugin.state"
        assert SharedKeys.plugin_key("abc123", "cfg") == "plugin.abc123.cfg"

    def test_plugin_key_rejects_invalid_namespace(self) -> None:
        with pytest.raises(ValueError):
            SharedKeys.plugin_key("", "k")
        with pytest.raises(ValueError):
            SharedKeys.plugin_key("has space", "k")
        with pytest.raises(ValueError):
            SharedKeys.plugin_key("has.dot", "k")
        with pytest.raises(ValueError):
            SharedKeys.plugin_key("1starts-with-digit", "k")

    def test_plugin_key_rejects_empty_key(self) -> None:
        with pytest.raises(ValueError):
            SharedKeys.plugin_key("ns", "")
