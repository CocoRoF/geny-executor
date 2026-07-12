"""Tool_use/tool_result pairing repairs (audit D4, 2.51.0)."""

from __future__ import annotations

import pytest

from geny_executor.core.message_repair import (
    repair_dangling_tool_calls,
    strip_leading_orphan_tool_results,
)


def _assistant_tool_use(tid: str):
    return {"role": "assistant", "content": [{"type": "tool_use", "id": tid, "name": "x", "input": {}}]}


def _tool_result(tid: str):
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": "ok"}]}


class TestRepairDangling:
    def test_dangling_tool_use_gets_synthetic_result(self):
        msgs = [{"role": "user", "content": "hi"}, _assistant_tool_use("t1")]
        n = repair_dangling_tool_calls(msgs)
        assert n == 1
        assert len(msgs) == 3
        block = msgs[2]["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "t1"
        assert block["is_error"] is True

    def test_answered_tool_use_untouched(self):
        msgs = [_assistant_tool_use("t1"), _tool_result("t1")]
        assert repair_dangling_tool_calls(msgs) == 0
        assert len(msgs) == 2

    def test_partial_answer_only_fills_missing(self):
        msgs = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "x", "input": {}},
                {"type": "tool_use", "id": "t2", "name": "y", "input": {}},
            ]},
            _tool_result("t1"),  # only t1 answered
        ]
        n = repair_dangling_tool_calls(msgs)
        assert n == 1
        # synthetic inserted right after the assistant (index 1), before the real result
        synthetic_ids = {
            b["tool_use_id"] for m in msgs for b in (m.get("content") or [])
            if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("is_error")
        }
        assert synthetic_ids == {"t2"}

    def test_no_tool_use_noop(self):
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        assert repair_dangling_tool_calls(msgs) == 0
        assert len(msgs) == 2

    def test_empty(self):
        msgs = []
        assert repair_dangling_tool_calls(msgs) == 0


class TestStripLeadingOrphans:
    def test_drops_leading_orphan_tool_result(self):
        window = [_tool_result("gone"), {"role": "assistant", "content": "hi"}]
        out = strip_leading_orphan_tool_results(window)
        assert out == window[1:]

    def test_keeps_clean_window(self):
        window = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        assert strip_leading_orphan_tool_results(window) is window

    def test_drops_multiple_leading(self):
        window = [_tool_result("a"), _tool_result("b"), {"role": "user", "content": "real"}]
        out = strip_leading_orphan_tool_results(window)
        assert out == window[2:]


class TestCompactorBoundary:
    @pytest.mark.asyncio
    async def test_truncate_never_starts_on_orphan_tool_result(self):
        from geny_executor.core.state import PipelineState
        from geny_executor.stages.s02_context.artifact.default.compactors import (
            TruncateCompactor,
        )

        state = PipelineState()
        # 10 messages; the message that would be first-in-window is a
        # tool_result whose tool_use is dropped.
        state.messages = [{"role": "user", "content": f"m{i}"} for i in range(7)]
        state.messages.append(_assistant_tool_use("t1"))
        state.messages.append(_tool_result("t1"))
        state.messages.append({"role": "user", "content": "latest"})
        # keep_last=2 would slice [tool_result(t1), latest] — tool_result's
        # tool_use (index 7) is dropped → orphan. Must be stripped.
        await TruncateCompactor(keep_last=2).compact(state)
        assert not (
            state.messages[0].get("role") == "user"
            and isinstance(state.messages[0].get("content"), list)
            and state.messages[0]["content"][0].get("type") == "tool_result"
        )
