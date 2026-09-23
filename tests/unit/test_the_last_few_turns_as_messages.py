"""What the agent is given about the turns just before this one.

Geny starts every turn from a fresh ``PipelineState``, so ``state.messages``
is empty and the only thing carrying the conversation was the retriever's
``recent_turns`` layer. That layer counted STM **rows** (one tool use makes a
turn four rows, so ``recent(6)`` was about one and a half turns), kept only
text blocks (dropping every ``tool_use`` and ``tool_result``), and rendered
the survivors into the system prompt's ``# Relevant Knowledge``.

Three consequences, all reported: finished work re-run, completion not
recognised, and three-turn-old statements treated as facts on a par with the
user's pinned notes.

This module rebuilds the recent conversation as messages. What is pinned
here is the part that made the difference: tool evidence survives, turns are
counted as turns, and the newest turn is the last thing to go.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List


from geny_executor.core.token_estimate import estimate_text_tokens
from geny_executor.memory.short_term_window import (
    DEFAULT_MAX_TOKENS,
    MAX_RESULT_TOKENS,
    MIN_MAX_TOKENS,
    MIN_RESULT_TOKENS,
    WindowConfig,
    build_window,
    group_logical_turns,
    window_token_budget,
)


@dataclass
class Row:
    """One STM row, the shape ``stm.recent()`` returns."""

    role: str
    content: Any


def _turn(instruction: str, answer: str, *, tool: str = "", target: str = "",
          result: str = "ok", failed: bool = False, uid: str = "t1") -> List[Row]:
    """A logical turn: instruction, optional tool round trip, answer."""
    rows = [Row("user", [{"type": "text", "text": instruction}])]
    if tool:
        rows.append(Row("assistant", [
            {"type": "tool_use", "id": uid, "name": tool,
             "input": {"file_path": target} if target else {}},
        ]))
        block = {"type": "tool_result", "tool_use_id": uid, "content": result}
        if failed:
            block["is_error"] = True
        rows.append(Row("user", [block]))
    rows.append(Row("assistant", [{"type": "text", "text": answer}]))
    return rows


def _busy_turn(label: str, *, calls: int, result_chars: int, uid: str) -> List[Row]:
    """A turn where the agent worked — many calls, each with a real result.

    This is the shape that puts a window under pressure. One huge result does
    not: the per-result cap trims it during assembly, long before the budget
    ladder is consulted.
    """
    rows = [Row("user", [{"type": "text", "text": f"{label} 지시"}])]
    for i in range(calls):
        rows.append(Row("assistant", [{"type": "tool_use", "id": f"{uid}-{i}",
                                       "name": "Read", "input": {"file_path": f"{label}{i}.txt"}}]))
        rows.append(Row("user", [{"type": "tool_result", "tool_use_id": f"{uid}-{i}",
                                  "content": "R" * result_chars}]))
    rows.append(Row("assistant", [{"type": "text", "text": f"{label} 완료"}]))
    return rows


def _text(messages) -> str:
    out = []
    for m in messages:
        for b in m["content"]:
            if b.get("type") == "text":
                out.append(str(b.get("text")))
            elif b.get("type") == "tool_result":
                out.append(str(b.get("content")))
            elif b.get("type") == "tool_use":
                out.append(f"{b.get('name')}({b.get('input')})")
    return "\n".join(out)


class TestATurnIsATurn:
    """The old layer counted rows. One tool use makes a turn four rows, so
    asking for six 'turns' delivered about one and a half."""

    def test_six_rows_of_one_tool_turn_is_one_turn(self) -> None:
        rows = _turn("do a", "did a", tool="Read", target="a.txt")
        assert len(rows) == 4
        assert len(group_logical_turns(rows, 6)) == 1

    def test_a_tool_result_does_not_start_a_turn(self) -> None:
        """It is a user message, which is exactly how the miscount happened."""
        rows = _turn("do a", "did a", tool="Bash")
        turns = group_logical_turns(rows, 6)
        assert len(turns) == 1
        assert len(turns[0].messages) == 4

    def test_five_instructions_give_five_turns_however_many_tools(self) -> None:
        rows: List[Row] = []
        for i in range(5):
            rows += _turn(f"step {i}", f"done {i}", tool="Read",
                          target=f"f{i}.txt", uid=f"u{i}")
        assert len(rows) == 20
        assert len(group_logical_turns(rows, 5)) == 5

    def test_a_fragment_before_the_first_instruction_is_dropped(self) -> None:
        """An answer with no instruction in front of it reads as a claim."""
        rows = [Row("assistant", [{"type": "text", "text": "orphan"}])]
        rows += _turn("real", "answer")
        turns = group_logical_turns(rows, 5)
        assert len(turns) == 1
        assert "orphan" not in _text(build_window(rows, WindowConfig()).messages)


class TestTheNearestTurnsKeepTheirTools:
    """The evidence that work is finished lives in tool results. Dropping
    them is why the agent could not tell."""

    def _window(self, **kw):
        rows: List[Row] = []
        for i in range(5):
            rows += _turn(f"step {i}", f"done {i}", tool="Read",
                          target=f"f{i}.txt", result=f"CONTENT-{i}", uid=f"u{i}")
        return build_window(rows, WindowConfig(**kw))

    def test_the_tool_result_survives(self) -> None:
        body = _text(self._window().messages)
        assert "CONTENT-4" in body, "the last turn lost its tool evidence"
        assert "CONTENT-3" in body

    def test_the_tool_call_survives_with_its_id(self) -> None:
        messages = self._window().messages
        uses = [b for m in messages for b in m["content"] if b.get("type") == "tool_use"]
        results = [b for m in messages for b in m["content"] if b.get("type") == "tool_result"]
        assert {u["id"] for u in uses} == {r["tool_use_id"] for r in results}

    def test_five_turns_are_represented(self) -> None:
        result = self._window()
        assert result.turns == 5
        assert (result.full, result.dialogue) == (2, 3)


class TestTheFartherTurnsKeepTheConversation:
    def test_their_tool_blocks_are_gone(self) -> None:
        rows: List[Row] = []
        for i in range(5):
            rows += _turn(f"step {i}", f"done {i}", tool="Read",
                          target=f"f{i}.txt", result=f"CONTENT-{i}", uid=f"u{i}")
        body = _text(build_window(rows, WindowConfig()).messages)
        assert "CONTENT-0" not in body, "a far turn kept its tool result"
        assert "step 0" in body and "done 0" in body

    def test_but_what_they_did_is_named_with_its_target(self) -> None:
        """``Read ×3`` cannot answer "have I already read that file"."""
        rows = _turn("look", "looked", tool="Read", target="/w/inv.txt", uid="a")
        rows += _turn("x", "y") + _turn("p", "q") + _turn("r", "s")
        body = _text(build_window(rows, WindowConfig()).messages)
        assert "[used tools: Read(inv.txt)]" in body

    def test_a_failure_is_named(self) -> None:
        """"it is done" and "it was attempted" is the distinction missed."""
        rows = _turn("try", "tried", tool="Bash", result="boom", failed=True, uid="a")
        rows += _turn("x", "y") + _turn("p", "q") + _turn("r", "s")
        body = _text(build_window(rows, WindowConfig()).messages)
        assert "(1 failed)" in body

    def test_repeats_collapse_with_a_count(self) -> None:
        rows = [Row("user", [{"type": "text", "text": "go"}])]
        for i in range(3):
            rows.append(Row("assistant", [{"type": "tool_use", "id": f"u{i}",
                                           "name": "Bash", "input": {"command": "ls"}}]))
            rows.append(Row("user", [{"type": "tool_result",
                                      "tool_use_id": f"u{i}", "content": "ok"}]))
        rows.append(Row("assistant", [{"type": "text", "text": "done"}]))
        rows += _turn("x", "y") + _turn("p", "q") + _turn("r", "s")
        body = _text(build_window(rows, WindowConfig()).messages)
        assert "Bash(ls) ×3" in body

    def test_the_line_can_be_turned_off(self) -> None:
        rows = _turn("look", "looked", tool="Read", target="a.txt", uid="a")
        rows += _turn("x", "y") + _turn("p", "q") + _turn("r", "s")
        body = _text(build_window(rows, WindowConfig(used_tools_line=False)).messages)
        assert "used tools" not in body


class TestWhatIsNotReplayed:
    def test_thinking_is_dropped(self) -> None:
        """Bound to the model that produced it; another hop rejects it."""
        rows = [
            Row("user", [{"type": "text", "text": "hi"}]),
            Row("assistant", [
                {"type": "thinking", "thinking": "SECRET", "signature": "s"},
                {"type": "text", "text": "hello"},
            ]),
        ]
        body = _text(build_window(rows, WindowConfig()).messages)
        assert "SECRET" not in body and "hello" in body

    def test_an_image_becomes_a_note(self) -> None:
        rows = [
            Row("user", [{"type": "text", "text": "look"}]),
            Row("assistant", [{"type": "tool_use", "id": "u", "name": "Shot", "input": {}}]),
            Row("user", [{"type": "tool_result", "tool_use_id": "u", "content": "shot"}]),
            Row("assistant", [
                {"type": "image", "source": {"type": "base64", "data": "AAAA"}},
                {"type": "text", "text": "saw it"},
            ]),
        ]
        body = _text(build_window(rows, WindowConfig()).messages)
        assert "AAAA" not in body
        assert "[image from an earlier turn]" in body


class TestThePairsAreNeverSplit:
    def test_a_call_with_no_result_gets_one(self) -> None:
        """A turn that ended mid-loop. The wire rejects the dangling call."""
        rows = [
            Row("user", [{"type": "text", "text": "go"}]),
            Row("assistant", [{"type": "tool_use", "id": "u1", "name": "Bash", "input": {}}]),
        ]
        messages = build_window(rows, WindowConfig()).messages
        results = [b for m in messages for b in m["content"] if b.get("type") == "tool_result"]
        assert [r["tool_use_id"] for r in results] == ["u1"]

    def test_the_window_never_opens_on_an_orphan_result(self) -> None:
        rows = [
            Row("user", [{"type": "tool_result", "tool_use_id": "gone", "content": "x"}]),
            Row("user", [{"type": "text", "text": "go"}]),
            Row("assistant", [{"type": "text", "text": "ok"}]),
        ]
        messages = build_window(rows, WindowConfig()).messages
        first = messages[0]["content"][0]
        assert first.get("type") != "tool_result"


class TestTheBudget:
    """Five turns within the budget; the newest one always.

    The budget is a share of what the route can hold, in tokens. It used to
    be 40,000 characters flat — about 10k tokens of English and 40k of
    Korean, so one number meant four different windows in four languages,
    and a third of a 32k local model either way.
    """

    def test_unknown_window_gets_the_common_case(self) -> None:
        assert window_token_budget(None) == DEFAULT_MAX_TOKENS

    def test_a_share_of_the_effective_window(self) -> None:
        """The answer has to fit in the same window: 15% of what is left
        after the output reservation, the tail Hermes protects."""
        assert window_token_budget(200_000, reserved_output=32_000) == 25_200
        assert window_token_budget(32_768, reserved_output=8_192) == 3_686

    def test_a_large_window_is_not_capped_at_a_small_constant(self) -> None:
        """A 1M route keeping 10k tokens of replay throws away the evidence
        the window exists to keep."""
        assert window_token_budget(1_000_000, reserved_output=32_000) == 145_200

    def test_a_reservation_larger_than_the_window_is_ignored(self) -> None:
        assert window_token_budget(8_000, reserved_output=64_000) == MIN_MAX_TOKENS

    def test_it_never_goes_below_a_usable_floor(self) -> None:
        assert window_token_budget(1_000) == MIN_MAX_TOKENS

    def test_the_budget_means_the_same_in_korean_and_english(self) -> None:
        """Measured in tokens, a Korean result is cut to far fewer characters
        than an English one — because it costs that much more."""
        ko = _turn("go", "done", tool="Read", target="f", result="가" * 60_000, uid="k")
        en = _turn("go", "done", tool="Read", target="f", result="a" * 60_000, uid="e")
        cfg = WindowConfig(max_tokens=20_000)
        ko_body = _text(build_window(ko, cfg).messages)
        en_body = _text(build_window(en, cfg).messages)
        assert ko_body.count("가") < en_body.count("a") / 2
        for rows in (ko, en):
            assert build_window(rows, cfg).tokens <= 20_000

    def test_one_result_cannot_take_the_window(self) -> None:
        """A 200 KB file read keeps a quarter of the window, not all of it —
        and never more than the absolute per-result cap."""
        cfg = WindowConfig(max_tokens=1_000_000)
        assert cfg.result_cap() == MAX_RESULT_TOKENS
        assert WindowConfig(max_tokens=20_000).result_cap() == 5_000

    def test_everything_fits_nothing_is_touched(self) -> None:
        rows: List[Row] = []
        for i in range(5):
            rows += _busy_turn(f"s{i}", calls=2, result_chars=1_000, uid=f"u{i}")
        result = build_window(rows, WindowConfig(max_tokens=25_000))
        assert result.degraded == []
        assert (result.full, result.dialogue) == (2, 3)
        assert "R" * 1_000 in _text(result.messages)

    def test_the_older_full_turn_sheds_results_before_the_newest(self) -> None:
        """Oldest first, as ``clear_tool_uses`` clears oldest first."""
        rows: List[Row] = []
        for i in range(5):
            rows += _busy_turn(f"s{i}", calls=4, result_chars=6_000, uid=f"u{i}")
        result = build_window(rows, WindowConfig(max_tokens=12_000))
        results = {
            b["tool_use_id"]: str(b["content"])
            for m in result.messages
            for b in m["content"]
            if b.get("type") == "tool_result"
        }
        assert "trimmed" in results["u3-0"]
        assert "R" * 6_000 in results["u4-0"], "the newest turn lost evidence first"

    def test_bulk_goes_before_structure(self) -> None:
        """Results shrink before any turn is dropped: that a call happened is
        what stops it being redone; its tail rarely is."""
        rows: List[Row] = []
        for i in range(5):
            rows += _busy_turn(f"s{i}", calls=8, result_chars=5_000, uid=f"u{i}")
        result = build_window(rows, WindowConfig(max_tokens=8_000))
        assert result.degraded[0] == "result_tokens"
        first_drop = next(
            (i for i, step in enumerate(result.degraded) if step == "drop_dialogue_turn"),
            len(result.degraded),
        )
        assert "result_tokens" not in result.degraded[first_drop:]
        assert result.turns == 5 and result.tokens <= 8_000

    def test_every_tool_call_in_the_full_turns_survives_pressure(self) -> None:
        rows: List[Row] = []
        for i in range(5):
            rows += _busy_turn(f"s{i}", calls=6, result_chars=5_000, uid=f"u{i}")
        result = build_window(rows, WindowConfig(max_tokens=4_000))
        ids = {
            b["id"] for m in result.messages for b in m["content"] if b.get("type") == "tool_use"
        }
        assert {f"u4-{i}" for i in range(6)} <= ids

    def test_the_newest_turn_is_kept_even_over_budget(self) -> None:
        rows: List[Row] = []
        for i in range(5):
            rows += _busy_turn(f"s{i}", calls=40, result_chars=5_000, uid=f"u{i}")
        result = build_window(rows, WindowConfig(max_tokens=MIN_MAX_TOKENS))
        body = _text(result.messages)
        assert "s4 지시" in body and "s4 완료" in body
        assert result.full == 1 and result.dialogue == 0
        assert result.degraded[-1] == "over_budget_newest_kept"

    def test_degradation_is_reported(self) -> None:
        rows: List[Row] = []
        for i in range(5):
            rows += _busy_turn(f"s{i}", calls=12, result_chars=5_000, uid=f"u{i}")
        result = build_window(rows, WindowConfig(max_tokens=3_000))
        meta = result.as_metadata()
        assert result.degraded, "a silently degraded window is one nobody can debug"
        assert meta["tokens"] == result.tokens and meta["budget"] == 3_000

    def test_a_result_is_never_trimmed_below_its_floor(self) -> None:
        rows = _busy_turn("solo", calls=10, result_chars=9_000, uid="u")
        result = build_window(rows, WindowConfig(max_tokens=MIN_MAX_TOKENS))
        kept = [
            str(b["content"])
            for m in result.messages
            for b in m["content"]
            if b.get("type") == "tool_result"
        ]
        assert len(kept) == 10, "the only turn lost its tools"
        head = kept[0].split("…[+")[0]
        assert estimate_text_tokens(head) >= MIN_RESULT_TOKENS - 2


class TestTheConversationTurnsKeepWhatWasSaid:
    def test_the_whole_narration_not_just_the_last_line(self) -> None:
        """"done" without what was done is the completion the agent keeps
        failing to recognise."""
        rows = [
            Row("user", [{"type": "text", "text": "fix the config"}]),
            Row("assistant", [
                {"type": "text", "text": "the key RETRY is missing; adding it"},
                {"type": "tool_use", "id": "a", "name": "Edit", "input": {"file_path": "c.yaml"}},
            ]),
            Row("user", [{"type": "tool_result", "tool_use_id": "a", "content": "ok"}]),
            Row("assistant", [{"type": "text", "text": "done"}]),
        ]
        rows += _turn("x", "y") + _turn("p", "q") + _turn("r", "s")
        body = _text(build_window(rows, WindowConfig()).messages)
        assert "RETRY is missing" in body and "done" in body
        assert "[used tools: Edit(c.yaml)]" in body

    def test_a_long_answer_keeps_its_conclusion(self) -> None:
        rows = [
            Row("user", [{"type": "text", "text": "write it"}]),
            Row("assistant", [{"type": "text", "text": "draft " * 4_000 + "FINAL: shipped v2"}]),
        ]
        rows += _turn("x", "y") + _turn("p", "q") + _turn("r", "s")
        body = _text(build_window(rows, WindowConfig(utterance_tokens=300)).messages)
        assert "FINAL: shipped v2" in body
        assert "chars trimmed]…" in body

    def test_an_image_only_instruction_is_still_an_instruction(self) -> None:
        rows = [
            Row("user", [{"type": "image", "source": {"type": "base64", "data": "AAAA"}}]),
            Row("assistant", [{"type": "text", "text": "a cat"}]),
        ]
        rows += _turn("x", "y") + _turn("p", "q") + _turn("r", "s")
        messages = build_window(rows, WindowConfig()).messages
        assert messages[0]["role"] == "user"
        assert messages[0]["content"][0]["text"] == "[image]"


class TestTrimmingSaysSo:
    def test_a_cut_result_says_how_much_was_cut(self) -> None:
        """Silently cutting is how a truncated listing reads as an empty
        directory."""
        rows = _turn("go", "done", tool="Read", target="f", result="Y" * 90_000, uid="u")
        body = _text(build_window(rows, WindowConfig()).messages)
        assert "chars trimmed" in body

    def test_a_result_under_the_cap_is_untouched(self) -> None:
        rows = _turn("go", "done", tool="Read", target="f", result="Z" * 9_000, uid="u")
        body = _text(build_window(rows, WindowConfig()).messages)
        assert "Z" * 9_000 in body and "trimmed" not in body


class TestTheFloorHolds:
    """A floor that does not hold is worse than no floor: the number in the
    config reads as a guarantee. ``_clip`` counted its own note inside the
    limit, so "keep at least 300" returned 298."""

    def test_the_kept_length_is_content_not_bytes(self) -> None:
        from geny_executor.memory.short_term_window import _clip

        kept = _clip("A" * 5_000, 300)
        assert kept.startswith("A" * 300)
        assert not kept.startswith("A" * 301)
        assert "chars trimmed" in kept

    def test_short_enough_text_is_returned_whole(self) -> None:
        from geny_executor.memory.short_term_window import _clip

        assert _clip("short", 300) == "short"


class TestNothingToShow:
    def test_no_turns_is_an_empty_window(self) -> None:
        assert build_window([], WindowConfig()).messages == []

    def test_disabled_is_an_empty_window(self) -> None:
        rows = _turn("a", "b")
        assert build_window(rows, WindowConfig(full_turns=0, dialogue_turns=0)).messages == []
