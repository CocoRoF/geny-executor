"""Unit tests for the Stage 16 multi-format yield (S7.12)."""

from __future__ import annotations

import pytest

from geny_executor.core.state import PipelineState, TokenUsage
from geny_executor.stages.s21_yield import (
    MultiFormatFormatter,
    YieldStage,
    build_markdown,
    build_structured,
)


# ── helpers ───────────────────────────────────────────────────────────────


def _state(
    *,
    text: str = "hello",
    model: str = "claude-sonnet-4-6",
    iteration: int = 1,
    cost: float = 0.0123,
    completion_signal: str | None = None,
    completion_detail: str | None = None,
    thinking_history: list | None = None,
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> PipelineState:
    state = PipelineState()
    state.final_text = text
    state.model = model
    state.iteration = iteration
    state.total_cost_usd = cost
    state.completion_signal = completion_signal
    state.completion_detail = completion_detail
    state.thinking_history = list(thinking_history or [])
    state.token_usage = TokenUsage(
        input_tokens=input_tokens, output_tokens=output_tokens
    )
    return state


# ── build_structured ────────────────────────────────────────────────────


class TestBuildStructured:
    def test_full_payload(self):
        s = _state(
            text="answer",
            model="claude-opus-4-7",
            iteration=3,
            cost=1.5,
            completion_signal="success",
            input_tokens=200,
            output_tokens=80,
        )
        out = build_structured(s)
        assert out == {
            "text": "answer",
            "model": "claude-opus-4-7",
            "iterations": 3,
            "total_cost_usd": 1.5,
            "token_usage": {
                "input_tokens": 200,
                "output_tokens": 80,
                "total_tokens": 280,
            },
            "completion_signal": "success",
        }

    def test_no_completion_signal(self):
        s = _state(completion_signal=None)
        assert build_structured(s)["completion_signal"] is None


# ── build_markdown ──────────────────────────────────────────────────────


class TestBuildMarkdown:
    def test_minimum_layout(self):
        s = _state(text="hi")
        md = build_markdown(s)
        assert md.startswith("# Result\n")
        assert "hi" in md
        assert "---" in md
        assert "claude-sonnet-4-6" in md
        assert "iterations 1" in md
        assert "$0.0123" in md

    def test_includes_status_when_signal_set(self):
        s = _state(completion_signal="success", completion_detail="all green")
        md = build_markdown(s)
        assert "## Status" in md
        assert "signal: `success`" in md
        assert "detail: all green" in md

    def test_no_status_when_signal_unset(self):
        s = _state(completion_signal=None)
        md = build_markdown(s)
        assert "## Status" not in md

    def test_thinking_excluded_by_default(self):
        s = _state(thinking_history=[{"text": "secret reasoning"}])
        md = build_markdown(s)
        assert "secret reasoning" not in md
        assert "## Thinking" not in md

    def test_thinking_included_when_flag_set(self):
        s = _state(thinking_history=[{"text": "first"}, {"text": "second"}])
        md = build_markdown(s, include_thinking=True)
        assert "## Thinking" in md
        # last thinking turn used
        assert "second" in md
        assert "first" not in md

    def test_thinking_section_skipped_when_history_empty(self):
        s = _state(thinking_history=[])
        md = build_markdown(s, include_thinking=True)
        assert "## Thinking" not in md

    def test_thinking_section_skipped_when_history_text_blank(self):
        s = _state(thinking_history=[{"text": ""}])
        md = build_markdown(s, include_thinking=True)
        assert "## Thinking" not in md


# ── MultiFormatFormatter ────────────────────────────────────────────────


class TestMultiFormatFormatter:
    def test_default_emits_all_three(self):
        f = MultiFormatFormatter()
        state = _state(text="answer", completion_signal="success")
        f.format(state)
        out = state.final_output
        assert isinstance(out, dict)
        assert set(out.keys()) == {"text", "structured", "markdown"}
        assert out["text"] == "answer"
        assert out["structured"]["text"] == "answer"
        assert "# Result" in out["markdown"]

    def test_select_subset_of_formats(self):
        f = MultiFormatFormatter(formats=("text", "markdown"))
        state = _state()
        f.format(state)
        out = state.final_output
        assert set(out.keys()) == {"text", "markdown"}
        assert "structured" not in out

    def test_only_text(self):
        f = MultiFormatFormatter(formats=("text",))
        state = _state(text="x")
        f.format(state)
        assert state.final_output == {"text": "x"}

    def test_only_structured(self):
        f = MultiFormatFormatter(formats=("structured",))
        state = _state(text="x")
        f.format(state)
        assert "structured" in state.final_output
        assert state.final_output["structured"]["text"] == "x"

    def test_only_markdown(self):
        f = MultiFormatFormatter(formats=("markdown",))
        state = _state()
        f.format(state)
        assert "markdown" in state.final_output
        assert "# Result" in state.final_output["markdown"]

    def test_unknown_format_rejected(self):
        with pytest.raises(ValueError, match="unknown format"):
            MultiFormatFormatter(formats=("text", "yaml"))

    def test_empty_formats_rejected(self):
        with pytest.raises(ValueError, match="at least one format"):
            MultiFormatFormatter(formats=())

    def test_duplicate_formats_deduped(self):
        f = MultiFormatFormatter(formats=("text", "text", "markdown"))
        assert f.formats == ("text", "markdown")

    def test_include_thinking_flag(self):
        f = MultiFormatFormatter(
            formats=("markdown",), include_thinking=True
        )
        state = _state(thinking_history=[{"text": "deep thought"}])
        f.format(state)
        assert "deep thought" in state.final_output["markdown"]

    def test_include_thinking_false_by_default(self):
        f = MultiFormatFormatter(formats=("markdown",))
        state = _state(thinking_history=[{"text": "deep thought"}])
        f.format(state)
        assert "deep thought" not in state.final_output["markdown"]

    def test_name_and_description(self):
        f = MultiFormatFormatter()
        assert f.name == "multi_format"
        assert "text" in f.description.lower() or "format" in f.description.lower()

    def test_handles_empty_final_text(self):
        f = MultiFormatFormatter()
        state = _state(text="")
        f.format(state)
        assert state.final_output["text"] == ""


# ── YieldStage registry wiring ──────────────────────────────────────────


class TestYieldStageRegistry:
    def test_multi_format_registered(self):
        stage = YieldStage()
        registry = stage.get_strategy_slots()["formatter"].registry
        assert "multi_format" in registry
        assert registry["multi_format"] is MultiFormatFormatter

    @pytest.mark.asyncio
    async def test_stage_with_multi_format_runs_clean(self):
        stage = YieldStage(formatter=MultiFormatFormatter())
        state = _state(text="hello")
        result = await stage.execute(input=None, state=state)
        # When final_output is a dict it's returned over final_text.
        assert result == state.final_output
        assert isinstance(result, dict)
        assert set(result.keys()) == {"text", "structured", "markdown"}
