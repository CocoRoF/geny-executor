"""Stage 9: Parse — parses API response into structured form."""

from __future__ import annotations

from typing import Any, List, Optional

from geny_executor.core.stage import Stage, StrategyInfo
from geny_executor.core.state import PipelineState
from geny_executor.stages.s06_api.types import APIResponse
from geny_executor.stages.s09_parse.parsers import DefaultParser, ResponseParser
from geny_executor.stages.s09_parse.signals import (
    CompletionSignal,
    CompletionSignalDetector,
    RegexDetector,
)
from geny_executor.stages.s09_parse.types import ParsedResponse


class ParseStage(Stage[Any, ParsedResponse]):
    """Stage 9: Parse.

    Dual abstraction:
      - Level 2 parser: extracts text, tool calls, thinking
      - Level 2 signal_detector: detects completion signals
    """

    def __init__(
        self,
        parser: Optional[ResponseParser] = None,
        signal_detector: Optional[CompletionSignalDetector] = None,
    ):
        self._parser = parser or DefaultParser()
        self._signal_detector = signal_detector or RegexDetector()

    @property
    def name(self) -> str:
        return "parse"

    @property
    def order(self) -> int:
        return 9

    @property
    def category(self) -> str:
        return "execution"

    async def execute(self, input: Any, state: PipelineState) -> ParsedResponse:
        # Accept either APIResponse directly or pull from state
        if isinstance(input, APIResponse):
            api_response = input
        elif state.last_api_response and isinstance(state.last_api_response, APIResponse):
            api_response = state.last_api_response
        else:
            api_response = input  # trust the pipeline

        parsed = self._parser.parse(api_response)

        # Detect completion signals
        if parsed.text:
            signal, detail = self._signal_detector.detect(parsed.text)
            if signal != CompletionSignal.NONE:
                parsed.signal = signal.value
                parsed.signal_detail = detail
                state.completion_signal = signal.value
                state.completion_detail = detail

        # Store tool calls in state for Stage 10 (Tool)
        # Always clear first to prevent stale calls from prior iteration
        state.pending_tool_calls = []
        if parsed.has_tool_calls:
            state.pending_tool_calls = [
                {
                    "tool_use_id": tc.tool_use_id,
                    "tool_name": tc.tool_name,
                    "tool_input": tc.tool_input,
                }
                for tc in parsed.tool_calls
            ]

        # Store thinking in state for Stage 8 (Think) — or if Think is bypassed
        if parsed.thinking_texts:
            for txt in parsed.thinking_texts:
                state.thinking_history.append({
                    "iteration": state.iteration,
                    "text": txt,
                })

        # Update final text
        state.final_text = parsed.text

        state.add_event("parse.complete", {
            "text_length": len(parsed.text),
            "tool_calls": len(parsed.tool_calls),
            "signal": parsed.signal,
            "stop_reason": parsed.stop_reason,
        })

        return parsed

    def list_strategies(self) -> List[StrategyInfo]:
        return [
            StrategyInfo(
                slot_name="parser",
                current_impl=type(self._parser).__name__,
                available_impls=["DefaultParser", "StructuredOutputParser"],
            ),
            StrategyInfo(
                slot_name="signal_detector",
                current_impl=type(self._signal_detector).__name__,
                available_impls=["RegexDetector", "StructuredDetector", "HybridDetector"],
            ),
        ]
