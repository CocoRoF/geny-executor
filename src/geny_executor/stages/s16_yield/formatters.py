"""Result formatters — Level 2 strategies for final output formatting."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Optional

from geny_executor.core.stage import Strategy
from geny_executor.core.state import PipelineState


class ResultFormatter(Strategy):
    """Base interface for result formatting."""

    @abstractmethod
    def format(self, state: PipelineState) -> None:
        """Format the final result. Modifies state.final_text / state.final_output."""
        ...


class DefaultFormatter(ResultFormatter):
    """Default formatter — text passthrough."""

    @property
    def name(self) -> str:
        return "default"

    @property
    def description(self) -> str:
        return "Passes text output as-is"

    def format(self, state: PipelineState) -> None:
        # final_text is already set by ParseStage
        pass


class StructuredFormatter(ResultFormatter):
    """Packages result as a structured dict."""

    @property
    def name(self) -> str:
        return "structured"

    @property
    def description(self) -> str:
        return "Packages result as structured dict with metadata"

    def format(self, state: PipelineState) -> None:
        state.final_output = {
            "text": state.final_text,
            "model": state.model,
            "iterations": state.iteration,
            "total_cost_usd": state.total_cost_usd,
            "token_usage": {
                "input_tokens": state.token_usage.input_tokens,
                "output_tokens": state.token_usage.output_tokens,
                "total_tokens": state.token_usage.total_tokens,
            },
            "completion_signal": state.completion_signal,
        }


class StreamingFormatter(ResultFormatter):
    """Emits a final summary event for streaming mode."""

    @property
    def name(self) -> str:
        return "streaming"

    @property
    def description(self) -> str:
        return "Emits streaming completion summary"

    def format(self, state: PipelineState) -> None:
        state.add_event("yield.summary", {
            "text_length": len(state.final_text),
            "iterations": state.iteration,
            "total_cost_usd": state.total_cost_usd,
        })
