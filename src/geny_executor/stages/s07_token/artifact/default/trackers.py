"""Token trackers — concrete implementations for token usage tracking."""

from __future__ import annotations

from geny_executor.core.state import PipelineState, TokenUsage
from geny_executor.stages.s06_api.types import APIResponse
from geny_executor.stages.s07_token.interface import TokenTracker


class DefaultTracker(TokenTracker):
    """Standard tracker — extracts usage from API response."""

    @property
    def name(self) -> str:
        return "default"

    @property
    def description(self) -> str:
        return "Tracks tokens from API response usage field"

    def track(self, response: APIResponse, state: PipelineState) -> TokenUsage:
        usage = response.usage

        # Accumulate into state
        state.token_usage += usage
        state.turn_token_usage.append(usage)

        return usage


class DetailedTracker(TokenTracker):
    """Detailed tracker — tracks per-stage, per-tool breakdowns."""

    @property
    def name(self) -> str:
        return "detailed"

    @property
    def description(self) -> str:
        return "Per-turn, per-stage detailed token tracking"

    def track(self, response: APIResponse, state: PipelineState) -> TokenUsage:
        usage = response.usage

        state.token_usage += usage
        state.turn_token_usage.append(usage)

        # Store detailed breakdown in metadata
        if "token_breakdown" not in state.metadata:
            state.metadata["token_breakdown"] = []

        state.metadata["token_breakdown"].append(
            {
                "iteration": state.iteration,
                "stage": state.current_stage,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_creation": usage.cache_creation_input_tokens,
                "cache_read": usage.cache_read_input_tokens,
            }
        )

        return usage
