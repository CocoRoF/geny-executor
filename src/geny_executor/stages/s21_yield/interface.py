"""Stage 16: Yield — interface definitions."""

from __future__ import annotations

from abc import abstractmethod

from geny_executor.core.stage import Strategy
from geny_executor.core.state import PipelineState


class ResultFormatter(Strategy):
    """Base interface for result formatting."""

    @abstractmethod
    def format(self, state: PipelineState) -> None:
        """Format the final result. Modifies state.final_text / state.final_output."""
        ...
