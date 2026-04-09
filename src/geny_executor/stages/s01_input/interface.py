"""Stage 1: Input — interface definitions.

This module defines the abstract contracts (ABCs) for the Input stage.
All artifacts implementing Stage 1 should use these strategy interfaces.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional

from geny_executor.core.stage import Strategy
from geny_executor.stages.s01_input.types import NormalizedInput


class InputValidator(Strategy):
    """Base interface for input validation (Level 2 strategy)."""

    @abstractmethod
    def validate(self, raw_input: Any) -> Optional[str]:
        """Validate input. Returns error message if invalid, None if valid."""
        ...


class InputNormalizer(Strategy):
    """Base interface for input normalization (Level 2 strategy)."""

    @abstractmethod
    def normalize(self, raw_input: Any) -> NormalizedInput:
        """Transform raw input into NormalizedInput."""
        ...
