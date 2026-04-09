"""Stage 9: Parse — interface definitions."""

from __future__ import annotations

from abc import abstractmethod
from enum import Enum
from typing import Optional, Tuple

from geny_executor.core.stage import Strategy
from geny_executor.stages.s06_api.types import APIResponse
from geny_executor.stages.s09_parse.types import ParsedResponse


class CompletionSignal(str, Enum):
    """Structured completion signals."""

    CONTINUE = "continue"
    COMPLETE = "complete"
    BLOCKED = "blocked"
    ERROR = "error"
    DELEGATE = "delegate"
    NONE = "none"


class ResponseParser(Strategy):
    """Base interface for response parsing."""

    @abstractmethod
    def parse(self, response: APIResponse) -> ParsedResponse:
        """Parse API response into structured form."""
        ...


class CompletionSignalDetector(Strategy):
    """Base interface for detecting completion signals in text."""

    @abstractmethod
    def detect(self, text: str) -> Tuple[CompletionSignal, Optional[str]]:
        """Detect signal in text. Returns (signal, detail)."""
        ...
