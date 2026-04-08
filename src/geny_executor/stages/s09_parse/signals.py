"""Completion signal detection — Level 2 strategies."""

from __future__ import annotations

import re
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from geny_executor.core.stage import Strategy


class CompletionSignal(str, Enum):
    """Structured completion signals."""

    CONTINUE = "continue"
    COMPLETE = "complete"
    BLOCKED = "blocked"
    ERROR = "error"
    DELEGATE = "delegate"
    NONE = "none"


class CompletionSignalDetector(Strategy):
    """Base interface for detecting completion signals in text."""

    @abstractmethod
    def detect(self, text: str) -> Tuple[CompletionSignal, Optional[str]]:
        """Detect signal in text. Returns (signal, detail)."""
        ...


class RegexDetector(CompletionSignalDetector):
    """Regex-based signal detection — matches [SIGNAL: detail] patterns."""

    # Patterns matching Geny's existing protocol
    PATTERNS = {
        CompletionSignal.CONTINUE: re.compile(
            r"\[CONTINUE(?::?\s*(.+?))?\]", re.IGNORECASE
        ),
        CompletionSignal.COMPLETE: re.compile(
            r"\[(?:TASK_)?COMPLETE(?::?\s*(.+?))?\]", re.IGNORECASE
        ),
        CompletionSignal.BLOCKED: re.compile(
            r"\[BLOCKED(?::?\s*(.+?))?\]", re.IGNORECASE
        ),
        CompletionSignal.ERROR: re.compile(
            r"\[ERROR(?::?\s*(.+?))?\]", re.IGNORECASE
        ),
        CompletionSignal.DELEGATE: re.compile(
            r"\[DELEGATE(?::?\s*(.+?))?\]", re.IGNORECASE
        ),
    }

    @property
    def name(self) -> str:
        return "regex"

    @property
    def description(self) -> str:
        return "Regex-based [SIGNAL: detail] pattern matching"

    def detect(self, text: str) -> Tuple[CompletionSignal, Optional[str]]:
        for signal, pattern in self.PATTERNS.items():
            match = pattern.search(text)
            if match:
                detail = match.group(1) if match.lastindex else None
                return signal, detail
        return CompletionSignal.NONE, None


class StructuredDetector(CompletionSignalDetector):
    """JSON-based signal detection for structured output."""

    @property
    def name(self) -> str:
        return "structured"

    @property
    def description(self) -> str:
        return "JSON-based structured signal detection"

    def detect(self, text: str) -> Tuple[CompletionSignal, Optional[str]]:
        import json

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                signal_str = data.get("signal", data.get("status", ""))
                detail = data.get("detail", data.get("reason", None))
                try:
                    signal = CompletionSignal(signal_str.lower())
                    return signal, detail
                except ValueError:
                    pass
        except (json.JSONDecodeError, TypeError):
            pass

        return CompletionSignal.NONE, None


class HybridDetector(CompletionSignalDetector):
    """Tries regex first, then structured."""

    def __init__(self):
        self._regex = RegexDetector()
        self._structured = StructuredDetector()

    @property
    def name(self) -> str:
        return "hybrid"

    @property
    def description(self) -> str:
        return "Regex + JSON hybrid detection"

    def detect(self, text: str) -> Tuple[CompletionSignal, Optional[str]]:
        signal, detail = self._regex.detect(text)
        if signal != CompletionSignal.NONE:
            return signal, detail
        return self._structured.detect(text)
