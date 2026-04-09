"""Input validators — Level 2 strategies for input validation."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional

from geny_executor.core.stage import Strategy


class InputValidator(Strategy):
    """Base interface for input validation."""

    @abstractmethod
    def validate(self, raw_input: Any) -> Optional[str]:
        """Validate input. Returns error message if invalid, None if valid."""
        ...


class DefaultValidator(InputValidator):
    """Standard validator — length and type checks."""

    def __init__(self, max_length: int = 1_000_000, min_length: int = 1):
        self._max_length = max_length
        self._min_length = min_length

    @property
    def name(self) -> str:
        return "default"

    @property
    def description(self) -> str:
        return "Standard length and type validation"

    def validate(self, raw_input: Any) -> Optional[str]:
        if raw_input is None:
            return "Input cannot be None"

        text = str(raw_input).strip()
        if len(text) < self._min_length:
            return f"Input too short (min {self._min_length} chars)"
        if len(text) > self._max_length:
            return f"Input too long (max {self._max_length} chars)"
        return None


class PassthroughValidator(InputValidator):
    """No validation — pass everything through."""

    @property
    def name(self) -> str:
        return "passthrough"

    @property
    def description(self) -> str:
        return "No validation, accepts all input"

    def validate(self, raw_input: Any) -> Optional[str]:
        return None


class StrictValidator(InputValidator):
    """Strict validator — additional pattern checks."""

    def __init__(
        self,
        max_length: int = 100_000,
        blocked_patterns: Optional[List[str]] = None,
    ):
        self._max_length = max_length
        self._blocked_patterns = blocked_patterns or []

    @property
    def name(self) -> str:
        return "strict"

    @property
    def description(self) -> str:
        return "Strict validation with pattern blocking"

    def validate(self, raw_input: Any) -> Optional[str]:
        if raw_input is None:
            return "Input cannot be None"

        text = str(raw_input).strip()
        if not text:
            return "Input cannot be empty"
        if len(text) > self._max_length:
            return f"Input too long (max {self._max_length} chars)"

        text_lower = text.lower()
        for pattern in self._blocked_patterns:
            if pattern.lower() in text_lower:
                return "Input contains blocked pattern"

        return None


class SchemaValidator(InputValidator):
    """JSON Schema based validation for structured input."""

    def __init__(self, schema: Dict[str, Any]):
        self._schema = schema

    @property
    def name(self) -> str:
        return "schema"

    @property
    def description(self) -> str:
        return "JSON Schema based validation"

    def validate(self, raw_input: Any) -> Optional[str]:
        if not isinstance(raw_input, dict):
            return "Input must be a dictionary for schema validation"
        # Basic type checking without jsonschema dependency
        required = self._schema.get("required", [])
        for key in required:
            if key not in raw_input:
                return f"Missing required field: {key}"
        return None
