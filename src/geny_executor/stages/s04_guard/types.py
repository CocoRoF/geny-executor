"""Guard stage data types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GuardResult:
    """Result of a guard check."""

    passed: bool
    guard_name: str = ""
    message: str = ""
    action: str = "reject"  # "reject" | "modify" | "warn"
