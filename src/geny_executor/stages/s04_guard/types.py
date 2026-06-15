"""Guard stage data types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GuardResult:
    """Result of a guard check."""

    passed: bool
    guard_name: str = ""
    message: str = ""
    # "reject"  — hard stop (raise GuardRejectError)
    # "warn"    — log and proceed
    # "modify"  — reserved
    # "compact" — recoverable: GuardStage compacts history and re-checks
    #             once, escalating to a reject only if it still fails.
    action: str = "reject"
