"""Slash command types — ABC + context + result + category."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SlashCategory(str, Enum):
    """Categorical bucket for help / discovery UIs.

    INTROSPECTION — read-only ``/cost`` ``/status`` ``/help`` etc.
    CONTROL       — mutating ``/cancel`` ``/compact`` ``/clear`` etc.
    DOMAIN        — service-specific commands (``/preset`` for Geny).
    """

    INTROSPECTION = "introspection"
    CONTROL = "control"
    DOMAIN = "domain"


@dataclass
class SlashContext:
    """Runtime context passed to handlers.

    All fields optional — handlers should defensively check before
    use. ``pipeline`` lets handlers introspect strategies; ``extras``
    is the host escape hatch (matches ToolContext.extras pattern).
    """

    pipeline: Optional[Any] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    session_state: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SlashResult:
    """Handler return value.

    ``content`` is rendered as a system message in the chat stream
    (markdown supported). When ``follow_up_prompt`` is set, the host
    sends it to the LLM as the next user turn — this is how
    template-style commands (``/skill-foo args``) chain into model
    input without the LLM having to write the prompt itself.
    """

    content: str
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    follow_up_prompt: Optional[str] = None


class SlashCommand(ABC):
    """One slash command. Subclass and ``execute``; register with the registry."""

    name: str = ""
    description: str = ""
    category: SlashCategory = SlashCategory.INTROSPECTION
    aliases: List[str] = []

    @abstractmethod
    async def execute(self, args: List[str], ctx: SlashContext) -> SlashResult:
        """Run the command. ``args`` are positional tokens after the
        command name (shlex-split). Return a :class:`SlashResult`."""
        ...


__all__ = [
    "SlashCategory",
    "SlashCommand",
    "SlashContext",
    "SlashResult",
]
