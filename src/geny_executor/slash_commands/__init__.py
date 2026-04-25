"""Slash commands subsystem (PR-A.2.1).

A slash command is a server-side dispatch handler addressed by a
short name with a leading ``/``. Hosts type ``/cost`` and the
registered handler runs synchronously — no LLM round trip, no token
cost. Output is rendered into the chat stream as a system message
or, optionally, as a follow-up prompt that the LLM then answers.

This module ships:

* :class:`SlashCommandRegistry` — register / resolve / list. Singleton
  available via :func:`get_default_registry`.
* :func:`parse_slash` — detect a slash prefix in user input and split
  it into command + args + remaining_prompt.
* :class:`SlashCommand` ABC — implement ``execute`` to add a command.
* :class:`SlashContext` — pipeline / session handle passed to handlers.
* :class:`SlashResult` — handler return value (content +
  optional follow_up_prompt).
* :class:`SlashCategory` — categorical tag for help / discovery.
"""

from geny_executor.slash_commands.parser import ParsedSlash, parse_slash
from geny_executor.slash_commands.registry import (
    SlashCommandRegistry,
    get_default_registry,
)
from geny_executor.slash_commands.types import (
    SlashCategory,
    SlashCommand,
    SlashContext,
    SlashResult,
)

__all__ = [
    "ParsedSlash",
    "SlashCategory",
    "SlashCommand",
    "SlashCommandRegistry",
    "SlashContext",
    "SlashResult",
    "get_default_registry",
    "parse_slash",
]
