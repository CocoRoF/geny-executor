"""MCP prompt → Skill bridge (S8.4).

Walks an :class:`MCPManager`'s connected servers, lists each
server's MCP prompts, and turns them into discoverable
:class:`Skill` records with the canonical id ``mcp__<server>__<prompt>``.

The bridge is **advisory**: the produced Skills carry the prompt's
metadata in :attr:`SkillMetadata.extras` (``server``, ``prompt_name``,
``arguments``) so hosts can introspect what's available, but the
Skill body is intentionally a short placeholder. Actual prompt
fetching happens via :meth:`MCPManager.get_mcp_prompt` — hosts
wanting a "prompt-as-tool" wrapper subclass :class:`SkillTool` and
look up the call target via the extras.

This split keeps the bridge cheap (no extra MCP round-trips during
registration) and decouples discovery from invocation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

from geny_executor.skills.types import Skill, SkillMetadata

if TYPE_CHECKING:
    from geny_executor.tools.mcp.manager import MCPManager

logger = logging.getLogger(__name__)


SKILL_ID_PREFIX = "mcp__"
SKILL_SOURCE_TAG = "mcp"


def mcp_skill_id(server_name: str, prompt_name: str) -> str:
    """Canonical Skill id for an MCP-bridged prompt."""
    if not server_name:
        raise ValueError("server_name must be non-empty")
    if not prompt_name:
        raise ValueError("prompt_name must be non-empty")
    return f"{SKILL_ID_PREFIX}{server_name}__{prompt_name}"


def _placeholder_body(server: str, prompt: str) -> str:
    return (
        f"This skill bridges the MCP prompt `{prompt}` on server "
        f"`{server}`. Hosts invoke it by calling "
        f"`MCPManager.get_mcp_prompt({server!r}, {prompt!r}, args)`."
    )


async def mcp_prompts_to_skills(manager: "MCPManager") -> List[Skill]:
    """Project every CONNECTED server's MCP prompts as :class:`Skill` records.

    Returns an empty list when no servers are connected or none expose
    prompts. Per-server failures are swallowed: a server whose
    ``list_prompts`` raises is skipped (logged at WARNING by the
    connection layer; if the manager-level call itself raises we
    catch and log here so a single misbehaving server can't wedge
    the whole bridge).

    Each Skill has:

    * ``id`` → :func:`mcp_skill_id`
    * ``metadata.name`` → the prompt name as exposed by the server
    * ``metadata.description`` → the prompt description (empty string
      when the server doesn't supply one)
    * ``metadata.extras`` → ``{"server": ..., "prompt_name": ...,
      "arguments": [{"name", "description", "required"}]}``
    * ``body`` → a short placeholder describing the bridge
    """
    out: List[Skill] = []
    for server_name in manager.list_servers():
        if not manager.is_connected(server_name):
            continue
        conn = manager._servers.get(server_name)  # type: ignore[attr-defined]
        if conn is None:
            continue
        try:
            entries = await conn.list_prompts()
        except Exception as exc:  # noqa: BLE001 — per-server isolation
            logger.warning("mcp_prompts_to_skills: %s list_prompts failed: %s", server_name, exc)
            continue
        for entry in entries:
            prompt_name = entry.get("name", "")
            if not prompt_name:
                continue
            description = entry.get("description", "") or ""
            arguments = list(entry.get("arguments") or [])
            meta = SkillMetadata(
                name=prompt_name,
                description=description,
                extras={
                    "server": server_name,
                    "prompt_name": prompt_name,
                    "arguments": arguments,
                    "source": SKILL_SOURCE_TAG,
                    # Phase 10.3 — tag MCP-bridged skills so the shell
                    # block executor strips ``\`\`\`!`` blocks from
                    # bodies served by remote prompt servers. Hosts
                    # that wire other untrusted bridges should follow
                    # the same convention.
                    "source_kind": "mcp",
                },
            )
            out.append(
                Skill(
                    id=mcp_skill_id(server_name, prompt_name),
                    metadata=meta,
                    body=_placeholder_body(server_name, prompt_name),
                    source=None,
                )
            )
    return out


__all__ = [
    "SKILL_ID_PREFIX",
    "SKILL_SOURCE_TAG",
    "mcp_prompts_to_skills",
    "mcp_skill_id",
]
