"""Fork execution mode for skills — Phase 10.5.

A *fork-mode* skill runs in an isolated sub-agent: separate context,
separate token budget, optional model override, optional tool roster
narrowing. The result text returns to the parent as the
:class:`ToolResult` content. The parent's conversation history /
permissions / pipeline state are not mutated by the fork.

Inline mode (Phase 4) returns the rendered body and lets the *parent
LLM* execute it. Fork mode (this module) actually fires off a
secondary completion / agent run and returns the *result of that
run*. The parent never sees the body of a fork-mode skill — it sees
the answer.

Architecture
============

We ship a runner *protocol* plus a default implementation that uses
the executor's :class:`BaseClient` directly. Hosts that want a
heavier sub-agent (full Pipeline.from_manifest_async + MCP +
session-scoped state) can plug in a custom runner without touching
the SkillTool wiring.

The protocol::

    class SkillForkRunner(Protocol):
        async def __call__(
            self,
            *,
            skill: Skill,
            rendered_body: str,
            invoke_args: Dict[str, Any],
            parent_context: ToolContext,
        ) -> ForkResult: ...

Wiring entry points:

* :class:`SkillToolProvider(fork_runner=...)` — sets the runner
  every :class:`SkillTool` it produces uses.
* :class:`SkillTool(fork_runner=...)` — set per tool when hand-
  building.
* :func:`make_default_fork_runner` — convenience factory that
  binds an Anthropic-backed client. Returns ``None`` when the
  required env var is missing so callers can fall back gracefully.

Tests pass a stub runner so the suite never makes real API calls.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional

from geny_executor.skills.types import Skill
from geny_executor.tools.base import ToolContext

logger = logging.getLogger(__name__)


@dataclass
class ForkResult:
    """Outcome of running a fork-mode skill. Mirrors the relevant
    bits of :class:`ToolResult` so :class:`SkillTool.execute()` can
    forward them with minimal translation."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_error: bool = False


# Runner signature: async callable. Using ``Callable`` instead of a
# Protocol class keeps the type compatible with both function- and
# method-based runners.
SkillForkRunner = Callable[..., Awaitable[ForkResult]]


_DEFAULT_FORK_MODEL = "claude-sonnet-4-6"
_DEFAULT_FORK_MAX_TOKENS = 4096


def make_default_fork_runner(
    api_key: Optional[str] = None,
    *,
    fallback_model: str = _DEFAULT_FORK_MODEL,
    max_tokens: int = _DEFAULT_FORK_MAX_TOKENS,
) -> Optional[SkillForkRunner]:
    """Build an Anthropic-backed default fork runner.

    Args:
        api_key: API key for the provider. ``None`` → reads
            ``ANTHROPIC_API_KEY`` from the env. When neither is set
            this returns ``None`` so the caller can decide whether
            to no-op or surface an error.
        fallback_model: Model used when the skill itself didn't
            declare a ``model_override``. Defaults to the executor's
            current "good general-purpose" pick.
        max_tokens: Per-fork output ceiling. Skills that need more
            should override via the ``effort`` field once 10.5
            wires it (or via a custom runner).

    Returns:
        A runner callable, or ``None`` when no API key is available.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        logger.debug(
            "make_default_fork_runner: no api_key / ANTHROPIC_API_KEY; "
            "returning None — caller must wire a custom runner"
        )
        return None

    async def _runner(
        *,
        skill: Skill,
        rendered_body: str,
        invoke_args: Dict[str, Any],
        parent_context: ToolContext,
    ) -> ForkResult:
        # Lazy imports — keep ``geny_executor.skills`` cheap to
        # import for hosts that never use fork mode.
        # NOTE: This default runner is Anthropic-only by design — host
        # apps that need multi-provider fork should wire their own runner
        # using `CredentialBundle` + the appropriate client class. Phase
        # D4 of the LLM backend upgrade cycle revisits this surface to
        # promote multi-provider as the default.
        from geny_executor.core.config import ModelConfig
        from geny_executor.llm_client import AnthropicClient

        model = skill.metadata.model_override or fallback_model
        client = AnthropicClient(api_key=key)
        model_config = ModelConfig(model=model, max_tokens=max_tokens)

        # The skill body is the system prompt; the user message tells
        # the sub-agent to execute. Args become a structured payload
        # so the sub-agent can read them deterministically.
        user_content = "Execute the skill following the system prompt."
        if invoke_args:
            import json

            user_content = user_content + "\n\nArguments:\n" + json.dumps(invoke_args, indent=2)

        try:
            response = await client.create_message(
                model_config=model_config,
                messages=[{"role": "user", "content": user_content}],
                system=rendered_body,
                purpose=f"skill_fork:{skill.id}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "default fork runner: skill=%s api call failed: %s",
                skill.id,
                exc,
            )
            return ForkResult(
                content=f"fork API call failed: {exc}",
                metadata={"model": model, "skill_id": skill.id},
                is_error=True,
            )

        # Extract text from the response. The :class:`APIResponse`
        # carries an Anthropic-shaped ``content`` list.
        text_parts = []
        for block in getattr(response, "content", None) or []:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            else:
                if getattr(block, "type", None) == "text":
                    text_parts.append(getattr(block, "text", ""))

        text = "\n".join(p for p in text_parts if p) or "(no text in fork response)"

        usage = getattr(response, "usage", None)
        meta: Dict[str, Any] = {"model": model, "skill_id": skill.id}
        if usage is not None:
            meta["input_tokens"] = getattr(usage, "input_tokens", 0)
            meta["output_tokens"] = getattr(usage, "output_tokens", 0)

        return ForkResult(content=text, metadata=meta)

    return _runner


def make_credential_bundle_fork_runner(
    credentials: Any,
    *,
    fallback_provider: str = "anthropic",
    fallback_model: str = _DEFAULT_FORK_MODEL,
    max_tokens: int = _DEFAULT_FORK_MAX_TOKENS,
) -> SkillForkRunner:
    """Multi-provider fork runner backed by a :class:`CredentialBundle`.

    The runner chooses its client based on ``skill.metadata.provider``
    (falling back to ``fallback_provider``). Authentication flows through
    the host-supplied :class:`CredentialBundle`, so a fork-mode skill
    can run on any of the 6 registered providers — including the CLI
    backends — without separately wiring API keys.

    Returns a runner unconditionally; missing credentials surface at
    invocation time as a :class:`ForkResult` with ``is_error=True``
    rather than crashing at construction.
    """
    from geny_executor.core.config import ModelConfig
    from geny_executor.core.pipeline import _creds_to_client_kwargs
    from geny_executor.llm_client.credentials import ConfigError, CredentialBundle
    from geny_executor.llm_client.registry import ClientRegistry

    if not isinstance(credentials, CredentialBundle):
        # Defensive: hosts wiring this runner pass a CredentialBundle.
        credentials = CredentialBundle()

    async def _runner(
        *,
        skill: "Skill",
        rendered_body: str,
        invoke_args: Dict[str, Any],
        parent_context: ToolContext,
    ) -> ForkResult:
        provider = getattr(skill.metadata, "provider", None) or fallback_provider

        try:
            creds = credentials.require(provider)
        except ConfigError as e:
            return ForkResult(
                content=(
                    f"fork runner: provider {provider!r} has no configured "
                    f"credentials in this session ({e})"
                ),
                metadata={"skill_id": skill.id, "provider": provider},
                is_error=True,
            )

        try:
            client_cls = ClientRegistry.get(provider)
        except (KeyError, ValueError) as e:
            return ForkResult(
                content=f"fork runner: unknown provider {provider!r} ({e})",
                metadata={"skill_id": skill.id, "provider": provider},
                is_error=True,
            )

        client = client_cls(**_creds_to_client_kwargs(provider, creds))
        model = skill.metadata.model_override or fallback_model
        model_config = ModelConfig(model=model, max_tokens=max_tokens)

        user_content = "Execute the skill following the system prompt."
        if invoke_args:
            import json as _json

            user_content = user_content + "\n\nArguments:\n" + _json.dumps(invoke_args, indent=2)

        try:
            response = await client.create_message(
                model_config=model_config,
                messages=[{"role": "user", "content": user_content}],
                system=rendered_body,
                purpose=f"skill_fork:{skill.id}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "credential-bundle fork runner: skill=%s provider=%s API call failed: %s",
                skill.id,
                provider,
                exc,
            )
            return ForkResult(
                content=f"fork API call failed: {exc}",
                metadata={"model": model, "provider": provider, "skill_id": skill.id},
                is_error=True,
            )

        text_parts = []
        for block in getattr(response, "content", None) or []:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            else:
                if getattr(block, "type", None) == "text":
                    text_parts.append(getattr(block, "text", ""))

        text = "\n".join(p for p in text_parts if p) or "(no text in fork response)"

        usage = getattr(response, "usage", None)
        meta: Dict[str, Any] = {
            "model": model,
            "provider": provider,
            "skill_id": skill.id,
        }
        if usage is not None:
            meta["input_tokens"] = getattr(usage, "input_tokens", 0)
            meta["output_tokens"] = getattr(usage, "output_tokens", 0)
            cost = getattr(usage, "cost_usd", None)
            if cost is not None:
                meta["cost_usd"] = cost

        return ForkResult(content=text, metadata=meta)

    return _runner


__all__ = [
    "ForkResult",
    "SkillForkRunner",
    "make_default_fork_runner",
]
