"""Declarative provider profiles for OpenAI-compatible (local) LLM backends.

Where the five built-in clients (``anthropic`` / ``openai`` / ``google`` /
``vllm`` / ``claude_code_cli``) are each a hand-written class, this module
adds a *data-driven* layer: a :class:`ProviderProfile` describes an
OpenAI-compatible backend declaratively (display name, default endpoint,
quirks) and the client class is generated from it. Adding a new local
backend then costs one profile, not one class — the mechanism the
hermes-agent benchmark (``hermes_docs/07_개선_로드맵.md`` P0-A-1/A-2)
flagged as the clean way to make local LLMs first-class.

The three branded profiles below cover the common self-hosted stacks:

* ``ollama``   — Ollama's OpenAI-compatible endpoint (``/v1``)
* ``lmstudio`` — LM Studio's local server
* ``custom``   — any other OpenAI-compatible endpoint (llama.cpp server,
  text-generation-webui, LiteLLM, …); aliased ``local``

All three speak the OpenAI Chat Completions wire format, so the generated
clients subclass :class:`~geny_executor.llm_client.openai.OpenAIClient`
(see :mod:`geny_executor.llm_client.openai_compatible`). This module holds
only the *data* + pure helpers so it can be imported (e.g. by
``_creds_to_client_kwargs``) without pulling the OpenAI SDK path — the
client classes are imported lazily by :func:`get_profiled_client_class`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from geny_executor.llm_client.base import ClientCapabilities


# Capabilities shared by the branded local backends. Mirrors
# ``OpenAIClient`` (tools on, structured output on) rather than the
# conservative ``VLLMClient`` defaults: the branded providers target the
# common agentic local stacks (Ollama / LM Studio running a tool-capable
# model such as qwen2.5-coder or llama3.1). A deployment whose model has
# no tool support downgrades at runtime via
# ``client.configure_capabilities(supports_tools=False, ...)``.
_LOCAL_CAPABILITIES = ClientCapabilities(
    supports_thinking=False,
    supports_tools=True,
    supports_streaming=True,
    supports_tool_choice=True,
    supports_stop_sequences=True,
    supports_top_k=False,
    supports_system_prompt=True,
    supports_structured_output=True,
    supports_session_continuity=False,
    supports_mcp_passthrough=False,
    supports_budget_limit=False,
    supports_token_usage=True,
    supports_cost_usage=False,
    is_subprocess=False,
    requires_workspace=False,
    streaming_granularity="token",
    # Same OpenAI-compatible surface as OpenAIClient: the classic
    # ``thinking_enabled`` (Anthropic shape) and ``top_k`` have no home in
    # Chat Completions kwargs, so they are dropped + reported.
    drops=("thinking_enabled", "top_k"),
)


@dataclass(frozen=True)
class ProviderProfile:
    """Declarative description of an OpenAI-compatible LLM backend.

    Attributes:
        name: Stable provider id (registered in ``ClientRegistry`` and
            used as ``stages[6].config["provider"]``).
        capabilities: Feature flags the generated client advertises.
        aliases: Extra registry names that resolve to this same profile.
        default_base_url: Endpoint used when the host supplies no
            ``base_url`` (e.g. Ollama's ``http://localhost:11434/v1``).
            ``None`` + ``requires_base_url=True`` means the host MUST
            provide one.
        requires_base_url: When True, constructing the client without a
            resolvable ``base_url`` raises ``ValueError`` (matches
            ``VLLMClient``) — a generic ``custom`` endpoint has no sane
            default to fall back to.
        default_max_tokens: Floor sent as ``max_tokens`` when the request
            carries no positive value. Guards the Ollama footgun where a
            missing token cap collapses to ``num_predict=128`` and the
            reply is silently truncated (upstream ollama #3417 / hermes
            #39281). ``None`` disables the floor.
        is_local: Marks the backend as a local/self-hosted endpoint. Read
            by hosts (Geny surfaces a "local model" card) and reserved for
            the auto-tuning step (P0-A-3: context-window probing).
        description: Human-facing one-liner for host UIs.
    """

    name: str
    capabilities: ClientCapabilities
    aliases: Tuple[str, ...] = ()
    default_base_url: Optional[str] = None
    requires_base_url: bool = False
    default_max_tokens: Optional[int] = None
    is_local: bool = True
    description: str = ""

    def all_names(self) -> Tuple[str, ...]:
        """Primary name followed by every alias."""
        return (self.name, *self.aliases)


# ── Built-in profiles ────────────────────────────────────────────────

OLLAMA_PROFILE = ProviderProfile(
    name="ollama",
    capabilities=_LOCAL_CAPABILITIES,
    default_base_url="http://localhost:11434/v1",
    requires_base_url=False,
    default_max_tokens=8192,
    is_local=True,
    description="Ollama (local) — OpenAI-compatible endpoint at /v1.",
)

LMSTUDIO_PROFILE = ProviderProfile(
    name="lmstudio",
    capabilities=_LOCAL_CAPABILITIES,
    default_base_url="http://127.0.0.1:1234/v1",
    requires_base_url=False,
    default_max_tokens=8192,
    is_local=True,
    description="LM Studio (local) — OpenAI-compatible server.",
)

CUSTOM_PROFILE = ProviderProfile(
    name="custom",
    capabilities=_LOCAL_CAPABILITIES,
    aliases=("local",),
    default_base_url=None,
    requires_base_url=True,
    default_max_tokens=8192,
    is_local=True,
    description=(
        "Any OpenAI-compatible endpoint (llama.cpp server, "
        "text-generation-webui, LiteLLM, …). Requires base_url."
    ),
)

#: Primary profiles in registration order.
BUILTIN_PROFILES: Tuple[ProviderProfile, ...] = (
    OLLAMA_PROFILE,
    LMSTUDIO_PROFILE,
    CUSTOM_PROFILE,
)


# Name (incl. aliases) → profile. Built once at import; aliases must not
# collide with a primary name (asserted below so a future edit can't
# silently shadow one).
_NAME_TO_PROFILE: Dict[str, ProviderProfile] = {}
for _profile in BUILTIN_PROFILES:
    for _alias in _profile.all_names():
        if _alias in _NAME_TO_PROFILE and _NAME_TO_PROFILE[_alias] is not _profile:
            raise RuntimeError(
                f"provider-profile name collision on {_alias!r} between "
                f"{_NAME_TO_PROFILE[_alias].name!r} and {_profile.name!r}"
            )
        _NAME_TO_PROFILE[_alias] = _profile
del _profile, _alias  # type: ignore[name-defined]  # keep module namespace clean


# ── Pure helpers (no OpenAI SDK / client import) ──────────────────────


def is_profiled_provider(name: str) -> bool:
    """True if *name* (primary or alias) maps to a built-in profile."""
    return name in _NAME_TO_PROFILE


def resolve_profile(name: str) -> ProviderProfile:
    """Profile for *name* (primary or alias). Raises ``ValueError`` if unknown."""
    try:
        return _NAME_TO_PROFILE[name]
    except KeyError as exc:
        raise ValueError(
            f"no provider profile for {name!r}. Known: {sorted(_NAME_TO_PROFILE)}"
        ) from exc


def profiled_provider_names() -> List[str]:
    """Every registry name (primary + alias) backed by a profile, sorted."""
    return sorted(_NAME_TO_PROFILE)


def builtin_profiles() -> Tuple[ProviderProfile, ...]:
    """The primary built-in profiles (for host introspection / UIs)."""
    return BUILTIN_PROFILES


def profiled_client_kwargs(name: str, creds: Any) -> Dict[str, Any]:
    """Map ``ProviderCredentials`` → constructor kwargs for a profiled client.

    Co-located with the profiles (not in ``pipeline._creds_to_client_kwargs``)
    so the local-backend wire knobs live next to the profile data. Threads:

    * ``api_key`` — falls back to ``"EMPTY"`` so ``AsyncOpenAI`` (which
      rejects an empty key) constructs against a keyless local server.
    * ``base_url`` / ``default_headers`` — endpoint overrides.
    * ``num_ctx`` — from ``extras["ollama_num_ctx"]`` (or ``extras["num_ctx"]``);
      becomes ``extra_body.options.num_ctx`` so Ollama loads the model with
      the requested context window.
    * ``think`` — from ``extras["think"]``; becomes ``extra_body.think``
      (Ollama's native reasoning toggle).

    Pure: imports nothing from the SDK path. ``creds`` is a
    ``ProviderCredentials`` (typed ``Any`` to avoid a circular import).
    """
    kwargs: Dict[str, Any] = {"api_key": getattr(creds, "api_key", "") or "EMPTY"}
    base_url = getattr(creds, "base_url", None)
    if base_url is not None:
        kwargs["base_url"] = base_url
    default_headers = getattr(creds, "default_headers", None)
    if default_headers is not None:
        kwargs["default_headers"] = dict(default_headers)

    extras: Mapping[str, Any] = getattr(creds, "extras", None) or {}
    num_ctx = extras.get("ollama_num_ctx", extras.get("num_ctx"))
    if num_ctx is not None:
        kwargs["num_ctx"] = int(num_ctx)
    think = extras.get("think")
    if think is not None:
        kwargs["think"] = bool(think)
    return kwargs


def get_profiled_client_class(name: str) -> type:
    """Return the generated ``BaseClient`` subclass for *name*.

    Lazily imports :mod:`geny_executor.llm_client.openai_compatible` (which
    pulls the OpenAI client path) so merely *registering* these providers
    in ``ClientRegistry`` stays free of the SDK import — same lazy contract
    the registry's other factories honour.
    """
    profile = resolve_profile(name)
    from geny_executor.llm_client.openai_compatible import CLIENT_CLASSES

    return CLIENT_CLASSES[profile.name]


__all__ = [
    "ProviderProfile",
    "OLLAMA_PROFILE",
    "LMSTUDIO_PROFILE",
    "CUSTOM_PROFILE",
    "BUILTIN_PROFILES",
    "is_profiled_provider",
    "resolve_profile",
    "profiled_provider_names",
    "builtin_profiles",
    "profiled_client_kwargs",
    "get_profiled_client_class",
]
