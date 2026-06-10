"""Provider-credentials bundle — the single channel by which hosts inject
authentication into the executor.

Replaces the legacy ``Pipeline.from_manifest_async(api_key=...)`` single-string
path (kept alive for back-compat in earlier executor versions; removed once
Phase A3 lands).

Design notes
------------

* ``ProviderCredentials`` is provider-shaped: API providers care about
  ``api_key`` (+ optional ``base_url`` / headers), CLI providers care about
  ``binary_path`` plus any extras (workspace_root, MCP config, etc).
* ``CredentialBundle`` is just a ``provider_name → ProviderCredentials`` map
  with two helpers: ``.get`` (soft) and ``.require`` (raises ``ConfigError``).
* ``ProviderCredentials.__repr__`` redacts ``api_key`` so credentials cannot
  leak through logs / event_sink dumps / debug repr.
* No convenience ``from_legacy_api_key`` / ``from_env`` constructors — hosts
  must build the bundle explicitly. Geny owns its own builder
  (``backend/service/settings/credentials.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from geny_executor.core.errors import GenyExecutorError


class ConfigError(GenyExecutorError):
    """Configuration error — missing credentials, unknown provider, etc."""


@dataclass(frozen=True)
class ProviderCredentials:
    """Authentication / configuration for one provider.

    Field semantics:
    - ``api_key``: vendor API key (Anthropic / OpenAI / Google) or "" for
      providers that don't use an API key (vLLM with EMPTY, CLI backends
      that authenticate via subscription).
    - ``auth_mode``: which credential channel the provider should be
      driven through — ``'api_key' | 'oauth' | 'setup_token' | 'auto'``.
      Only CLI backends consume it today (it decides e.g. whether
      ``claude`` runs with ``--bare``); API providers ignore it.
      ``'auto'`` (default) means "infer from the other fields": a
      non-empty ``api_key`` resolves to the API-key path, otherwise the
      subscription/OAuth path. Explicit values exist because inference
      was previously done by sniffing the *spawning process's*
      ``ANTHROPIC_API_KEY`` env var — a variable the scrubbed child env
      never necessarily contained — which broke every subscription user
      the moment an unrelated key was exported (PR #868 history).
      Declaring ``'oauth'`` / ``'setup_token'`` also marks the
      credentials as non-empty: the credential material lives on disk
      (``claude auth`` state), not in this object.
    - ``base_url``: HTTP endpoint override (vLLM, custom Anthropic proxy).
    - ``default_headers``: extra HTTP headers (e.g. Anthropic-Beta).
    - ``binary_path``: CLI backend binary path (claude, gh).
    - ``extras``: provider-specific knobs (workspace_root, mcp_config,
      allow_tools, ...). Each client knows how to read its own keys.
    """

    api_key: str = ""
    base_url: Optional[str] = None
    default_headers: Optional[Mapping[str, str]] = None
    binary_path: Optional[str] = None
    extras: Mapping[str, Any] = field(default_factory=dict)
    auth_mode: str = "auto"

    def __repr__(self) -> str:  # noqa: D401 — short form
        redacted = "<redacted>" if self.api_key else ""
        return (
            "ProviderCredentials("
            f"api_key={redacted!r}, "
            f"auth_mode={self.auth_mode!r}, "
            f"base_url={self.base_url!r}, "
            f"binary_path={self.binary_path!r}, "
            f"extras_keys={list(self.extras)!r})"
        )

    def is_empty(self) -> bool:
        """True if no credential material is present at all.

        An explicit (non-``'auto'``) ``auth_mode`` counts as material:
        it is the host's declaration that a disk-resident credential
        (subscription OAuth / setup-token state) exists for this
        provider, which is exactly what ``CredentialBundle.has`` /
        ``preferred_provider`` need to know.
        """
        return (
            not self.api_key
            and self.base_url is None
            and not self.binary_path
            and not self.extras
            and self.auth_mode == "auto"
        )


@dataclass(frozen=True)
class CredentialBundle:
    """Bundle of per-provider credentials.

    The host (Geny) builds one bundle per session and passes it to
    ``Pipeline.from_manifest_async``. Stages and sub-pipelines look up the
    needed provider by name.

    Honesty note (audit §2.6): this bundle is the *intended* single
    channel, but it is not yet the only one — the embedding/LTM boundary
    still reads its key from an env-var ladder outside this object.
    Until that boundary is migrated, do not assume every credential in
    the process flowed through here.
    """

    by_provider: Mapping[str, ProviderCredentials] = field(default_factory=dict)

    def get(self, provider: str) -> ProviderCredentials:
        """Soft lookup. Returns an empty ``ProviderCredentials`` if missing."""
        return self.by_provider.get(provider, ProviderCredentials())

    def require(self, provider: str) -> ProviderCredentials:
        """Strict lookup. Raises ``ConfigError`` if the provider has no
        usable credential material."""
        cred = self.get(provider)
        if cred.is_empty():
            raise ConfigError(
                f"No credentials configured for provider {provider!r}. "
                "Either supply them via CredentialBundle or the appropriate "
                "environment variable."
            )
        return cred

    def has(self, provider: str) -> bool:
        """True if this bundle carries non-empty credentials for ``provider``."""
        return not self.get(provider).is_empty()

    def providers(self) -> list[str]:
        """Names of providers carrying non-empty credentials."""
        return sorted(p for p, c in self.by_provider.items() if not c.is_empty())

    def preferred_provider(
        self,
        order: Sequence[str] = (
            "claude_code_cli",
            "anthropic",
            "openai",
            "google",
            "vllm",
        ),
    ) -> Optional[str]:
        """First provider in ``order`` with non-empty credentials.

        This is the library-owned answer to "which backend should a new
        environment default to". Both hosts previously re-implemented the
        heuristic (Geny's ``backend_resolver``, GAPT's equivalent) on top
        of ``has()`` — and re-implementations drift: the question belongs
        next to the data it inspects. The default order encodes the
        executor's preference for the agentic CLI backend when its
        credentials exist, then the vendor APIs by capability breadth.
        Hosts with different priorities pass their own ``order``; an
        empty bundle (or one whose providers are all outside ``order``)
        returns ``None`` so callers must handle the "nothing configured"
        case explicitly instead of inheriting a silent default.
        """
        for provider in order:
            if self.has(provider):
                return provider
        return None
