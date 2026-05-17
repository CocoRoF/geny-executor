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
from typing import Any, Mapping, Optional

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

    def __repr__(self) -> str:  # noqa: D401 — short form
        redacted = "<redacted>" if self.api_key else ""
        return (
            "ProviderCredentials("
            f"api_key={redacted!r}, "
            f"base_url={self.base_url!r}, "
            f"binary_path={self.binary_path!r}, "
            f"extras_keys={list(self.extras)!r})"
        )

    def is_empty(self) -> bool:
        """True if no credential material is present at all."""
        return (
            not self.api_key
            and self.base_url is None
            and not self.binary_path
            and not self.extras
        )


@dataclass(frozen=True)
class CredentialBundle:
    """Bundle of per-provider credentials.

    The host (Geny) builds one bundle per session and passes it to
    ``Pipeline.from_manifest_async``. Stages and sub-pipelines look up the
    needed provider by name; no other credential channel exists.
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
