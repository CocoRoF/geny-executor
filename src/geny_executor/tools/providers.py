"""Runtime-provided tool protocol.

Host applications often carry their own tool implementations that cannot
round-trip through :class:`~geny_executor.tools.adhoc.AdhocToolDefinition`
— e.g. Python classes that bind to internal services, have behavior not
expressible as ``http | script | template | composite``, or depend on
per-request context (DB session, per-user auth state, …). Instead of
forcing those into the manifest, the host can hand
:meth:`Pipeline.from_manifest` (or
:meth:`Pipeline.from_manifest_async`) a list of
:class:`AdhocToolProvider` implementations.

The manifest's ``tools.external`` field names which provider-backed
tools are active in a given environment. The pipeline walks
``manifest.tools.external``; for each name the first provider that
returns a :class:`Tool` from :meth:`AdhocToolProvider.get` wins. Names
that no provider claims are skipped silently — same policy as missing
``built_in`` entries, so a partial manifest keeps loading.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

from geny_executor.tools.base import Tool


@runtime_checkable
class AdhocToolProvider(Protocol):
    """Supplies runtime tools not expressible as :class:`AdhocToolDefinition`.

    Implementations are typically wired by the host and passed into
    :meth:`Pipeline.from_manifest` / :meth:`Pipeline.from_manifest_async`
    via the ``adhoc_providers`` kwarg.

    A provider only advertises *which names it can supply*; the pipeline
    decides *which of those names are active* by consulting
    ``manifest.tools.external``. This keeps the manifest authoritative
    while letting the provider evolve its catalog without manifest
    churn.
    """

    def list_names(self) -> List[str]:
        """Names this provider can supply.

        Used by hosts and debugging surfaces (e.g. UI) to enumerate what
        the provider offers. The pipeline itself does not require the
        union of ``list_names()`` to cover ``manifest.tools.external``;
        unclaimed names are silently skipped.
        """
        ...

    def get(self, name: str) -> Optional[Tool]:
        """Return the tool for *name*, or ``None`` if this provider
        does not supply it.

        The pipeline calls :meth:`get` with every name in
        ``manifest.tools.external`` and registers the first non-``None``
        return. Providers should return ``None`` (not raise) for
        unknown names so a multi-provider chain can fall through.
        """
        ...
