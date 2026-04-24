"""Pluggable tool source abstraction.

Cycle 20260424 executor uplift — Phase 3 Week 7.

The :class:`~geny_executor.tools.providers.AdhocToolProvider` already
exists for *name-keyed lookup*: the manifest declares
``tools.external = ["foo", "bar"]`` and the pipeline asks each adhoc
provider whether it supplies a tool for each requested name. Useful
when hosts want the manifest to stay authoritative.

This module adds the complementary pattern: **self-contained bundles**.
A :class:`ToolProvider` declares its own name, owns its lifecycle, and
exposes the complete set of tools it ships via :meth:`list_tools`.
Hosts compose a pipeline by handing several providers into
:meth:`Pipeline.from_manifest_async` — think of each provider as a
"feature pack" the host decides to enable wholesale.

Two shipped implementations:

* :class:`BuiltInToolProvider` — wraps the executor's built-in
  catalogue (:data:`~geny_executor.tools.built_in.BUILT_IN_TOOL_CLASSES`)
  with optional feature-gated subset selection.
* Hosts can subclass :class:`ToolProvider` directly for their
  platform-specific tools (e.g. Geny's creature / feed / knowledge
  tools arrive through a ``GenyPlatformToolProvider``).

Unlike :class:`AdhocToolProvider`, the manifest does **not** need to
list the tool names ahead of time — the provider owns the roster, and
the host's decision to wire the provider in IS the opt-in. This fits
the "geny-executor first" principle: hosts import and configure
providers rather than enumerating tool names.

See ``executor_uplift/06_design_tool_system.md`` §8 and
``executor_uplift/12_detailed_plan.md`` §3 (Week 7 Provider Protocol).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

from geny_executor.tools.base import Tool

logger = logging.getLogger(__name__)


class ToolProvider(ABC):
    """Self-contained, lifecycle-aware bundle of tools.

    Subclass this and implement :attr:`name` + :meth:`list_tools` at
    minimum. Override :meth:`startup` / :meth:`shutdown` if your
    provider needs to establish / tear down resources (database
    connections, MCP clients, HTTP sessions).

    The ``Pipeline`` that owns a provider calls :meth:`startup` before
    registering the tools and :meth:`shutdown` when the pipeline is
    destroyed. Both default to no-ops so providers that don't need
    lifecycle hooks can skip them.

    Provider names should be short, lowercase, and stable — they surface
    in logs and error messages. Name collisions between providers are
    detected during registration and surface as ``ValueError``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short provider identifier (e.g. ``"builtin"``, ``"geny-platform"``)."""
        ...

    @property
    def description(self) -> str:
        """Human-readable summary. Default is empty."""
        return ""

    @abstractmethod
    def list_tools(self) -> List[Tool]:
        """Every tool this provider ships.

        Must be callable before :meth:`startup` — the pipeline may
        introspect the catalogue to decide whether to proceed.
        Implementations should return fresh ``Tool`` instances per
        call so host-side modifications don't bleed between pipelines.
        """
        ...

    async def startup(self) -> None:
        """Acquire resources before tools are used. Default no-op."""
        return None

    async def shutdown(self) -> None:
        """Release resources when the pipeline is destroyed. Default no-op.

        Called in reverse registration order so later providers can
        rely on earlier ones during their own shutdown.
        """
        return None


class BuiltInToolProvider(ToolProvider):
    """Wraps the executor's built-in tool catalogue.

    Use this when a host wants the executor's built-ins available
    alongside their own platform tools without enumerating tool names
    in the manifest. Accepts the same ``features=`` / ``names=`` kwargs
    as :func:`~geny_executor.tools.built_in.get_builtin_tools` so hosts
    can pull a subset (e.g. just ``filesystem`` + ``web``).

    Example::

        pipeline = await Pipeline.from_manifest_async(
            manifest,
            tool_providers=[
                BuiltInToolProvider(features=["filesystem", "web", "workflow"]),
                GenyPlatformToolProvider(session=session),
            ],
        )
    """

    def __init__(
        self,
        *,
        features: Optional[Iterable[str]] = None,
        names: Optional[Iterable[str]] = None,
    ):
        # Freeze the selector eagerly so list_tools() is deterministic.
        # Local import avoids the package-init circular dependency.
        from geny_executor.tools.built_in import get_builtin_tools

        self._classes = get_builtin_tools(
            features=list(features) if features is not None else None,
            names=list(names) if names is not None else None,
        )

    @property
    def name(self) -> str:
        return "builtin"

    @property
    def description(self) -> str:
        return (
            f"Executor built-in tools "
            f"({len(self._classes)} tool{'s' if len(self._classes) != 1 else ''})"
        )

    def list_tools(self) -> List[Tool]:
        return [cls() for cls in self._classes.values()]


async def register_providers(
    providers: List[ToolProvider],
    registry: "ToolRegistry",  # type: ignore[name-defined]  # noqa: F821
) -> List[ToolProvider]:
    """Start each provider and register its tools into ``registry``.

    Policy:
        * Providers are started in the order they were declared.
        * A provider name collision (two providers with the same
          :attr:`name`) raises ``ValueError`` before any tools are
          registered — a misconfiguration the host wants to see early.
        * Tool-name collisions inside the registry win in
          *registration order* — when two providers ship a tool with
          the same name, the first provider's tool wins and the second
          is logged at WARNING but not rejected. This keeps a partial
          build usable instead of failing the whole pipeline.

    Returns the list of providers that were successfully started, in
    the order they were registered — callers can pass this to
    :func:`shutdown_providers` to unwind cleanly.

    If any provider's ``startup()`` raises, every previously started
    provider is shut down before the exception propagates — no
    half-started state leaks out.
    """
    started: List[ToolProvider] = []
    seen_names: set[str] = set()

    try:
        for provider in providers:
            pname = provider.name
            if pname in seen_names:
                raise ValueError(
                    f"duplicate tool provider name {pname!r} — providers must have unique names"
                )
            seen_names.add(pname)

            await provider.startup()
            started.append(provider)

            for tool in provider.list_tools():
                if registry.get(tool.name) is not None:
                    logger.warning(
                        "provider %r's tool %r skipped — registry already has a tool with that name",
                        pname,
                        tool.name,
                    )
                    continue
                registry.register(tool)
    except BaseException:
        await shutdown_providers(started)
        raise

    return started


async def shutdown_providers(providers: List[ToolProvider]) -> None:
    """Shut down each provider in reverse registration order.

    Each :meth:`shutdown` is wrapped — a failure in one provider is
    logged at WARNING but does not prevent later providers from
    shutting down. This matches typical resource-cleanup semantics:
    best-effort, don't let one broken shutdown strand the others.
    """
    for provider in reversed(providers):
        try:
            await provider.shutdown()
        except Exception:
            logger.warning(
                "provider %r shutdown raised; continuing",
                provider.name,
                exc_info=True,
            )
