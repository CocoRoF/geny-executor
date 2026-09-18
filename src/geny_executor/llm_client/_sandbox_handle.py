"""The sandbox contract the executor needs from its host.

One protocol, deliberately tiny. Tools that run *inside* a container
(``sandbox_exec``, the built-in file tools when a sandbox is attached)
need exactly two things from the host platform: the name of a running
container and a way to make sure it is up. Everything else — how the
container is created, cloned, bind-mounted, snapshotted or torn down —
is the host's concern and the executor never learns it.

Historically this Protocol lived next to a CLI process runner, because
the thing it targeted was a coding-agent CLI spawned *in* the container.
That runner is gone (2.68.0): no backend owns the agentic loop any more,
every provider is a pure LLM behind this library's own harness. The
contract outlived the runner, so it moved here.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["SandboxHandle"]


@runtime_checkable
class SandboxHandle(Protocol):
    """Minimal handle needed to target a sandbox container.

    Any object exposing a ``container_name`` and an idempotent async
    ``ensure()`` satisfies this — e.g. GAPT's ``WorkspaceSandbox`` or
    Geny's ``GaptSandboxHandle``.
    """

    @property
    def container_name(self) -> str: ...

    async def ensure(self) -> None: ...
