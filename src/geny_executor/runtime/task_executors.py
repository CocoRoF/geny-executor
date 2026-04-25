"""Background task executors (PR-A.1.3).

A :class:`BackgroundTaskExecutor` knows how to run **one kind** of
task. It is invoked by :class:`BackgroundTaskRunner` once per
submitted :class:`TaskRecord`. The runner owns scheduling, lifecycle,
and output persistence; the executor owns the actual work.

Built-in executors:

* :class:`LocalBashExecutor` — runs ``payload['command']`` via shell.
* :class:`LocalAgentExecutor` — runs a subagent type registered on a
  :class:`SubagentTypeOrchestrator`. Yields the orchestrator's
  serialized result as a single chunk on completion (the runner
  persists per-step output via separate event hooks if needed).

Hosts that need additional task kinds (``remote_agent``,
``monitor_mcp``, ``in_process_teammate``) implement
:class:`BackgroundTaskExecutor` and pass their executor map to
:class:`BackgroundTaskRunner`.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Callable, Optional

from geny_executor.stages.s13_task_registry.types import TaskRecord


class BackgroundTaskExecutor(ABC):
    """One executor handles one task ``kind``.

    Implementations yield output bytes as work progresses. The runner
    persists each chunk via :meth:`TaskRegistry.append_output`. Raise
    on failure — the runner catches the exception and marks the task
    ``FAILED`` with ``str(exc)``.
    """

    @abstractmethod
    async def execute(self, record: TaskRecord) -> AsyncIterator[bytes]:
        """Execute the task. Yield output chunks as they become
        available. The runner appends each chunk to the registry's
        per-task output buffer.
        """
        ...


class LocalBashExecutor(BackgroundTaskExecutor):
    """Runs ``record.payload['command']`` via the shell.

    Uses ``asyncio.create_subprocess_shell`` so chained pipes work.
    Stdout + stderr are merged so the user sees errors interleaved
    with normal output. Non-zero exit raises ``RuntimeError``.
    """

    def __init__(self, *, max_output_bytes: int = 64 * 1024 * 1024) -> None:
        self._max_output_bytes = max_output_bytes

    async def execute(self, record: TaskRecord) -> AsyncIterator[bytes]:
        command = record.payload.get("command")
        if not command:
            raise ValueError("local_bash task requires payload['command']")
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert proc.stdout is not None
        emitted = 0
        try:
            while True:
                chunk = await proc.stdout.read(4096)
                if not chunk:
                    break
                emitted += len(chunk)
                yield chunk
                if emitted >= self._max_output_bytes:
                    proc.kill()
                    raise RuntimeError(
                        f"local_bash exceeded max_output_bytes={self._max_output_bytes}"
                    )
        finally:
            # Make sure we never leave a zombie even if the consumer
            # cancelled or raised.
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
        rc = proc.returncode if proc.returncode is not None else await proc.wait()
        if rc != 0:
            raise RuntimeError(f"local_bash exited rc={rc}")


class LocalAgentExecutor(BackgroundTaskExecutor):
    """Runs a subagent type via :class:`SubagentTypeOrchestrator`.

    Required ``record.payload`` keys:
        ``subagent_type``: registered descriptor id
        ``prompt``: initial prompt for the sub-pipeline

    Optional:
        ``model``: per-call model override

    The executor delegates to ``orchestrator_factory()`` so the host
    can build a fresh orchestrator per task (avoids leaking session
    state across background runs).
    """

    def __init__(self, orchestrator_factory: Callable[[], Any]) -> None:
        self._factory = orchestrator_factory

    async def execute(self, record: TaskRecord) -> AsyncIterator[bytes]:
        subagent_type = record.payload.get("subagent_type")
        prompt = record.payload.get("prompt")
        if not subagent_type:
            raise ValueError("local_agent task requires payload['subagent_type']")
        if prompt is None:
            raise ValueError("local_agent task requires payload['prompt']")

        orch = self._factory()
        # SubagentTypeOrchestrator API: spawn(...) returns AgentResult-like.
        # We dispatch via a generic ``run_subagent`` shim so the runtime
        # remains decoupled from the orchestrator's exact method names.
        runner = getattr(orch, "run_subagent", None) or getattr(orch, "spawn", None)
        if runner is None:
            raise RuntimeError(
                "orchestrator_factory() returned an object without run_subagent / spawn"
            )
        result = await runner(
            subagent_type,
            prompt,
            model=record.payload.get("model"),
        )
        # Serialize the result so callers reading via stream_output see
        # something meaningful. Hosts that want richer streaming can
        # supply their own executor.
        if isinstance(result, (bytes, bytearray)):
            yield bytes(result)
        elif isinstance(result, str):
            yield result.encode("utf-8")
        else:
            yield json.dumps(result, ensure_ascii=False, default=str).encode("utf-8")


__all__ = [
    "BackgroundTaskExecutor",
    "LocalAgentExecutor",
    "LocalBashExecutor",
]
