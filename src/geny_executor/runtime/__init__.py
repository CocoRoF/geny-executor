"""Framework runtime layer — background workers, schedulers, lifecycles.

Modules here run **outside** the synchronous pipeline path. They are
service-instantiated at startup (FastAPI lifespan, CLI bootstrap,
SDK bootstrap) and torn down at shutdown.

Public surface:

* :class:`BackgroundTaskExecutor` — ABC for one type of background
  task. Yields output bytes; raises on failure.
* :class:`LocalBashExecutor` — runs a shell command via subprocess.
* :class:`LocalAgentExecutor` — runs a subagent type via the
  :class:`SubagentTypeOrchestrator`.
* :class:`BackgroundTaskRunner` — owns the :class:`asyncio.Task`
  futures, talks to a :class:`TaskRegistry` for state + output
  persistence, and supports submit / stop / shutdown.
"""

from geny_executor.runtime.task_executors import (
    BackgroundTaskExecutor,
    LocalAgentExecutor,
    LocalBashExecutor,
)
from geny_executor.runtime.task_runner import BackgroundTaskRunner

__all__ = [
    "BackgroundTaskExecutor",
    "BackgroundTaskRunner",
    "LocalAgentExecutor",
    "LocalBashExecutor",
]
