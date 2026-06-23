"""SandboxExecTool — a tool whose implementation is *code that runs inside a
sandbox container* (``docker exec``).

This is the execution core of **Sandbox Tool Packs**: an agent authors a script
in an isolated GAPT workspace, and this tool dispatches the tool's input into
that script — inside the sandbox, never on the host — and returns its output.

Invocation contract (tool ↔ script), language-agnostic + shell-testable:

  * the validated tool ``input`` is passed as **JSON on stdin**;
  * the script prints its **result as JSON on stdout**
    (or ``{"error": "..."}`` to signal a handled failure);
  * a non-zero exit code (or anything on stderr with a non-zero exit) →
    ``ToolResult(is_error=True)``.

The tool carries a :class:`SandboxHandle` (``container_name`` + async
``ensure()``). It has **no host fallback** — without a sandbox it errors, by
design: a sandboxed tool's whole point is isolation.

The spec (everything except the live sandbox handle) is serializable via
:meth:`to_dict` / :meth:`from_dict`, so a host can persist the tool and rebuild
it later against a freshly-provisioned sandbox.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional, Sequence

from geny_executor.tools._sandbox import sandbox_exec
from geny_executor.tools.base import Tool, ToolCapabilities, ToolContext, ToolResult

_NO_SANDBOX = (
    "This sandboxed tool has no sandbox to run in (no SandboxHandle attached "
    "and ToolContext.sandbox is unset). Sandboxed tools execute their code "
    "inside an isolated container; without one they cannot run."
)

# Cap stderr we echo into metadata so a noisy script can't bloat the turn.
_STDERR_CAP = 2000


class SandboxExecTool(Tool):
    """A Tool that runs an authored script inside a sandbox via ``docker exec``."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        input_schema: Optional[Dict[str, Any]] = None,
        entrypoint: str,
        runtime: str = "python3",
        argv: Sequence[str] = (),
        timeout_s: float = 60.0,
        workdir: str = "/workspace",
        sandbox: Optional[Any] = None,
        network_egress: bool = False,
        read_only: bool = False,
    ) -> None:
        self._name = str(name)
        self._description = str(description)
        self._input_schema = input_schema or {"type": "object", "properties": {}}
        self._runtime = str(runtime)
        self._entrypoint = str(entrypoint)
        self._argv = [str(a) for a in (argv or ())]
        self._timeout_s = float(timeout_s)
        self._workdir = str(workdir or "/workspace")
        self._sandbox = sandbox
        self._network_egress = bool(network_egress)
        self._read_only = bool(read_only)

    # ── required Tool contract ────────────────────────────────────────
    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def input_schema(self) -> Dict[str, Any]:
        return self._input_schema

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        # Sandboxed code is isolated; default to serial + bounded. Authors may
        # declare read_only / network_egress in the spec to relax / annotate.
        return ToolCapabilities(
            concurrency_safe=self._read_only,
            read_only=self._read_only,
            network_egress=self._network_egress,
        )

    # ── late binding of the live sandbox (host wires it at load) ───────
    def attach_sandbox(self, sandbox: Optional[Any]) -> None:
        """Point this tool at the (freshly-provisioned) sandbox it runs in."""
        self._sandbox = sandbox

    # ── execution ─────────────────────────────────────────────────────
    async def execute(
        self, input: Dict[str, Any], context: Optional[ToolContext] = None
    ) -> ToolResult:
        sandbox = self._sandbox or (getattr(context, "sandbox", None) if context else None)
        if sandbox is None:
            return ToolResult(content=_NO_SANDBOX, is_error=True)

        argv = [self._runtime, self._entrypoint, *self._argv]
        payload = json.dumps(input or {}, ensure_ascii=False).encode("utf-8")
        try:
            rc, out, err = await sandbox_exec(
                sandbox,
                argv,
                cwd=self._workdir,
                input_bytes=payload,
                timeout_s=self._timeout_s,
            )
        except asyncio.TimeoutError:
            return ToolResult(
                content=f"sandboxed tool '{self._name}' timed out after {self._timeout_s:g}s",
                is_error=True,
                metadata={"timeout": True},
            )
        except Exception as exc:  # noqa: BLE001 — surface launch failures cleanly
            return ToolResult(
                content=f"sandboxed tool '{self._name}' failed to launch: {exc}",
                is_error=True,
            )

        stdout = out.decode("utf-8", "replace")
        stderr = err.decode("utf-8", "replace")
        if rc != 0:
            detail = stderr.strip() or stdout.strip() or f"exit code {rc}"
            return ToolResult(
                content=f"sandboxed tool '{self._name}' errored (exit {rc}): {detail}",
                is_error=True,
                metadata={"exit_code": rc, "stderr": stderr[:_STDERR_CAP]},
            )

        text = stdout.strip()
        # Lenient JSON contract: a {"error": ...} payload signals a handled fail.
        parsed: Any = None
        try:
            parsed = json.loads(text) if text else None
        except ValueError:
            parsed = None
        if isinstance(parsed, dict) and parsed.get("error"):
            return ToolResult(
                content=str(parsed["error"]), is_error=True, metadata={"exit_code": 0}
            )

        meta: Dict[str, Any] = {"exit_code": 0}
        if stderr.strip():
            meta["stderr"] = stderr[:_STDERR_CAP]
        return ToolResult(content=text, metadata=meta)

    # ── spec serialization (hosts persist this; sandbox is runtime) ────
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self._name,
            "description": self._description,
            "input_schema": self._input_schema,
            "runtime": self._runtime,
            "entrypoint": self._entrypoint,
            "argv": list(self._argv),
            "timeout_s": self._timeout_s,
            "workdir": self._workdir,
            "network_egress": self._network_egress,
            "read_only": self._read_only,
        }

    @classmethod
    def from_dict(cls, spec: Dict[str, Any], *, sandbox: Optional[Any] = None) -> "SandboxExecTool":
        return cls(
            name=spec["name"],
            description=spec.get("description", ""),
            input_schema=spec.get("input_schema"),
            entrypoint=spec["entrypoint"],
            runtime=spec.get("runtime", "python3"),
            argv=spec.get("argv", ()),
            timeout_s=spec.get("timeout_s", 60.0),
            workdir=spec.get("workdir", "/workspace"),
            sandbox=sandbox,
            network_egress=spec.get("network_egress", False),
            read_only=spec.get("read_only", False),
        )
