"""Subprocess hook runner.

Cycle 20260424 executor uplift — Phase 5 Week 9.

For each registered :class:`HookConfigEntry` matching a fired event,
``HookRunner`` spawns the configured subprocess, sends the
:class:`HookEventPayload` as JSON on stdin, reads JSON from stdout,
and parses it into a :class:`HookOutcome`. Multiple matching entries
are fired in declaration order and combined via
:meth:`HookOutcome.combine` (most-restrictive wins).

Safety + ergonomics:

* **Opt-in.** A runner constructed from a disabled :class:`HookConfig`
  (``enabled=False``) or in an environment without
  ``GENY_ALLOW_HOOKS=1`` short-circuits every fire to passthrough.
  Both switches must be true to actually invoke subprocesses.
* **Subprocess execution.** Always ``asyncio.create_subprocess_exec``
  with an explicit argv list — never ``shell=True``.
* **Timeout.** Per-entry ``timeout_ms`` enforced via
  ``asyncio.wait_for``. Timeout → kill, log WARNING, fail-open
  passthrough so a slow hook never blocks the agent.
* **Crash isolation.** Subprocess non-zero exit, non-JSON stdout,
  permission denied — every failure mode produces a passthrough
  outcome plus a WARNING log. The pipeline keeps moving.
* **Audit log.** Optional JSONL sink (``audit_log_path``) records one
  line per invocation with event, command, exit code, latency,
  outcome summary. Hosts that want richer telemetry attach a custom
  callback via :meth:`HookRunner.set_audit_callback`.

See ``executor_uplift/12_detailed_plan.md`` §5.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from geny_executor.hooks.config import (
    HookConfig,
    HookConfigEntry,
    hooks_opt_in_from_env,
)
from geny_executor.hooks.events import HookEvent, HookEventPayload, HookOutcome

logger = logging.getLogger(__name__)


AuditCallback = Callable[[Dict[str, Any]], Awaitable[None]]


class HookRunner:
    """Spawns subprocess hooks and combines their outcomes.

    Construct once per pipeline (or per session if hot-reloading
    config) and call :meth:`fire` for each event. The runner owns the
    audit log file handle; call :meth:`close` during teardown.

    Thread / loop safety: every invocation runs through asyncio and
    is safe to call from multiple coroutines on the same loop.
    Concurrent fires for the same event spawn separate subprocesses
    (no internal serialisation) — hooks should be self-contained.
    """

    def __init__(
        self,
        config: HookConfig,
        *,
        env: Optional[Dict[str, str]] = None,
        audit_callback: Optional[AuditCallback] = None,
    ):
        self._config = config
        self._env = dict(env) if env is not None else dict(os.environ)
        self._opt_in = hooks_opt_in_from_env(self._env)
        self._audit_callback = audit_callback
        self._audit_path = Path(config.audit_log_path) if config.audit_log_path else None

    @property
    def enabled(self) -> bool:
        """True if both the config and the env opt-in are set."""
        return self._config.enabled and self._opt_in

    @property
    def config(self) -> HookConfig:
        return self._config

    def set_audit_callback(self, callback: Optional[AuditCallback]) -> None:
        """Set or clear the audit callback (called once per invocation)."""
        self._audit_callback = callback

    async def fire(
        self,
        event: HookEvent,
        payload: HookEventPayload,
    ) -> HookOutcome:
        """Fire all hooks matching ``event`` and combine their outcomes.

        Returns :meth:`HookOutcome.passthrough` when:

        * the runner is disabled (config.enabled is False or
          ``GENY_ALLOW_HOOKS`` is unset);
        * no entries are registered for the event;
        * no entries match the payload (e.g. tool-name filter mismatch).

        Otherwise returns the combined outcome of every matching hook.
        """
        if not self.enabled:
            return HookOutcome.passthrough()

        entries = self._config.entries_for(event)
        if not entries:
            return HookOutcome.passthrough()

        matches: List[HookConfigEntry] = [e for e in entries if e.matches(event, payload.tool_name)]
        if not matches:
            return HookOutcome.passthrough()

        outcome = HookOutcome.passthrough()
        for entry in matches:
            entry_outcome = await self._invoke_one(entry, event, payload)
            outcome = outcome.combine(entry_outcome)
            if outcome.blocked:
                # Stop firing further hooks once the operation is
                # already blocked — a downstream audit hook can't
                # un-block, and we'd just be wasting subprocess spawns.
                break
        return outcome

    async def _invoke_one(
        self,
        entry: HookConfigEntry,
        event: HookEvent,
        payload: HookEventPayload,
    ) -> HookOutcome:
        """Spawn one hook process and parse its outcome.

        Always returns a :class:`HookOutcome` — failures map to
        passthrough + a WARNING log so the pipeline never dies on a
        broken hook script.
        """
        stdin_payload = json.dumps(payload.to_json_dict(), ensure_ascii=False).encode("utf-8")
        env = dict(self._env)
        env.update(entry.env)

        t0 = time.monotonic()
        exit_code: Optional[int] = None
        stdout_bytes = b""
        stderr_bytes = b""
        outcome = HookOutcome.passthrough()
        error_label: Optional[str] = None

        try:
            proc = await asyncio.create_subprocess_exec(
                entry.command,
                *entry.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=entry.working_dir,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(input=stdin_payload),
                    timeout=entry.timeout_ms / 1000.0,
                )
            except asyncio.TimeoutError:
                error_label = "timeout"
                proc.kill()
                # Best-effort drain; ignore further failures.
                try:
                    await proc.communicate()
                except Exception:
                    pass
                logger.warning(
                    "hook %s for event %s timed out after %dms — fail-open passthrough",
                    entry.command,
                    event.value,
                    entry.timeout_ms,
                )
            else:
                exit_code = proc.returncode
                if exit_code != 0:
                    error_label = f"exit_code={exit_code}"
                    logger.warning(
                        "hook %s for event %s exited %d — fail-open passthrough; stderr: %s",
                        entry.command,
                        event.value,
                        exit_code,
                        stderr_bytes.decode("utf-8", errors="replace")[:500],
                    )
                else:
                    parsed = self._parse_stdout(stdout_bytes, entry, event)
                    if parsed is not None:
                        outcome = parsed
        except FileNotFoundError:
            error_label = "command_not_found"
            logger.warning(
                "hook command not found: %s for event %s — fail-open passthrough",
                entry.command,
                event.value,
            )
        except PermissionError:
            error_label = "permission_denied"
            logger.warning(
                "hook command not executable: %s for event %s — fail-open passthrough",
                entry.command,
                event.value,
            )
        except Exception as exc:  # pragma: no cover - defensive
            error_label = "spawn_error"
            logger.warning(
                "hook %s for event %s spawn failed: %s — fail-open passthrough",
                entry.command,
                event.value,
                exc,
                exc_info=True,
            )

        latency_ms = int((time.monotonic() - t0) * 1000)
        await self._record_audit(
            event=event,
            payload=payload,
            entry=entry,
            outcome=outcome,
            exit_code=exit_code,
            latency_ms=latency_ms,
            error=error_label,
            stdout_preview=stdout_bytes[:500].decode("utf-8", errors="replace"),
        )
        return outcome

    def _parse_stdout(
        self,
        stdout_bytes: bytes,
        entry: HookConfigEntry,
        event: HookEvent,
    ) -> Optional[HookOutcome]:
        """Parse the hook's stdout into a :class:`HookOutcome`.

        Empty stdout → no outcome change (passthrough). Non-JSON
        stdout is logged at WARNING and treated as passthrough — we
        never want a hook script's typo to silently override the
        engine's permission decisions.
        """
        text = stdout_bytes.decode("utf-8", errors="replace").strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(
                "hook %s for event %s returned non-JSON stdout — fail-open passthrough; "
                "first 200 bytes: %r",
                entry.command,
                event.value,
                text[:200],
            )
            return None
        if not isinstance(parsed, dict):
            logger.warning(
                "hook %s for event %s returned non-object JSON (%s) — fail-open passthrough",
                entry.command,
                event.value,
                type(parsed).__name__,
            )
            return None
        try:
            return HookOutcome.from_response(parsed)
        except Exception:  # pragma: no cover - HookOutcome.from_response is forgiving
            logger.warning(
                "hook %s for event %s returned an outcome we couldn't parse",
                entry.command,
                event.value,
                exc_info=True,
            )
            return None

    async def _record_audit(
        self,
        *,
        event: HookEvent,
        payload: HookEventPayload,
        entry: HookConfigEntry,
        outcome: HookOutcome,
        exit_code: Optional[int],
        latency_ms: int,
        error: Optional[str],
        stdout_preview: str,
    ) -> None:
        """Append one audit line and call the audit callback."""
        record: Dict[str, Any] = {
            "event": event.value,
            "session_id": payload.session_id,
            "timestamp": payload.timestamp,
            "command": entry.command,
            "args": list(entry.args),
            "tool_name": payload.tool_name,
            "exit_code": exit_code,
            "latency_ms": latency_ms,
            "outcome": {
                "continue": outcome.continue_,
                "decision": outcome.decision,
                "suppress_output": outcome.suppress_output,
                "blocked": outcome.blocked,
                "stop_reason": outcome.stop_reason,
            },
        }
        if error is not None:
            record["error"] = error
        if stdout_preview:
            record["stdout_preview"] = stdout_preview

        if self._audit_path is not None:
            try:
                self._audit_path.parent.mkdir(parents=True, exist_ok=True)
                with self._audit_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError as exc:
                logger.warning("hook audit log write failed: %s", exc)

        if self._audit_callback is not None:
            try:
                await self._audit_callback(record)
            except Exception:  # pragma: no cover - defensive
                logger.warning("hook audit callback raised; ignored", exc_info=True)
