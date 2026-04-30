"""Shell-block execution for skill bodies.

Phase 10.3 (Skills uplift) — skills can embed shell commands directly
in their markdown body. The renderer parses two forms:

* **Fenced**::

      ```!
      git diff origin/main...HEAD
      ```

  Triple-backtick block opened with ``!`` (no language tag, the
  bang is the language marker). The whole inner text is fed to the
  configured shell. Multi-line scripts work fine.

* **Inline**::

      Use !`git rev-parse HEAD` to get the current commit.

  ``!`` followed by a single-backtick literal. One-line, intended for
  inline substitution like the example above.

Both forms are *replaced in the body* with the captured stdout (and
stderr if the command failed) before the rendered body is handed to
the LLM. The LLM never sees the raw `!`...`` ` syntax — only the
result text. This means the model does not need a separate Bash tool
call to learn what a skill knows about the host's state.

Security notes:

* Execution honours the skill's permission grants made by Phase 10.2
  (`SkillTool._grant_allowed_tools` adds ALLOW rules to the live
  ToolContext). The shell command runs as a single subprocess —
  Phase 10.2's grants don't gate the subprocess directly, but the
  *containing skill* has been deemed safe to run, and that skill
  documents the tools it'll touch.
* MCP-sourced skills are explicitly **stripped of shell blocks** by
  ``execute_blocks(..., trust_shell=False)`` — bodies coming from
  remote prompt servers are untrusted code paths. Hosts call
  ``trust_shell=False`` for any skill whose ``source_kind`` is MCP
  or otherwise untrusted.
* Each block has a per-skill wall-clock timeout (frontmatter
  ``shell_timeout_s``, default 30s).
* The skill's ``working_dir`` (from ``ToolContext.working_dir``) is
  the cwd. Empty / missing → process default cwd.
* ``ToolContext.env_vars`` is overlaid onto ``os.environ`` for the
  subprocess, so hosts that inject SECRETS for a session see them
  too.

Implementation is sync (subprocess) for simplicity. The skill's
overall execution path stays async; we run the subprocess in a
default executor when called from async code.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# ── Block parsing ─────────────────────────────────────────────────────


# Fenced ``` ``` ! ``` ``` ``` pattern. The ``!`` immediately follows
# the opening triple-backtick; no language tag. Captures everything
# until the closing fence.
_FENCED_RE = re.compile(r"```!\s*\n(.*?)\n```", re.DOTALL)

# Inline ``!`...`` pattern. The ``!`` is right before the opening
# backtick, the closing backtick ends the block. The inner text is one
# line — newlines aren't allowed (we use ``[^`\n]+`` to make sure a
# stray ``!`` in prose doesn't accidentally swallow paragraphs).
_INLINE_RE = re.compile(r"!`([^`\n]+)`")


@dataclass
class ShellBlock:
    """One parsed block. ``span`` marks where in the original body it
    sits so the substitution can splice the result back in."""

    kind: str  # "fenced" | "inline"
    command: str
    span: tuple  # (start, end) in the source body


def parse_blocks(body: str) -> List[ShellBlock]:
    """Find every shell block in ``body``. Order is left-to-right
    (the order the LLM would read them)."""
    blocks: List[ShellBlock] = []
    for m in _FENCED_RE.finditer(body):
        blocks.append(ShellBlock(kind="fenced", command=m.group(1), span=m.span()))
    for m in _INLINE_RE.finditer(body):
        # Skip inline matches that fall inside a fenced span — a fenced
        # block may legitimately contain ``!`echo hi`...`` inside its
        # content and we don't want to double-execute.
        start, end = m.span()
        if any(b.span[0] <= start < b.span[1] for b in blocks if b.kind == "fenced"):
            continue
        blocks.append(ShellBlock(kind="inline", command=m.group(1), span=m.span()))
    blocks.sort(key=lambda b: b.span[0])
    return blocks


# ── Execution ────────────────────────────────────────────────────────


@dataclass
class ShellRunOutcome:
    """One block's execution result. ``rendered`` is the text that
    should replace the block in the body."""

    block: ShellBlock
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    skipped: bool = False
    skipped_reason: Optional[str] = None

    @property
    def rendered(self) -> str:
        if self.skipped:
            # Leave a clear marker — skipped blocks still need to
            # appear in the body so the LLM doesn't get confused
            # about missing context.
            return f"[shell skipped: {self.skipped_reason or 'untrusted'}]"
        if self.timed_out:
            return "[shell timed out after the configured limit]"
        text = self.stdout
        if self.exit_code != 0:
            err = self.stderr.strip()
            if err:
                text = f"{text}\n[shell exit={self.exit_code}: {err}]"
            else:
                text = f"{text}\n[shell exit={self.exit_code}]"
        return text.rstrip()


@dataclass
class ShellRunSummary:
    """Aggregate outcome of running every block in a body."""

    rendered_body: str
    outcomes: List[ShellRunOutcome] = field(default_factory=list)
    any_failed: bool = False
    any_skipped: bool = False


def _run_one(
    block: ShellBlock,
    *,
    shell: str,
    cwd: Optional[str],
    env: Optional[Dict[str, str]],
    timeout_s: float,
) -> ShellRunOutcome:
    """Synchronous subprocess invocation. Called from a default
    executor by the async ``execute_blocks`` so the pipeline event
    loop never blocks."""
    import os
    import subprocess

    shell_path = shutil.which(shell) or shell
    cmd = [shell_path, "-c", block.command]
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd if cwd else None,
            env=full_env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ShellRunOutcome(
            block=block,
            exit_code=-1,
            stdout="",
            stderr="",
            timed_out=True,
        )
    except FileNotFoundError as exc:
        return ShellRunOutcome(
            block=block,
            exit_code=-1,
            stdout="",
            stderr=f"shell not found: {exc}",
        )
    return ShellRunOutcome(
        block=block,
        exit_code=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )


async def execute_blocks(
    body: str,
    *,
    shell: str = "bash",
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    timeout_s: float = 30.0,
    trust_shell: bool = True,
) -> ShellRunSummary:
    r"""Find every shell block in ``body``, execute it, and return the
    body with blocks replaced by their captured output.

    Args:
        body: Skill body, possibly with ``\`\`\`!``...``\`\`\`\`` and
            ``!\`...\``` blocks.
        shell: Shell binary to invoke (default ``bash``). Resolved via
            ``shutil.which`` so a missing shell errors immediately
            rather than after every block.
        cwd: Working directory for the subprocess. ``None`` / empty →
            inherits from the executor's process.
        env: Environment overlay. Merged onto ``os.environ`` for the
            subprocess so hosts can inject session-scoped vars
            without leaking host-wide changes.
        timeout_s: Wall-clock timeout per block.
        trust_shell: When ``False`` every block is *skipped* and
            replaced with a clear marker. Hosts pass ``False`` for
            MCP-sourced skills and any other skill whose body comes
            from an untrusted remote.

    Returns:
        :class:`ShellRunSummary` with the rendered body + per-block
        outcomes for audit.
    """
    blocks = parse_blocks(body)
    if not blocks:
        return ShellRunSummary(rendered_body=body)

    outcomes: List[ShellRunOutcome] = []
    if not trust_shell:
        # Skip every block; build outcomes purely for telemetry.
        for b in blocks:
            outcomes.append(
                ShellRunOutcome(
                    block=b,
                    exit_code=-1,
                    stdout="",
                    stderr="",
                    skipped=True,
                    skipped_reason="skill body is untrusted (trust_shell=False)",
                )
            )
    else:
        loop = asyncio.get_running_loop()
        for b in blocks:
            outcome = await loop.run_in_executor(
                None,
                lambda blk=b: _run_one(
                    blk,
                    shell=shell,
                    cwd=cwd,
                    env=env,
                    timeout_s=timeout_s,
                ),
            )
            outcomes.append(outcome)

    rendered = _splice(body, outcomes)
    any_failed = any((not o.skipped) and (o.exit_code != 0 or o.timed_out) for o in outcomes)
    any_skipped = any(o.skipped for o in outcomes)
    return ShellRunSummary(
        rendered_body=rendered,
        outcomes=outcomes,
        any_failed=any_failed,
        any_skipped=any_skipped,
    )


def _splice(body: str, outcomes: Sequence[ShellRunOutcome]) -> str:
    """Replace each outcome's span with its rendered text. Walk
    right-to-left so earlier spans aren't shifted."""
    result = body
    for outcome in sorted(outcomes, key=lambda o: o.block.span[0], reverse=True):
        start, end = outcome.block.span
        result = result[:start] + outcome.rendered + result[end:]
    return result


# ── Source-kind helper ───────────────────────────────────────────────


def is_trusted_source(source: Optional[Path], extras: Dict) -> bool:
    """Decide whether to honour shell blocks in a skill's body.

    A skill is trusted when it comes from disk (the operator put it
    there) or was registered in-code by the executor / a host. MCP-
    bridged skills (advertised by remote MCP servers) are *never*
    trusted — their ``extras`` carry an explicit marker we check.

    Defaults to trusted when the marker is absent so existing
    in-code bundled skills continue to work without an opt-in.
    """
    # Convention: ``extras["source_kind"] == "mcp"`` means the bridge
    # built this skill from an MCP prompt definition.
    kind = extras.get("source_kind") if isinstance(extras, dict) else None
    if kind == "mcp":
        return False
    # Skills with no on-disk source AND no extras hint are bundled
    # in-code by trusted code paths — trust them.
    return True
