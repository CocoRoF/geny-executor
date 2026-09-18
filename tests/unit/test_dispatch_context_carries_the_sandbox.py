"""Whatever ``attach_runtime`` stamps on the tool context must survive dispatch.

Stage 10 builds a FRESH ``ToolContext`` per dispatch, field by field, from
the stage's own context. Anything the copy forgets is silently dropped —
and it forgot ``sandbox``. The symptom on a real session (2026-09-19,
prod): a container was bound, the manager logged "tools sandboxed", the
system prompt said the workspace was at ``/workspace``, and then

    Write /workspace/verify.txt
    → Access denied: /workspace/verify.txt is outside allowed directories

because the tool took its HOST branch. ``SandboxInfo`` agreed:
``{"attached": false}``. Every sandboxed session burned a turn discovering
the prompt was wrong.

It went unseen for so long because the only sandboxed path anyone
exercised was the ``claude_code_cli`` provider, which ran its tools inside
the container itself and never asked Stage 10 for a context.

The field-by-field test below is deliberately a *list*, not a spot check:
the next field added to ToolContext should fail here until it is carried.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from geny_executor.core.pipeline import Pipeline
from geny_executor.core.state import PipelineState
from geny_executor.stages.s10_tool import ToolStage
from geny_executor.tools.base import ToolContext


class _FakeSandbox:
    container_name = "gapt-ws-abc"

    async def ensure(self) -> None:  # pragma: no cover — never spawned here
        return None


def _stage_with(ctx: ToolContext) -> ToolStage:
    stage = ToolStage()
    stage._context = ctx
    return stage


def test_the_sandbox_reaches_the_tools() -> None:
    sandbox = _FakeSandbox()
    stage = _stage_with(ToolContext(sandbox=sandbox))
    ctx = stage.build_dispatch_context(PipelineState(session_id="s1"))
    assert ctx.sandbox is sandbox, (
        "a dispatched tool saw no sandbox — its host branch runs instead, "
        "on a session that has a container bound to it"
    )


def test_attach_runtime_reaches_the_tools() -> None:
    """The host's actual call path, end to end."""
    p = Pipeline()
    p.register_stage(ToolStage())
    sandbox = _FakeSandbox()
    p.attach_runtime(sandbox=sandbox)

    stage = next(s for s in p._stages.values() if getattr(s, "name", "") == "tool")
    ctx = stage.build_dispatch_context(PipelineState(session_id="s1"))
    assert ctx.sandbox is sandbox


def test_no_sandbox_stays_none() -> None:
    stage = _stage_with(ToolContext())
    assert stage.build_dispatch_context(PipelineState(session_id="s1")).sandbox is None


def test_every_host_settable_field_survives_dispatch() -> None:
    """The copy is field by field, so each one is a chance to forget.

    ``sandbox`` was forgotten for releases. Anything a host can set on the
    stage context and a tool can read belongs in this list.
    """
    sandbox = _FakeSandbox()
    source = ToolContext(
        working_dir="/w",
        storage_path="/s",
        env_vars={"K": "V"},
        allowed_paths=["/w"],
        metadata={"m": 1},
        extras={"web_search": {"key": "x"}},
        sandbox=sandbox,
    )
    source.hook_runner = "HOOKS"          # type: ignore[assignment]
    source.permission_mode = "plan"
    source.permission_rules = ["rule"]    # type: ignore[assignment]
    source.environment = "ENV"            # type: ignore[assignment]

    ctx = _stage_with(source).build_dispatch_context(PipelineState(session_id="s1"))

    assert ctx.working_dir == "/w"
    assert ctx.storage_path == "/s"
    assert ctx.env_vars == {"K": "V"}
    assert ctx.allowed_paths == ["/w"]
    assert ctx.metadata == {"m": 1}
    assert ctx.extras == {"web_search": {"key": "x"}}
    assert ctx.hook_runner == "HOOKS"
    assert ctx.permission_mode == "plan"
    assert ctx.permission_rules == ["rule"]
    assert ctx.environment == "ENV"
    assert ctx.sandbox is sandbox
