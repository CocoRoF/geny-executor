"""Phase 2 — built-in tool capability flags.

Confirms the read-only built-ins (Read / Grep / Glob) now advertise
``concurrency_safe=True`` so PartitionExecutor / StreamingToolExecutor
can fan them out instead of serialising behind the fail-closed default.
Write/Edit/Bash stay on the default (unsafe) — they either mutate
state or run arbitrary commands.
"""

from __future__ import annotations

from geny_executor.tools.built_in.bash_tool import BashTool
from geny_executor.tools.built_in.edit_tool import EditTool
from geny_executor.tools.built_in.glob_tool import GlobTool
from geny_executor.tools.built_in.grep_tool import GrepTool
from geny_executor.tools.built_in.read_tool import ReadTool
from geny_executor.tools.built_in.write_tool import WriteTool


def test_read_is_concurrency_safe():
    caps = ReadTool().capabilities({"file_path": "/tmp/x"})
    assert caps.concurrency_safe is True
    assert caps.read_only is True
    assert caps.idempotent is True
    assert caps.destructive is False


def test_grep_is_concurrency_safe():
    caps = GrepTool().capabilities({"pattern": "foo"})
    assert caps.concurrency_safe is True
    assert caps.read_only is True
    assert caps.idempotent is True


def test_glob_is_concurrency_safe():
    caps = GlobTool().capabilities({"pattern": "**/*.py"})
    assert caps.concurrency_safe is True
    assert caps.read_only is True
    assert caps.idempotent is True


def test_write_stays_unsafe_default():
    caps = WriteTool().capabilities({"file_path": "/tmp/x", "content": "y"})
    assert caps.concurrency_safe is False


def test_edit_stays_unsafe_default():
    caps = EditTool().capabilities({"file_path": "/tmp/x", "old_string": "a", "new_string": "b"})
    assert caps.concurrency_safe is False


def test_bash_stays_unsafe_default():
    caps = BashTool().capabilities({"command": "echo hi"})
    assert caps.concurrency_safe is False
