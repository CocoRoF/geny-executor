"""Cycle A+B cross-import audit (PR-C.1).

Verifies every public surface from new-executor-uplift cycles A+B
is reachable through its top-level package import. Catches
``__init__.py`` re-export drift before downstream Geny-side wiring
chases a broken import.

Treat any failure here as a regression: the contract with adopters
(Geny / future hosts) is "import the package, get the symbol".
"""

from __future__ import annotations

import importlib
from typing import Iterable, Tuple

import pytest


# ── Cycle A executor side: 1.1.0 surface ──────────────────────────────


CYCLE_A_EXPORTS: Tuple[Tuple[str, str], ...] = (
    # P0.1 task lifecycle
    ("geny_executor.stages.s13_task_registry", "TaskRegistry"),
    ("geny_executor.stages.s13_task_registry", "TaskRecord"),
    ("geny_executor.stages.s13_task_registry", "TaskFilter"),
    ("geny_executor.stages.s13_task_registry", "TaskStatus"),
    ("geny_executor.stages.s13_task_registry", "InMemoryRegistry"),
    ("geny_executor.stages.s13_task_registry", "FileBackedRegistry"),
    ("geny_executor.runtime", "BackgroundTaskRunner"),
    ("geny_executor.runtime", "BackgroundTaskExecutor"),
    ("geny_executor.runtime", "LocalBashExecutor"),
    ("geny_executor.runtime", "LocalAgentExecutor"),
    ("geny_executor.tools.built_in", "AgentTool"),
    ("geny_executor.tools.built_in", "TaskCreateTool"),
    ("geny_executor.tools.built_in", "TaskGetTool"),
    ("geny_executor.tools.built_in", "TaskListTool"),
    ("geny_executor.tools.built_in", "TaskUpdateTool"),
    ("geny_executor.tools.built_in", "TaskOutputTool"),
    ("geny_executor.tools.built_in", "TaskStopTool"),
    # P0.2 slash commands
    ("geny_executor.slash_commands", "SlashCommand"),
    ("geny_executor.slash_commands", "SlashCommandRegistry"),
    ("geny_executor.slash_commands", "SlashContext"),
    ("geny_executor.slash_commands", "SlashResult"),
    ("geny_executor.slash_commands", "SlashCategory"),
    ("geny_executor.slash_commands", "parse_slash"),
    ("geny_executor.slash_commands", "get_default_registry"),
    ("geny_executor.slash_commands.md_template", "MdTemplateCommand"),
    ("geny_executor.slash_commands.md_template", "load_md_command"),
    # P0.3 tool catalog
    ("geny_executor.tools.built_in", "AskUserQuestionTool"),
    ("geny_executor.tools.built_in", "PushNotificationTool"),
    ("geny_executor.tools.built_in", "MCPTool"),
    ("geny_executor.tools.built_in", "ListMcpResourcesTool"),
    ("geny_executor.tools.built_in", "ReadMcpResourceTool"),
    ("geny_executor.tools.built_in", "McpAuthTool"),
    ("geny_executor.tools.built_in", "EnterWorktreeTool"),
    ("geny_executor.tools.built_in", "ExitWorktreeTool"),
    ("geny_executor.tools.built_in", "LSPTool"),
    ("geny_executor.tools.built_in", "REPLTool"),
    ("geny_executor.tools.built_in", "BriefTool"),
    ("geny_executor.tools.built_in", "ConfigTool"),
    ("geny_executor.tools.built_in", "MonitorTool"),
    ("geny_executor.tools.built_in", "SendUserFileTool"),
    ("geny_executor.tools.built_in", "SendMessageTool"),
    ("geny_executor.notifications", "NotificationEndpoint"),
    ("geny_executor.notifications", "NotificationEndpointRegistry"),
    ("geny_executor.channels", "UserFileChannel"),
    ("geny_executor.channels", "SendMessageChannel"),
    ("geny_executor.channels", "SendMessageChannelRegistry"),
    ("geny_executor.channels", "StdoutSendMessageChannel"),
    # P0.4 cron
    ("geny_executor.cron", "CronJob"),
    ("geny_executor.cron", "CronJobStatus"),
    ("geny_executor.cron", "CronJobStore"),
    ("geny_executor.cron", "InMemoryCronJobStore"),
    ("geny_executor.cron", "FileBackedCronJobStore"),
    ("geny_executor.cron", "CronRunner"),
    ("geny_executor.tools.built_in", "CronCreateTool"),
    ("geny_executor.tools.built_in", "CronDeleteTool"),
    ("geny_executor.tools.built_in", "CronListTool"),
)


# ── Cycle B executor side: 1.2.0 surface ──────────────────────────────


CYCLE_B_EXPORTS: Tuple[Tuple[str, str], ...] = (
    # P1.1 in-process hooks (method on existing class — verified separately)
    # P1.2 auto-compaction
    ("geny_executor.stages.s19_summarize", "FrequencyPolicy"),
    ("geny_executor.stages.s19_summarize", "NeverPolicy"),
    ("geny_executor.stages.s19_summarize", "EveryNTurnsPolicy"),
    ("geny_executor.stages.s19_summarize", "OnContextFillPolicy"),
    ("geny_executor.stages.s19_summarize", "FrequencyAwareSummarizerProxy"),
    # P1.3 settings
    ("geny_executor.settings", "SettingsLoader"),
    ("geny_executor.settings", "get_default_loader"),
    ("geny_executor.settings", "register_section"),
    # P1.4 skill schema (extra fields on existing class — verified separately)
    # P1.5 permission modes (enum members — verified separately)
)


@pytest.mark.parametrize("module,name", CYCLE_A_EXPORTS + CYCLE_B_EXPORTS)
def test_public_surface_importable(module: str, name: str):
    mod = importlib.import_module(module)
    assert hasattr(mod, name), f"{module} missing public name {name!r}"


# ── Method / attr surface checks ──────────────────────────────────────


def test_hook_runner_has_register_in_process():
    """PR-B.1.1 — register_in_process must be on HookRunner instance."""
    from geny_executor.hooks.runner import HookRunner
    assert hasattr(HookRunner, "register_in_process")


def test_hook_runner_has_list_in_process_handlers():
    from geny_executor.hooks.runner import HookRunner
    assert hasattr(HookRunner, "list_in_process_handlers")


def test_skill_metadata_has_richer_fields():
    """PR-B.4.1 — category / effort / examples on SkillMetadata."""
    from dataclasses import fields
    from geny_executor.skills.types import SkillMetadata
    field_names = {f.name for f in fields(SkillMetadata)}
    for new_field in ("category", "effort", "examples"):
        assert new_field in field_names, f"SkillMetadata missing {new_field!r}"


def test_permission_mode_has_new_modes():
    """PR-B.5.1 — ACCEPT_EDITS / DONT_ASK enum members."""
    from geny_executor.permission.types import PermissionMode
    assert PermissionMode("acceptEdits") is PermissionMode.ACCEPT_EDITS
    assert PermissionMode("dontAsk") is PermissionMode.DONT_ASK


def test_permission_edit_tools_set_exported():
    from geny_executor.permission.types import EDIT_TOOLS
    assert "Write" in EDIT_TOOLS
    assert "Edit" in EDIT_TOOLS
    assert "NotebookEdit" in EDIT_TOOLS


def test_built_in_tool_classes_includes_all_new():
    """The registry mapping must enumerate every tool added in 1.1.0."""
    from geny_executor.tools.built_in import BUILT_IN_TOOL_CLASSES
    new_tools = (
        "Agent", "AskUserQuestion", "PushNotification",
        "MCP", "ListMcpResources", "ReadMcpResource", "McpAuth",
        "EnterWorktree", "ExitWorktree",
        "LSP", "REPL", "Brief",
        "Config", "Monitor", "SendUserFile", "SendMessage",
        "TaskCreate", "TaskGet", "TaskList", "TaskUpdate", "TaskOutput", "TaskStop",
        "CronCreate", "CronDelete", "CronList",
    )
    missing = [t for t in new_tools if t not in BUILT_IN_TOOL_CLASSES]
    assert not missing, f"BUILT_IN_TOOL_CLASSES missing: {missing}"


def test_built_in_tool_features_groups_present():
    from geny_executor.tools.built_in import BUILT_IN_TOOL_FEATURES
    expected_groups = {
        "agent", "tasks", "interaction", "notification",
        "mcp", "worktree", "dev", "operator", "messaging", "cron",
    }
    missing = expected_groups - set(BUILT_IN_TOOL_FEATURES)
    assert not missing, f"BUILT_IN_TOOL_FEATURES missing groups: {missing}"
