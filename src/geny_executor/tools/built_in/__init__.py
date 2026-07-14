"""Built-in tools for file system operations, shell execution, and search.

These tools provide the core capabilities that an agent needs to interact
with the local environment — reading/writing files, running commands,
and searching codebases. They ship with the executor so every consumer
gets a working tool surface without having to reimplement filesystem
access against the :class:`~geny_executor.tools.base.Tool` ABC.

:data:`BUILT_IN_TOOL_CLASSES` maps each tool's registry name to its
class; it is the single source of truth consumed by
``Pipeline.from_manifest_async`` when resolving
``manifest.tools.built_in`` entries.

:data:`BUILT_IN_TOOL_FEATURES` groups those same tools by capability
family (``filesystem`` / ``shell`` / ``web`` / ``workflow``). Use
:func:`get_builtin_tools` with the ``features=`` kwarg to select a
subset without hardcoding tool names.
"""

from typing import Dict, Iterable, List, Optional, Type

from geny_executor.tools.base import Tool
from geny_executor.tools.built_in.agent_tool import AgentTool
from geny_executor.tools.built_in.subagent_tools import (
    SubAgentAssignTool,
    SubAgentInboxReadTool,
    SubAgentListTool,
    SubAgentSpawnTool,
    SubAgentStopTool,
)
from geny_executor.tools.built_in.ask_user_question_tool import (
    AskUserQuestionTool,
    QuestionCancelled,
)
from geny_executor.tools.built_in.mcp_wrapper_tools import (
    ListMcpResourcesTool,
    MCPTool,
    McpAuthTool,
    ReadMcpResourceTool,
)
from geny_executor.tools.built_in.push_notification_tool import (
    PushNotificationTool,
)
from geny_executor.tools.built_in.dev_tools import (
    BriefTool,
    LSPTool,
    REPLTool,
)
from geny_executor.tools.built_in.operator_tools import (
    ConfigTool,
    MonitorTool,
    SendUserFileTool,
)
from geny_executor.tools.built_in.read_tool import ReadTool
from geny_executor.tools.built_in.workspace_tools import (
    SandboxFetchTool,
    SandboxInfoTool,
    SandboxPutTool,
    WorkspaceInfoTool,
)
from geny_executor.tools.built_in.cron_tools import (
    CronCreateTool,
    CronDeleteTool,
    CronListTool,
)
from geny_executor.tools.built_in.send_message_tool import SendMessageTool
from geny_executor.tools.built_in.worktree_tools import (
    EnterWorktreeTool,
    ExitWorktreeTool,
)
from geny_executor.tools.built_in.task_tools import (
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskOutputTool,
    TaskStopTool,
    TaskUpdateTool,
)
from geny_executor.tools.built_in.write_tool import WriteTool
from geny_executor.tools.built_in.edit_tool import EditTool
from geny_executor.tools.built_in.bash_tool import BashTool
from geny_executor.tools.built_in.glob_tool import GlobTool
from geny_executor.tools.built_in.grep_tool import GrepTool
from geny_executor.tools.built_in.web_fetch_tool import WebFetchTool
from geny_executor.tools.built_in.web_search_tool import WebSearchTool
from geny_executor.tools.built_in.todo_write_tool import TodoWriteTool
from geny_executor.tools.built_in.notebook_edit_tool import NotebookEditTool
from geny_executor.tools.built_in.tool_search_tool import ToolSearchTool
from geny_executor.tools.built_in.plan_mode_tools import (
    EnterPlanModeTool,
    ExitPlanModeTool,
)
from geny_executor.tools.built_in.env_tools import EnvTool
# Google Workspace — native Gmail/Calendar/Drive/Tasks tools. Read the OAuth token
# from ``ctx.extras['google']``; gated via required_config_keys → hidden until the
# host marks ``feature:google_connected`` satisfied.
from geny_executor.tools.built_in.google_tools import GOOGLE_TOOL_CLASSES
# Browser — AI-native web exploration on the an-web engine (semantic snapshots,
# per-session tabs, embedded V8; no Chromium). an-web itself imports lazily —
# 'pip install geny-executor[browser]' (Python >= 3.12).
from geny_executor.tools.built_in.browser_tools import (
    BROWSER_TOOL_CLASSES,
    BrowserActTool,
    BrowserBackTool,
    BrowserCloseTool,
    BrowserEvalTool,
    BrowserExtractTool,
    BrowserNavigateTool,
    BrowserSnapshotTool,
)
# Doc — office documents (docx/xlsx/pptx) on the edit2docs engine: addressable
# outlines, deterministic edits, generation. Lazy import — 'pip install
# geny-executor[docs]'.
from geny_executor.tools.built_in.doc_tools import (
    DOC_TOOL_CLASSES,
    DocAnalyzeTool,
    DocApplyEditsTool,
    DocBuildTool,
    DocEditTool,
    DocGenerateTool,
    DocRenderTool,
    DocXmlEditTool,
    DocXmlReadTool,
)
# NOT in BUILT_IN_TOOL_CLASSES: SandboxExecTool is instantiated per Sandbox Tool
# Pack (with a spec + a live SandboxHandle), not activated by a manifest name.
from geny_executor.tools.built_in.sandbox_exec_tool import SandboxExecTool
# SSH — run commands / move files on the session's pre-configured servers.
# Gated on feature:ssh_enabled; degrades to an install-hint error when the
# optional ``asyncssh`` dependency is absent.
from geny_executor.tools.built_in.ssh_tools import (
    SSH_TOOL_CLASSES,
    SshDownloadTool,
    SshListServersTool,
    SshRunTool,
    SshUploadTool,
)


BUILT_IN_TOOL_CLASSES: Dict[str, Type[Tool]] = {
    "Read": ReadTool,
    "Write": WriteTool,
    "Edit": EditTool,
    "Bash": BashTool,
    "Glob": GlobTool,
    "Grep": GrepTool,
    "WebFetch": WebFetchTool,
    "WebSearch": WebSearchTool,
    "TodoWrite": TodoWriteTool,
    "NotebookEdit": NotebookEditTool,
    "ToolSearch": ToolSearchTool,
    "EnterPlanMode": EnterPlanModeTool,
    "ExitPlanMode": ExitPlanModeTool,
    "Agent": AgentTool,
    "AskUserQuestion": AskUserQuestionTool,
    "PushNotification": PushNotificationTool,
    "MCP": MCPTool,
    "ListMcpResources": ListMcpResourcesTool,
    "ReadMcpResource": ReadMcpResourceTool,
    "McpAuth": McpAuthTool,
    "EnterWorktree": EnterWorktreeTool,
    "ExitWorktree": ExitWorktreeTool,
    "LSP": LSPTool,
    "REPL": REPLTool,
    "Brief": BriefTool,
    "Config": ConfigTool,
    "Monitor": MonitorTool,
    "SendUserFile": SendUserFileTool,
    "WorkspaceInfo": WorkspaceInfoTool,
    "SandboxInfo": SandboxInfoTool,
    "SandboxPut": SandboxPutTool,
    "SandboxFetch": SandboxFetchTool,
    "SendMessage": SendMessageTool,
    "CronCreate": CronCreateTool,
    "CronDelete": CronDeleteTool,
    "CronList": CronListTool,
    "TaskCreate": TaskCreateTool,
    "TaskGet": TaskGetTool,
    "TaskList": TaskListTool,
    "TaskUpdate": TaskUpdateTool,
    "TaskOutput": TaskOutputTool,
    "TaskStop": TaskStopTool,
    "SubAgentSpawn": SubAgentSpawnTool,
    "SubAgentAssign": SubAgentAssignTool,
    "SubAgentList": SubAgentListTool,
    "SubAgentStop": SubAgentStopTool,
    "SubAgentInboxRead": SubAgentInboxReadTool,
    # Self-modifying environment — one lean dispatcher; detailed guidance lives
    # in the bundled ``environment`` skill (progressive disclosure).
    "env": EnvTool,
    # Google Workspace (gated on feature:google_connected — hidden until the host
    # injects OAuth creds + marks Google connected).
    **GOOGLE_TOOL_CLASSES,
    # Browser (an-web) — semantic web exploration; degrades to an install-hint
    # error when the optional an-web dependency is absent.
    **BROWSER_TOOL_CLASSES,
    # Doc (edit2docs) — office document engine; same lazy-import contract.
    **DOC_TOOL_CLASSES,
    # SSH — command/SFTP on the session's configured servers (gated on
    # feature:ssh_enabled); lazy-imports asyncssh with an install-hint fallback.
    **SSH_TOOL_CLASSES,
}


# Feature groupings keep the catalog navigable as it grows. A tool may
# belong to exactly one family — the boundary is "which capability bucket
# does this power?", not "which source directory does it live in?" Hosts
# selecting by feature get a stable API even as we add, rename, or split
# individual tools.
BUILT_IN_TOOL_FEATURES: Dict[str, List[str]] = {
    "filesystem": ["Read", "Write", "Edit", "Glob", "Grep", "NotebookEdit"],
    "shell": ["Bash"],
    "web": ["WebFetch", "WebSearch"],
    # Interactive web exploration (an-web engine) — JS-rendered pages,
    # per-session tabs, semantic snapshots. Distinct from "web" (one-shot
    # fetch/search) so hosts can enable them independently.
    "browser": list(BROWSER_TOOL_CLASSES.keys()),
    # Office documents (edit2docs engine) — outline/edit/preview/generate.
    "documents": list(DOC_TOOL_CLASSES.keys()),
    "workflow": ["TodoWrite"],
    "meta": ["ToolSearch", "EnterPlanMode", "ExitPlanMode"],
    "agent": ["Agent"],
    "subagent": [
        "SubAgentSpawn",
        "SubAgentAssign",
        "SubAgentList",
        "SubAgentStop",
        "SubAgentInboxRead",
    ],
    "tasks": ["TaskCreate", "TaskGet", "TaskList", "TaskUpdate", "TaskOutput", "TaskStop"],
    "interaction": ["AskUserQuestion"],
    "notification": ["PushNotification"],
    "mcp": ["MCP", "ListMcpResources", "ReadMcpResource", "McpAuth"],
    "worktree": ["EnterWorktree", "ExitWorktree"],
    "dev": ["LSP", "REPL", "Brief"],
    "operator": ["Config", "Monitor", "SendUserFile"],
    # The session's two file spaces: inspect the host-side files workspace,
    # check the sandbox, and move files between them.
    "workspace": ["WorkspaceInfo", "SandboxInfo", "SandboxPut", "SandboxFetch"],
    "messaging": ["SendMessage"],
    "cron": ["CronCreate", "CronDelete", "CronList"],
    "environment": ["env"],
    "google": list(GOOGLE_TOOL_CLASSES.keys()),
    # Remote server ops over SSH/SFTP — run commands, transfer files, sudo.
    "ssh": list(SSH_TOOL_CLASSES.keys()),
}


def get_builtin_tools(
    *,
    features: Optional[Iterable[str]] = None,
    names: Optional[Iterable[str]] = None,
) -> Dict[str, Type[Tool]]:
    """Return a ``{tool_name: tool_class}`` mapping.

    Selection:
        * No args → every tool in :data:`BUILT_IN_TOOL_CLASSES`.
        * ``features=[...]`` → the union of every tool in those
          feature families (see :data:`BUILT_IN_TOOL_FEATURES`). An
          unknown feature name raises ``KeyError`` so typos surface
          at the call site rather than silently dropping tools.
        * ``names=[...]`` → exactly those tool names. An unknown name
          raises ``KeyError``. Can be combined with ``features`` to
          subtract or add specific entries from the feature union.

    The returned dict is fresh — callers may mutate it without
    affecting the registry constants.

    Examples:
        >>> sorted(get_builtin_tools(features=["filesystem"]).keys())
        ['Edit', 'Glob', 'Grep', 'Read', 'Write']

        >>> sorted(get_builtin_tools(features=["web"], names=["Read"]).keys())
        ['Read', 'WebFetch', 'WebSearch']
    """
    selected: Dict[str, Type[Tool]] = {}

    if features is None and names is None:
        return dict(BUILT_IN_TOOL_CLASSES)

    if features is not None:
        for feat in features:
            if feat not in BUILT_IN_TOOL_FEATURES:
                raise KeyError(
                    f"unknown built-in feature {feat!r}; "
                    f"known: {sorted(BUILT_IN_TOOL_FEATURES.keys())}"
                )
            for tool_name in BUILT_IN_TOOL_FEATURES[feat]:
                selected[tool_name] = BUILT_IN_TOOL_CLASSES[tool_name]

    if names is not None:
        for name in names:
            if name not in BUILT_IN_TOOL_CLASSES:
                raise KeyError(
                    f"unknown built-in tool {name!r}; known: {sorted(BUILT_IN_TOOL_CLASSES.keys())}"
                )
            selected[name] = BUILT_IN_TOOL_CLASSES[name]

    return selected


__all__ = [
    "AgentTool",
    "BROWSER_TOOL_CLASSES",
    "DOC_TOOL_CLASSES",
    "SSH_TOOL_CLASSES",
    "SshListServersTool",
    "SshRunTool",
    "SshUploadTool",
    "SshDownloadTool",
    "DocAnalyzeTool",
    "DocApplyEditsTool",
    "DocBuildTool",
    "DocEditTool",
    "DocGenerateTool",
    "DocRenderTool",
    "DocXmlEditTool",
    "DocXmlReadTool",
    "BrowserActTool",
    "BrowserBackTool",
    "BrowserCloseTool",
    "BrowserEvalTool",
    "BrowserExtractTool",
    "BrowserNavigateTool",
    "BrowserSnapshotTool",
    "SubAgentSpawnTool",
    "SubAgentAssignTool",
    "SubAgentListTool",
    "SubAgentStopTool",
    "SubAgentInboxReadTool",
    "AskUserQuestionTool",
    "BriefTool",
    "ConfigTool",
    "CronCreateTool",
    "CronDeleteTool",
    "CronListTool",
    "EnterWorktreeTool",
    "ExitWorktreeTool",
    "LSPTool",
    "ListMcpResourcesTool",
    "MonitorTool",
    "REPLTool",
    "SendMessageTool",
    "SendUserFileTool",
    "MCPTool",
    "McpAuthTool",
    "PushNotificationTool",
    "QuestionCancelled",
    "ReadMcpResourceTool",
    "ReadTool",
    "TaskCreateTool",
    "TaskGetTool",
    "TaskListTool",
    "TaskOutputTool",
    "TaskStopTool",
    "TaskUpdateTool",
    "WriteTool",
    "EditTool",
    "BashTool",
    "GlobTool",
    "GrepTool",
    "WebFetchTool",
    "WebSearchTool",
    "TodoWriteTool",
    "NotebookEditTool",
    "ToolSearchTool",
    "EnterPlanModeTool",
    "ExitPlanModeTool",
    "EnvTool",
    "SandboxExecTool",
    "WorkspaceInfoTool",
    "SandboxInfoTool",
    "SandboxPutTool",
    "SandboxFetchTool",
    "BUILT_IN_TOOL_CLASSES",
    "BUILT_IN_TOOL_FEATURES",
    "get_builtin_tools",
]
