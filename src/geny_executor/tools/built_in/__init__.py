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
    "workflow": ["TodoWrite"],
    "meta": ["ToolSearch", "EnterPlanMode", "ExitPlanMode"],
    "agent": ["Agent"],
    "tasks": ["TaskCreate", "TaskGet", "TaskList", "TaskUpdate", "TaskOutput", "TaskStop"],
    "interaction": ["AskUserQuestion"],
    "notification": ["PushNotification"],
    "mcp": ["MCP", "ListMcpResources", "ReadMcpResource", "McpAuth"],
    "worktree": ["EnterWorktree", "ExitWorktree"],
    "dev": ["LSP", "REPL", "Brief"],
    "operator": ["Config", "Monitor", "SendUserFile"],
    "messaging": ["SendMessage"],
    "cron": ["CronCreate", "CronDelete", "CronList"],
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
    "BUILT_IN_TOOL_CLASSES",
    "BUILT_IN_TOOL_FEATURES",
    "get_builtin_tools",
]
