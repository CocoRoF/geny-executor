"""Tool system — registration, routing, execution, composition."""

from geny_executor.tools.base import (
    Tool,
    ToolResult,
    ToolContext,
    ToolCapabilities,
    PermissionDecision,
    build_tool,
)
from geny_executor.tools.registry import ToolRegistry
from geny_executor.tools.adhoc import (
    AdhocTool,
    AdhocToolDefinition,
    AdhocToolFactory,
    HttpToolConfig,
    ScriptToolConfig,
    TemplateToolConfig,
    CompositeToolConfig,
    CompositeStep,
)
from geny_executor.tools.composer import ToolComposer, ToolInfo, ToolPreset
from geny_executor.tools.providers import AdhocToolProvider
from geny_executor.tools.scope import ToolScope, ToolScopeRule, ToolScopeManager
from geny_executor.tools.sandbox import ToolSandbox, SandboxConfig, SandboxPolicy
# Container-exec primitives (``docker exec`` into a SandboxHandle) — public so
# hosts can build sandboxed tools. ``_sandbox`` only depends on stdlib, so this
# import is cycle-safe (unlike importing the built_in package here).
# ``SandboxExecTool`` itself lives in ``geny_executor.tools.built_in`` to keep
# this module free of the built-in import cycle.
from geny_executor.tools._sandbox import (
    SandboxExecError,
    container_path,
    sandbox_exec,
    sb_read_bytes,
    sb_run,
    sb_write_bytes,
)
from geny_executor.tools.plugin import (
    TOOL_ENTRY_POINT_GROUP,
    ToolPluginRegistry,
    discover_tool_plugins,
    register_tool_plugins,
)

__all__ = [
    # Base
    "Tool",
    "ToolResult",
    "ToolContext",
    "ToolCapabilities",
    "PermissionDecision",
    "build_tool",
    "ToolRegistry",
    # Ad-hoc
    "AdhocTool",
    "AdhocToolDefinition",
    "AdhocToolFactory",
    "HttpToolConfig",
    "ScriptToolConfig",
    "TemplateToolConfig",
    "CompositeToolConfig",
    "CompositeStep",
    # Composer
    "ToolComposer",
    "ToolInfo",
    "ToolPreset",
    # Providers
    "AdhocToolProvider",
    # Scope
    "ToolScope",
    "ToolScopeRule",
    "ToolScopeManager",
    # Sandbox (policy)
    "ToolSandbox",
    "SandboxConfig",
    "SandboxPolicy",
    # Container exec (docker exec into a SandboxHandle) — for sandboxed tools
    "sandbox_exec",
    "sb_run",
    "sb_read_bytes",
    "sb_write_bytes",
    "container_path",
    "SandboxExecError",
    # Plugin discovery (entry-point group: geny_executor.tools)
    "TOOL_ENTRY_POINT_GROUP",
    "ToolPluginRegistry",
    "discover_tool_plugins",
    "register_tool_plugins",
]


# Lazy import for built-in tools to avoid circular dependencies
def get_built_in_registry(working_dir: str = "", **kwargs) -> ToolRegistry:
    """Create a ToolRegistry pre-loaded with all built-in tools.

    Args:
        working_dir: Working directory for file operations.
        **kwargs: Additional keyword arguments passed to ToolContext fields
            (storage_path, env_vars, allowed_paths).

    Returns:
        ToolRegistry with Read, Write, Edit, Bash, Glob, Grep registered.
    """
    from geny_executor.tools.built_in import (
        ReadTool,
        WriteTool,
        EditTool,
        BashTool,
        GlobTool,
        GrepTool,
    )

    registry = ToolRegistry()
    registry.register(ReadTool())
    registry.register(WriteTool())
    registry.register(EditTool())
    registry.register(BashTool())
    registry.register(GlobTool())
    registry.register(GrepTool())
    return registry
