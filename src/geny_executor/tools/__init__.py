"""Tool system — registration, routing, execution, composition."""

from geny_executor.tools.base import Tool, ToolResult, ToolContext
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
from geny_executor.tools.scope import ToolScope, ToolScopeRule, ToolScopeManager
from geny_executor.tools.sandbox import ToolSandbox, SandboxConfig, SandboxPolicy

__all__ = [
    # Base
    "Tool",
    "ToolResult",
    "ToolContext",
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
    # Scope
    "ToolScope",
    "ToolScopeRule",
    "ToolScopeManager",
    # Sandbox
    "ToolSandbox",
    "SandboxConfig",
    "SandboxPolicy",
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
