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
"""

from geny_executor.tools.built_in.read_tool import ReadTool
from geny_executor.tools.built_in.write_tool import WriteTool
from geny_executor.tools.built_in.edit_tool import EditTool
from geny_executor.tools.built_in.bash_tool import BashTool
from geny_executor.tools.built_in.glob_tool import GlobTool
from geny_executor.tools.built_in.grep_tool import GrepTool


BUILT_IN_TOOL_CLASSES: dict[str, type] = {
    "Read": ReadTool,
    "Write": WriteTool,
    "Edit": EditTool,
    "Bash": BashTool,
    "Glob": GlobTool,
    "Grep": GrepTool,
}


__all__ = [
    "ReadTool",
    "WriteTool",
    "EditTool",
    "BashTool",
    "GlobTool",
    "GrepTool",
    "BUILT_IN_TOOL_CLASSES",
]
