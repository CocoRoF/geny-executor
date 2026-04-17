"""Script Sandbox — AST-level security for ad-hoc Python scripts.

Validates scripts before execution by walking the AST to detect
forbidden imports, dangerous built-in calls, and file access.
"""

from __future__ import annotations

import ast
from typing import FrozenSet, List


class ScriptSecurityError(Exception):
    """Raised when a script contains forbidden constructs."""


# Modules that must never be imported from user scripts.
FORBIDDEN_MODULES: FrozenSet[str] = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "ctypes",
        "importlib",
        "socket",
        "http",
        "urllib",
        "requests",
        "pathlib",
        "pickle",
        "marshal",
        "shelve",
        "dbm",
        "signal",
        "threading",
        "multiprocessing",
        "builtins",
        "__builtin__",
        "code",
        "codeop",
        "compileall",
        "pty",
        "pipes",
        "resource",
        "webbrowser",
        "antigravity",
    }
)

# Built-in names considered safe for user scripts.
ALLOWED_BUILTINS: FrozenSet[str] = frozenset(
    {
        "abs",
        "all",
        "any",
        "bin",
        "bool",
        "bytes",
        "callable",
        "chr",
        "complex",
        "dict",
        "dir",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "getattr",
        "hasattr",
        "hash",
        "hex",
        "id",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "oct",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "vars",
        "zip",
    }
)

# Function names that must never be called.
FORBIDDEN_CALLS: FrozenSet[str] = frozenset(
    {
        "exec",
        "eval",
        "compile",
        "__import__",
        "breakpoint",
        "exit",
        "quit",
        "globals",
        "locals",
        "memoryview",
        "open",  # file access
    }
)


def validate_script(code: str) -> List[str]:
    """Parse *code* and return a list of security violations (empty = safe).

    This does **not** execute any code — it only analyses the AST.
    """
    violations: List[str] = []

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"Syntax error: {exc}"]

    for node in ast.walk(tree):
        # ── Import statements ────────────────────────
        if isinstance(node, ast.Import):
            for alias in node.names:
                base = alias.name.split(".")[0]
                if base in FORBIDDEN_MODULES:
                    violations.append(f"Forbidden import: {alias.name}")

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base = node.module.split(".")[0]
                if base in FORBIDDEN_MODULES:
                    violations.append(f"Forbidden import: {node.module}")

        # ── Dangerous function calls ─────────────────
        elif isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr

            if name and name in FORBIDDEN_CALLS:
                violations.append(f"Forbidden call: {name}()")

        # ── Attribute access to dunder ────────────────
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                # Allow __init__, __str__, __repr__ but block __subclasses__ etc.
                dangerous_dunders = {
                    "__subclasses__",
                    "__bases__",
                    "__mro__",
                    "__globals__",
                    "__builtins__",
                    "__code__",
                    "__import__",
                    "__loader__",
                    "__spec__",
                }
                if node.attr in dangerous_dunders:
                    violations.append(f"Forbidden attribute access: {node.attr}")

    return violations


def check_script(code: str) -> None:
    """Raise :class:`ScriptSecurityError` if *code* has violations."""
    violations = validate_script(code)
    if violations:
        raise ScriptSecurityError(
            "Script security violations:\n" + "\n".join(f"  - {v}" for v in violations)
        )
