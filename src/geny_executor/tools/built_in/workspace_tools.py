"""Workspace ↔ Sandbox tools — inspect the session's two file spaces and move
files between them.

Every session has up to two distinct spaces:

* **Files workspace** — the host-side session storage (``ToolContext.
  storage_path``). Owned by the framework/host built-in tools (file reads,
  document editing, user file delivery). It is NOT an execution environment —
  agents must not treat it as a system to configure.
* **Sandbox** — an optional isolated container (``ToolContext.sandbox``,
  mounted at a workdir, conventionally ``/workspace``). The agent's free
  environment: install packages, run services, build projects.

The system prompt carries only a short manifest of these spaces; the tools
here provide the concrete state on demand (progressive disclosure):

* ``WorkspaceInfo``  — list the files workspace (summary or subtree)
* ``SandboxInfo``    — is a sandbox attached? workdir + top-level contents
* ``SandboxPut``     — copy a file: files workspace → sandbox
* ``SandboxFetch``   — copy a file: sandbox → files workspace

Transfers stream through the existing ``_sandbox`` primitives (docker exec,
binary-safe) and are path-guarded to the session storage on the host side.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from geny_executor.tools.base import Tool, ToolCapabilities, ToolResult

DEFAULT_SANDBOX_WORKDIR = "/workspace"
MAX_TRANSFER_BYTES = 50 * 1024 * 1024  # docker-exec stdin/stdout transfers
_MAX_ENTRIES = 300


def _err(code: str, message: str) -> ToolResult:
    return ToolResult(content={"error": code, "message": message}, is_error=True)


def _storage_root(context) -> Optional[Path]:
    sp = getattr(context, "storage_path", None)
    return Path(sp).resolve() if sp else None


def _guarded(root: Path, path: str) -> Path:
    p = Path(path)
    target = (p if p.is_absolute() else root / p).resolve()
    target.relative_to(root)  # raises ValueError on escape
    return target


class WorkspaceInfoTool(Tool):
    """Inspect the session's files workspace (host-side session storage)."""

    @property
    def name(self) -> str:
        return "WorkspaceInfo"

    @property
    def description(self) -> str:
        return (
            "List the session's files workspace (host-side storage for user "
            "uploads, document drafts, and delivered outputs — not an execution "
            "environment). No path → top-level summary; pass path (e.g. "
            "'workspace/uploads') for a subtree listing."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Subdirectory to list (default: top-level summary)"},
                "max_entries": {"type": "integer", "description": "Cap on listed files (default 100)"},
            },
        }

    def capabilities(self, input):
        return ToolCapabilities(concurrency_safe=True)

    async def execute(self, input, context):
        root = _storage_root(context)
        if root is None or not root.is_dir():
            return _err("NO_STORAGE", "This session has no storage_path.")
        sub = (input.get("path") or "").strip()
        cap = min(int(input.get("max_entries") or 100), _MAX_ENTRIES)
        try:
            base = _guarded(root, sub) if sub else root
        except ValueError:
            return _err("PATH_ESCAPE", f"Path escapes the session storage: {sub}")
        if not base.exists():
            return _err("NOT_FOUND", f"No such directory: {sub or '/'}")

        if not sub:
            # Top-level summary: per-directory file count + size, root files.
            dirs: List[Dict[str, Any]] = []
            files: List[Dict[str, Any]] = []
            for entry in sorted(base.iterdir()):
                if entry.name.startswith("."):
                    continue
                if entry.is_dir():
                    count = 0
                    size = 0
                    for f in entry.rglob("*"):
                        if f.is_file():
                            count += 1
                            try:
                                size += f.stat().st_size
                            except OSError:
                                pass
                    dirs.append({"dir": entry.name + "/", "files": count, "bytes": size})
                elif entry.is_file():
                    files.append({"file": entry.name, "bytes": entry.stat().st_size})
            return ToolResult(content={
                "root": str(root),
                "directories": dirs,
                "files": files[:cap],
                "hint": "Pass path='<dir>' for a subtree listing.",
            })

        listing: List[Dict[str, Any]] = []
        truncated = False
        for f in sorted(base.rglob("*")):
            if not f.is_file() or f.name.startswith("."):
                continue
            if len(listing) >= cap:
                truncated = True
                break
            try:
                listing.append({
                    "path": f.relative_to(root).as_posix(),
                    "bytes": f.stat().st_size,
                })
            except OSError:
                continue
        return ToolResult(content={
            "root": str(root), "path": sub, "files": listing, "truncated": truncated,
        })


class SandboxInfoTool(Tool):
    """Report whether an isolated sandbox is attached + its top-level contents."""

    @property
    def name(self) -> str:
        return "SandboxInfo"

    @property
    def description(self) -> str:
        return (
            "Check the session's sandbox (isolated container — your free "
            "environment for installs/builds/services). Reports whether one is "
            "attached and lists the top of its workdir."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "workdir": {"type": "string", "description": f"Sandbox workdir (default {DEFAULT_SANDBOX_WORKDIR})"},
            },
        }

    def capabilities(self, input):
        return ToolCapabilities(concurrency_safe=True)

    async def execute(self, input, context):
        sandbox = getattr(context, "sandbox", None)
        if sandbox is None:
            return ToolResult(content={
                "attached": False,
                "note": "No sandbox is bound to this session — only the files "
                        "workspace is available (see WorkspaceInfo).",
            })
        workdir = (input.get("workdir") or DEFAULT_SANDBOX_WORKDIR).strip()
        from geny_executor.tools._sandbox import sb_run

        try:
            rc, out, err = await sb_run(sandbox, "ls -la", workdir=workdir, timeout_s=20)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(content={
                "attached": True, "workdir": workdir, "reachable": False,
                "error": str(exc)[:300],
            })
        return ToolResult(content={
            "attached": True,
            "workdir": workdir,
            "reachable": rc == 0,
            "listing": (out if rc == 0 else err)[:2000],
        })


class SandboxPutTool(Tool):
    """Copy a file from the files workspace into the sandbox."""

    @property
    def name(self) -> str:
        return "SandboxPut"

    @property
    def description(self) -> str:
        return (
            "Copy a file from the session's files workspace (host storage) INTO "
            "the sandbox container. source is relative to the session storage "
            "(e.g. 'workspace/uploads/data.csv'); dest is a sandbox path "
            "(default: same basename in the sandbox workdir)."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "File path in the files workspace"},
                "dest": {"type": "string", "description": "Target path inside the sandbox (optional)"},
                "workdir": {"type": "string", "description": f"Sandbox workdir (default {DEFAULT_SANDBOX_WORKDIR})"},
            },
            "required": ["source"],
        }

    def capabilities(self, input):
        return ToolCapabilities(concurrency_safe=False)

    async def execute(self, input, context):
        sandbox = getattr(context, "sandbox", None)
        if sandbox is None:
            return _err("NO_SANDBOX", "No sandbox is bound to this session.")
        root = _storage_root(context)
        if root is None:
            return _err("NO_STORAGE", "This session has no storage_path.")
        try:
            src = _guarded(root, input["source"])
        except ValueError:
            return _err("PATH_ESCAPE", f"source escapes the session storage: {input['source']}")
        if not src.is_file():
            return _err("NOT_FOUND", f"source not found: {input['source']}")
        size = src.stat().st_size
        if size > MAX_TRANSFER_BYTES:
            return _err("TOO_LARGE", f"{size} bytes exceeds the {MAX_TRANSFER_BYTES}-byte transfer cap.")

        workdir = (input.get("workdir") or DEFAULT_SANDBOX_WORKDIR).strip()
        dest = (input.get("dest") or src.name).strip()
        from geny_executor.tools._sandbox import container_path, sb_write_bytes

        try:
            written = await sb_write_bytes(sandbox, dest, src.read_bytes(), workdir=workdir)
        except Exception as exc:  # noqa: BLE001
            return _err("TRANSFER_FAILED", str(exc)[:300])
        return ToolResult(content={
            "copied": True,
            "source": src.relative_to(root).as_posix(),
            "sandbox_path": container_path(dest, workdir),
            "bytes": written,
        })


class SandboxFetchTool(Tool):
    """Copy a file from the sandbox into the files workspace."""

    @property
    def name(self) -> str:
        return "SandboxFetch"

    @property
    def description(self) -> str:
        return (
            "Copy a file FROM the sandbox container into the session's files "
            "workspace (host storage) — e.g. to deliver a build artifact to the "
            "user (SendUserFile) or preview it in the Canvas. source is a sandbox "
            "path; dest is relative to the session storage (default: "
            "'workspace/outputs/<basename>')."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "File path inside the sandbox"},
                "dest": {"type": "string", "description": "Target path in the files workspace (optional)"},
                "workdir": {"type": "string", "description": f"Sandbox workdir (default {DEFAULT_SANDBOX_WORKDIR})"},
            },
            "required": ["source"],
        }

    def capabilities(self, input):
        return ToolCapabilities(concurrency_safe=False)

    async def execute(self, input, context):
        sandbox = getattr(context, "sandbox", None)
        if sandbox is None:
            return _err("NO_SANDBOX", "No sandbox is bound to this session.")
        root = _storage_root(context)
        if root is None:
            return _err("NO_STORAGE", "This session has no storage_path.")

        workdir = (input.get("workdir") or DEFAULT_SANDBOX_WORKDIR).strip()
        source = input["source"]
        from geny_executor.tools._sandbox import sb_read_bytes

        try:
            data = await sb_read_bytes(sandbox, source, workdir=workdir)
        except FileNotFoundError:
            return _err("NOT_FOUND", f"sandbox file not found: {source}")
        except Exception as exc:  # noqa: BLE001
            return _err("TRANSFER_FAILED", str(exc)[:300])
        if len(data) > MAX_TRANSFER_BYTES:
            return _err("TOO_LARGE", f"{len(data)} bytes exceeds the {MAX_TRANSFER_BYTES}-byte transfer cap.")

        basename = Path(source).name or "fetched"
        dest_rel = (input.get("dest") or f"workspace/outputs/{basename}").strip()
        try:
            dest = _guarded(root, dest_rel)
        except ValueError:
            return _err("PATH_ESCAPE", f"dest escapes the session storage: {dest_rel}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return ToolResult(content={
            "copied": True,
            "source": source,
            "workspace_path": dest.relative_to(root).as_posix(),
            "bytes": len(data),
        })


__all__ = [
    "WorkspaceInfoTool",
    "SandboxInfoTool",
    "SandboxPutTool",
    "SandboxFetchTool",
    "DEFAULT_SANDBOX_WORKDIR",
    "MAX_TRANSFER_BYTES",
]
