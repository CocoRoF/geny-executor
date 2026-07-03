"""Unit tests for the Workspace ↔ Sandbox tools + s01 PDF document blocks."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from types import SimpleNamespace

import pytest

import geny_executor.tools._sandbox as sb_mod
from geny_executor.stages.s01_input.artifact.default.normalizers import MultimodalNormalizer
from geny_executor.tools.built_in.workspace_tools import (
    SandboxFetchTool,
    SandboxInfoTool,
    SandboxPutTool,
    WorkspaceInfoTool,
)


def _ctx(storage: str | None, sandbox=None):
    return SimpleNamespace(storage_path=storage, sandbox=sandbox, working_dir=storage or ".")


def _run(coro):
    return asyncio.run(coro)


# ── WorkspaceInfo ────────────────────────────────────────────────────


def test_workspace_info_summary_and_subtree(tmp_path):
    (tmp_path / "workspace/uploads").mkdir(parents=True)
    (tmp_path / "workspace/uploads/a.txt").write_text("hello")
    (tmp_path / "workspace/outputs").mkdir(parents=True)
    (tmp_path / "workspace/outputs/b.bin").write_bytes(b"12345678")
    (tmp_path / "note.md").write_text("root file")

    tool = WorkspaceInfoTool()
    summary = _run(tool.execute({}, _ctx(str(tmp_path)))).content
    assert any(d["dir"] == "workspace/" for d in summary["directories"])
    assert any(f["file"] == "note.md" for f in summary["files"])

    sub = _run(tool.execute({"path": "workspace"}, _ctx(str(tmp_path)))).content
    paths = {f["path"] for f in sub["files"]}
    assert paths == {"workspace/uploads/a.txt", "workspace/outputs/b.bin"}


def test_workspace_info_guards(tmp_path):
    tool = WorkspaceInfoTool()
    res = _run(tool.execute({"path": "../../etc"}, _ctx(str(tmp_path))))
    assert res.is_error and res.content["error"] == "PATH_ESCAPE"
    res2 = _run(tool.execute({}, _ctx(None)))
    assert res2.is_error and res2.content["error"] == "NO_STORAGE"


# ── Sandbox transfer (fake sandbox via monkeypatched primitives) ────


class _FakeSandboxFS:
    """In-memory 'container' filesystem keyed by resolved container path."""

    def __init__(self):
        self.files: dict[str, bytes] = {}


@pytest.fixture()
def fake_sandbox(monkeypatch):
    fs = _FakeSandboxFS()

    async def fake_write(sandbox, path, data, *, workdir):
        fs.files[sb_mod.container_path(path, workdir)] = data
        return len(data)

    async def fake_read(sandbox, path, *, workdir):
        cpath = sb_mod.container_path(path, workdir)
        if cpath not in fs.files:
            raise FileNotFoundError(path)
        return fs.files[cpath]

    monkeypatch.setattr(sb_mod, "sb_write_bytes", fake_write)
    monkeypatch.setattr(sb_mod, "sb_read_bytes", fake_read)
    return fs


def test_sandbox_put_and_fetch_roundtrip(tmp_path, fake_sandbox):
    (tmp_path / "workspace/uploads").mkdir(parents=True)
    src = tmp_path / "workspace/uploads/data.csv"
    src.write_bytes(b"a,b\n1,2\n")
    ctx = _ctx(str(tmp_path), sandbox=object())

    put = _run(SandboxPutTool().execute({"source": "workspace/uploads/data.csv"}, ctx))
    assert not put.is_error, put.content
    assert put.content["sandbox_path"] == "/workspace/data.csv"
    assert fake_sandbox.files["/workspace/data.csv"] == b"a,b\n1,2\n"

    # mutate in the "sandbox", then fetch back to the default outputs path
    fake_sandbox.files["/workspace/result.csv"] = b"a,b\n9,9\n"
    fetch = _run(SandboxFetchTool().execute({"source": "result.csv"}, ctx))
    assert not fetch.is_error, fetch.content
    assert fetch.content["workspace_path"] == "workspace/outputs/result.csv"
    assert (tmp_path / "workspace/outputs/result.csv").read_bytes() == b"a,b\n9,9\n"


def test_sandbox_tools_require_sandbox(tmp_path):
    ctx = _ctx(str(tmp_path), sandbox=None)
    assert _run(SandboxPutTool().execute({"source": "x"}, ctx)).content["error"] == "NO_SANDBOX"
    assert _run(SandboxFetchTool().execute({"source": "x"}, ctx)).content["error"] == "NO_SANDBOX"
    info = _run(SandboxInfoTool().execute({}, ctx)).content
    assert info["attached"] is False


def test_sandbox_fetch_dest_guard(tmp_path, fake_sandbox):
    fake_sandbox.files["/workspace/evil"] = b"x"
    ctx = _ctx(str(tmp_path), sandbox=object())
    res = _run(SandboxFetchTool().execute({"source": "evil", "dest": "../../etc/passwd"}, ctx))
    assert res.is_error and res.content["error"] == "PATH_ESCAPE"


# ── s01 PDF → Anthropic document block ──────────────────────────────


def test_pdf_file_becomes_document_block(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf_bytes = b"%PDF-1.4 fake"
    pdf.write_bytes(pdf_bytes)

    norm = MultimodalNormalizer().normalize({
        "text": "read this",
        "attachments": [{
            "kind": "file", "name": "doc.pdf",
            "mime_type": "application/pdf", "url": f"file://{pdf}",
        }],
    })
    blocks = norm.to_message_content()
    doc = [b for b in blocks if b.get("type") == "document"]
    assert len(doc) == 1
    assert doc[0]["source"]["media_type"] == "application/pdf"
    assert base64.b64decode(doc[0]["source"]["data"]) == pdf_bytes
    # no placeholder text block for the PDF
    assert not any("[attached file" in b.get("text", "") for b in blocks if b.get("type") == "text")


def test_non_pdf_file_keeps_placeholder(tmp_path):
    norm = MultimodalNormalizer().normalize({
        "text": "hi",
        "files": [{"name": "deck.pptx",
                   "mime_type": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                   "url": "file:///nonexistent/deck.pptx"}],
    })
    blocks = norm.to_message_content()
    assert any("[attached file: deck.pptx" in b.get("text", "") for b in blocks if b.get("type") == "text")
