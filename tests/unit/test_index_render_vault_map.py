"""EXEC-5 — `IndexHandle.render_vault_map` / `build_vault_map`.

Hosts (Geny) inject category descriptions; the executor produces
the prompt-injectable markdown block. Verifies the file +
ephemeral providers produce well-shaped output and respect the
host's category-description map.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from geny_executor.memory.provider import Importance, NoteDraft, Scope
from geny_executor.memory.providers.ephemeral import EphemeralMemoryProvider
from geny_executor.memory.providers.file.provider import FileMemoryProvider


def _run(coro):
    return asyncio.run(coro)


_DESCRIPTIONS = {
    "critical": "Always-pinned facts.",
    "topics": "Curated subject pages.",
}


def test_file_build_vault_map_includes_categories_and_descriptions():
    with tempfile.TemporaryDirectory() as td:
        provider = FileMemoryProvider(root=Path(td), scope=Scope.SESSION)

        async def go():
            await provider.initialize()
            await provider.notes().write(
                NoteDraft(
                    title="Pinned",
                    body="x",
                    category="critical",
                    filename="p.md",
                    importance=Importance.CRITICAL,
                    tags=["user"],
                )
            )
            await provider.notes().write(
                NoteDraft(
                    title="Topic",
                    body="y",
                    category="topics",
                    filename="t.md",
                    tags=["python"],
                )
            )
            return await provider.index().build_vault_map(
                category_descriptions=_DESCRIPTIONS,
            )

        vmap = _run(go())
        assert vmap["total_files"] == 2
        assert "critical" in vmap["categories"]
        assert vmap["categories"]["critical"]["files"] == 1
        assert (
            vmap["categories"]["critical"]["description"]
            == "Always-pinned facts."
        )
        assert vmap["categories"]["topics"]["files"] == 1


def test_file_render_vault_map_produces_markdown():
    with tempfile.TemporaryDirectory() as td:
        provider = FileMemoryProvider(root=Path(td), scope=Scope.SESSION)

        async def go():
            await provider.initialize()
            await provider.notes().write(
                NoteDraft(
                    title="P",
                    body="x",
                    category="critical",
                    filename="p.md",
                    tags=["user"],
                )
            )
            return await provider.index().render_vault_map(
                category_descriptions=_DESCRIPTIONS,
            )

        out = _run(go())
        assert out.startswith("## Vault Map")
        assert "- Categories:" in out
        assert "`critical`" in out
        assert "Always-pinned facts." in out
        assert "Top tags:" in out
        assert "user(1)" in out
        assert "Recently modified:" in out
        assert "`p.md`" in out


def test_render_vault_map_without_descriptions_omits_em_dash():
    with tempfile.TemporaryDirectory() as td:
        provider = FileMemoryProvider(root=Path(td), scope=Scope.SESSION)

        async def go():
            await provider.initialize()
            await provider.notes().write(
                NoteDraft(title="N", body="x", category="topics", filename="n.md")
            )
            return await provider.index().render_vault_map()

        out = _run(go())
        assert "## Vault Map" in out
        assert "- `topics` (1)" in out  # No description → no em-dash trailing


def test_ephemeral_render_vault_map_round_trip():
    provider = EphemeralMemoryProvider()

    async def go():
        await provider.initialize()
        await provider.notes().write(
            NoteDraft(
                title="Note",
                body="content",
                category="critical",
                filename="n.md",
                tags=["alpha"],
            )
        )
        return await provider.index().render_vault_map(
            category_descriptions={"critical": "Pinned."},
        )

    out = _run(go())
    assert "## Vault Map" in out
    assert "`critical` (1) — Pinned." in out


def test_build_vault_map_recent_limit_respected():
    with tempfile.TemporaryDirectory() as td:
        provider = FileMemoryProvider(root=Path(td), scope=Scope.SESSION)

        async def go():
            await provider.initialize()
            for i in range(8):
                await provider.notes().write(
                    NoteDraft(
                        title=f"N{i}",
                        body=f"body {i}",
                        category="topics",
                        filename=f"n{i}.md",
                    )
                )
            return await provider.index().build_vault_map(recent_limit=3)

        vmap = _run(go())
        assert len(vmap["recently_modified"]) == 3


def test_build_vault_map_top_tags_respected():
    with tempfile.TemporaryDirectory() as td:
        provider = FileMemoryProvider(root=Path(td), scope=Scope.SESSION)

        async def go():
            await provider.initialize()
            tags = ["a", "b", "c", "d", "e", "f", "g"]
            for i, tag in enumerate(tags):
                await provider.notes().write(
                    NoteDraft(
                        title=f"N{i}",
                        body="x",
                        category="topics",
                        filename=f"n{i}.md",
                        tags=[tag],
                    )
                )
            return await provider.index().build_vault_map(top_tags=3)

        vmap = _run(go())
        assert len(vmap["top_tags"]) == 3
