"""render_vault_map leads with the compressed-first drill guidance (2.20.0)."""
import pytest
from geny_executor.memory.providers.file.provider import FileMemoryProvider
from geny_executor.memory.provider import Scope


@pytest.mark.asyncio
async def test_vault_map_has_progressive_drill_guidance(tmp_path):
    provider = FileMemoryProvider(root=tmp_path, scope=Scope.SESSION)
    await provider.initialize()
    try:
        out = await provider.index().render_vault_map()
        assert "## Vault Map" in out
        low = out.lower()
        assert "compressed" in low and "drill" in low
        assert "memory_read" in out
    finally:
        await provider.close()
