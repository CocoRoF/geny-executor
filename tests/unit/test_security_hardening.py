"""Security hardening — SSRF wiring + Bash env scrub (audit S3/S5, 2.51.1)."""

from __future__ import annotations

import pytest

from geny_executor.security import SSRFError, validate_url


class TestSSRFGuard:
    @pytest.mark.parametrize(
        "url",
        [
            "http://169.254.169.254/latest/meta-data/",  # cloud metadata
            "http://127.0.0.1/admin",
            "http://localhost:8000/",
            "http://10.0.0.5/internal",
            "http://192.168.1.1/",
            "http://[::1]/",
        ],
    )
    def test_blocks_internal_targets(self, url, monkeypatch):
        monkeypatch.delenv("GENY_ALLOW_PRIVATE_URLS", raising=False)
        with pytest.raises(SSRFError):
            validate_url(url)

    def test_blocks_non_http_scheme(self):
        with pytest.raises(SSRFError):
            validate_url("file:///etc/passwd")

    def test_escape_hatch_allows_private(self, monkeypatch):
        monkeypatch.setenv("GENY_ALLOW_PRIVATE_URLS", "1")
        assert validate_url("http://127.0.0.1:9/") == "http://127.0.0.1:9/"
        # scheme is still validated even with the hatch on
        with pytest.raises(SSRFError):
            validate_url("file:///x")


class TestWebFetchSSRF:
    @pytest.mark.asyncio
    async def test_webfetch_rejects_metadata_ip(self, monkeypatch):
        monkeypatch.delenv("GENY_ALLOW_PRIVATE_URLS", raising=False)
        from geny_executor.tools.built_in.web_fetch_tool import WebFetchTool
        from geny_executor.tools.base import ToolContext

        tool = WebFetchTool()
        res = await tool.execute(
            {"url": "http://169.254.169.254/latest/meta-data/"}, ToolContext()
        )
        assert res.is_error
        assert "blocked" in res.content.lower() or "169.254" in res.content


class TestBashEnvScrub:
    @pytest.mark.asyncio
    async def test_secret_env_not_inherited(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GENY_BASH_INHERIT_ENV", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-should-not-leak")
        monkeypatch.setenv("GENY_AUTH_SECRET", "top-secret")
        from geny_executor.tools.built_in.bash_tool import BashTool
        from geny_executor.tools.base import ToolContext

        tool = BashTool()
        ctx = ToolContext(working_dir=str(tmp_path))
        res = await tool.execute({"command": "env"}, ctx)
        assert "sk-secret-should-not-leak" not in res.content
        assert "top-secret" not in res.content
        # PATH is still present so commands resolve.
        res2 = await tool.execute({"command": "echo $HOME; which sh"}, ctx)
        assert not res2.is_error

    @pytest.mark.asyncio
    async def test_inject_env_reaches_shell(self, monkeypatch, tmp_path):
        from geny_executor.tools.built_in.bash_tool import BashTool
        from geny_executor.tools.base import ToolContext

        tool = BashTool()
        ctx = ToolContext(working_dir=str(tmp_path), env_vars={"MY_VAR": "injected"})
        res = await tool.execute({"command": "echo $MY_VAR"}, ctx)
        assert "injected" in res.content

    @pytest.mark.asyncio
    async def test_opt_in_full_inherit(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GENY_BASH_INHERIT_ENV", "1")
        monkeypatch.setenv("SOME_HOST_VAR", "visible-when-opted-in")
        from geny_executor.tools.built_in.bash_tool import BashTool
        from geny_executor.tools.base import ToolContext

        tool = BashTool()
        res = await tool.execute(
            {"command": "echo $SOME_HOST_VAR"}, ToolContext(working_dir=str(tmp_path))
        )
        assert "visible-when-opted-in" in res.content
