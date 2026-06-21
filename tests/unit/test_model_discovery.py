"""Per-provider model discovery (2.9.0)."""
import pytest
from geny_executor.llm_client.model_discovery import (
    ModelDiscovery, ModelInfo, _HttpResult, discover_models,
)


def _transport(status, body):
    captured = {}
    async def t(url, headers):
        captured["url"] = url
        captured["headers"] = headers
        return _HttpResult(status=status, json=body)
    t.captured = captured
    return t


@pytest.mark.asyncio
async def test_openai_compatible_v1_models():
    t = _transport(200, {"data": [{"id": "gpt-4o"}, {"id": "o3"}]})
    out = await discover_models("openai", api_key="sk-x", transport=t)
    assert out.source == "live"
    assert [m.id for m in out.models] == ["gpt-4o", "o3"]
    assert t.captured["url"].endswith("/models")
    assert t.captured["headers"]["Authorization"] == "Bearer sk-x"


@pytest.mark.asyncio
async def test_ollama_uses_native_tags():
    t = _transport(200, {"models": [{"name": "llama3:latest"}]})
    out = await discover_models("ollama", base_url="http://h:11434/v1", transport=t)
    assert out.source == "live"
    assert out.models == [ModelInfo(id="llama3:latest")]
    assert t.captured["url"] == "http://h:11434/api/tags"  # /v1 stripped


@pytest.mark.asyncio
async def test_anthropic_keeps_display_name():
    t = _transport(200, {"data": [
        {"id": "claude-opus-4-7", "display_name": "Claude Opus 4.7"},
    ]})
    out = await discover_models("anthropic", api_key="k", transport=t)
    assert out.models[0].display_name == "Claude Opus 4.7"
    assert t.captured["headers"]["anthropic-version"] == "2023-06-01"


@pytest.mark.asyncio
async def test_google_filters_to_generatecontent_and_strips_prefix():
    t = _transport(200, {"models": [
        {"name": "models/gemini-3-pro", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/embedding-1", "supportedGenerationMethods": ["embedContent"]},
    ]})
    out = await discover_models("google", api_key="k", transport=t)
    assert [m.id for m in out.models] == ["gemini-3-pro"]


@pytest.mark.asyncio
async def test_claude_code_cli_unavailable():
    out = await discover_models("claude_code_cli")
    assert out.source == "unavailable" and out.error


@pytest.mark.asyncio
async def test_cloud_requires_api_key():
    assert (await discover_models("anthropic")).source == "unavailable"
    assert (await discover_models("google")).error == "no api_key"


@pytest.mark.asyncio
async def test_http_error_is_unavailable():
    t = _transport(401, {"error": "bad key"})
    out = await discover_models("openai", api_key="x", transport=t)
    assert out.source == "unavailable"
    assert out.error == "http 401"
