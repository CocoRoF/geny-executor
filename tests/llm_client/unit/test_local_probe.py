"""Local context-window probe (A-3) — Ollama /api/show discovery."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from geny_executor.llm_client.local_probe import (  # noqa: E402
    _extract_num_ctx,
    _ollama_native_root,
    probe_ollama_num_ctx,
    resolve_local_context_window,
)


# ── url derivation ────────────────────────────────────────────────────


def test_native_root_strips_v1():
    assert _ollama_native_root("http://localhost:11434/v1") == "http://localhost:11434"
    assert _ollama_native_root("http://localhost:11434/v1/") == "http://localhost:11434"
    assert _ollama_native_root("http://host:11434") == "http://host:11434"


# ── response parsing ──────────────────────────────────────────────────


def test_extract_from_modelfile_parameters():
    body = {"parameters": 'num_ctx 8192\nstop "<|im_end|>"'}
    assert _extract_num_ctx(body) == 8192


def test_extract_from_model_info_context_length():
    body = {
        "model_info": {
            "general.architecture": "qwen2",
            "qwen2.context_length": 32768,
        }
    }
    assert _extract_num_ctx(body) == 32768


def test_modelfile_num_ctx_wins_over_gguf():
    body = {
        "parameters": "num_ctx 4096",
        "model_info": {
            "general.architecture": "llama",
            "llama.context_length": 131072,
        },
    }
    assert _extract_num_ctx(body) == 4096


def test_extract_arch_key_absent_accepts_any_context_length():
    body = {"model_info": {"foo.context_length": 16384}}
    assert _extract_num_ctx(body) == 16384


def test_extract_none_when_absent():
    assert _extract_num_ctx({"model_info": {"general.architecture": "x"}}) is None
    assert _extract_num_ctx({}) is None
    assert _extract_num_ctx(None) is None


# ── probe (injected transport, no live server) ────────────────────────


async def test_probe_returns_num_ctx_and_hits_correct_url():
    captured = {}

    async def fake_transport(url, body):
        captured["url"] = url
        captured["body"] = body
        return {"parameters": "num_ctx 16384"}

    n = await probe_ollama_num_ctx(
        "http://localhost:11434/v1", "qwen2.5:7b", transport=fake_transport
    )
    assert n == 16384
    assert captured["url"] == "http://localhost:11434/api/show"
    assert captured["body"] == {"model": "qwen2.5:7b"}


async def test_probe_none_on_empty_inputs():
    async def fake_transport(url, body):  # pragma: no cover - must not run
        raise AssertionError("transport must not be called for empty inputs")

    assert await probe_ollama_num_ctx("", "m", transport=fake_transport) is None
    assert await probe_ollama_num_ctx("http://x/v1", "", transport=fake_transport) is None


async def test_probe_none_when_transport_fails():
    async def fake_transport(url, body):
        return None  # server down / non-200 / bad json all collapse to None

    assert (
        await probe_ollama_num_ctx("http://x/v1", "m", transport=fake_transport) is None
    )


# ── resolve dispatch by provider ──────────────────────────────────────


async def test_resolve_ollama_probes():
    async def fake_transport(url, body):
        return {"model_info": {"general.architecture": "llama", "llama.context_length": 8192}}

    n = await resolve_local_context_window(
        "ollama", "http://localhost:11434/v1", "llama3.1", transport=fake_transport
    )
    assert n == 8192


async def test_resolve_lmstudio_returns_none():
    async def fake_transport(url, body):  # pragma: no cover - must not run
        raise AssertionError("lmstudio has no /api/show probe")

    assert (
        await resolve_local_context_window(
            "lmstudio", "http://127.0.0.1:1234/v1", "m", transport=fake_transport
        )
        is None
    )


async def test_resolve_non_profiled_provider_returns_none():
    assert await resolve_local_context_window("anthropic", "", "claude") is None
    assert await resolve_local_context_window("", "", "") is None
