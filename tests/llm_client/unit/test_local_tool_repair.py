"""Tolerant tool-call argument parsing for local OpenAI-compatible clients (A-4)."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import pytest

pytest.importorskip("openai")

from geny_executor.llm_client.openai import OpenAIClient  # noqa: E402
from geny_executor.llm_client.openai_compatible import (  # noqa: E402
    OllamaClient,
    _repair_json,
)


# ── _repair_json unit ─────────────────────────────────────────────────


def test_repair_strict_valid():
    assert _repair_json('{"path": "a.py", "n": 3}') == {"path": "a.py", "n": 3}


def test_repair_trailing_comma():
    assert _repair_json('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}


def test_repair_python_literals():
    assert _repair_json('{"x": None, "ok": True, "no": False}') == {
        "x": None,
        "ok": True,
        "no": False,
    }


def test_repair_markdown_fence():
    raw = '```json\n{"cmd": "ls"}\n```'
    assert _repair_json(raw) == {"cmd": "ls"}


def test_repair_surrounding_prose():
    raw = 'Sure, here you go: {"q": "weather"} — hope that helps!'
    assert _repair_json(raw) == {"q": "weather"}


def test_repair_unsalvageable_returns_none():
    assert _repair_json("not json at all <<<") is None


def test_repair_scalar_is_rejected():
    # A bare scalar is not a valid tool-argument object.
    assert _repair_json("42") is None


def test_repair_empty_returns_none():
    assert _repair_json("   ") is None


# ── client integration ────────────────────────────────────────────────


def test_ollama_client_repairs_malformed_args():
    client = OllamaClient()
    assert client._parse_tool_arguments('{"a": 1,}') == {"a": 1}
    assert client._parse_tool_arguments('{"x": None}') == {"x": None}


def test_ollama_client_strict_path_unchanged():
    client = OllamaClient()
    assert client._parse_tool_arguments('{"a": 1}') == {"a": 1}


def test_ollama_client_unsalvageable_falls_back_to_empty():
    client = OllamaClient()
    assert client._parse_tool_arguments("totally broken") == {}


def test_repair_emits_event():
    events = []
    client = OllamaClient(event_sink=events.append)
    client._parse_tool_arguments('{"a": 1,}')  # needs repair
    repaired = [e for e in events if e.get("type") == "llm_client.tool_args_repaired"]
    assert len(repaired) == 1
    assert repaired[0]["provider"] == "ollama"


def test_no_event_when_strict_parse_succeeds():
    events = []
    client = OllamaClient(event_sink=events.append)
    client._parse_tool_arguments('{"a": 1}')  # valid → no repair
    assert not [e for e in events if e.get("type") == "llm_client.tool_args_repaired"]


def test_base_openai_client_does_not_repair():
    # The cloud OpenAI path keeps the strict ``{}`` fallback — no behaviour
    # change. (OpenAI emits well-formed JSON; repair is a local-only need.)
    client = OpenAIClient(api_key="sk-mock")
    assert client._parse_tool_arguments('{"a": 1,}') == {}
    assert client._parse_tool_arguments('{"a": 1}') == {"a": 1}
