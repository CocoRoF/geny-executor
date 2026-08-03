"""Audio/STT family — effect-proving tests.

The doctrine mirrors test_ssh_tools: every test asserts the MEASURED
contract the family promises — the gate hides unwired tools, the
sidecar cache eliminates repeat STT calls (call-count proven), the sha
binds cache to content, the path guard confines to the workspace, and
failures carry actionable categories.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from geny_executor.audio.stt import (
    STTError,
    STTResult,
    STTSegment,
    create_stt_client,
    register_stt_provider,
    unregister_stt_provider,
)
from geny_executor.tools.base import ToolContext
from geny_executor.tools.built_in import BUILT_IN_TOOL_CLASSES, BUILT_IN_TOOL_FEATURES
from geny_executor.tools.built_in import audio_tools
from geny_executor.tools.built_in.audio_tools import (
    AudioInfoTool,
    AudioListFilesTool,
    AudioTranscribeTool,
)


class FakeSTT:
    """Recording provider — proves exactly how often the model is hit."""

    calls: int = 0
    fail_category: str | None = None

    def __init__(self, **_cfg):
        pass

    @property
    def descriptor(self) -> str:
        return "fake/stt-test"

    async def transcribe(self, audio, *, mime_type, language=None, timestamps=False):
        FakeSTT.calls += 1
        if FakeSTT.fail_category:
            raise STTError("boom", category=FakeSTT.fail_category)
        segments = [STTSegment(0.0, 1.5, "안녕하세요"), STTSegment(1.5, 3.0, "테스트입니다")]
        return STTResult(
            text="안녕하세요 테스트입니다",
            language=language or "ko",
            duration_seconds=3.0,
            segments=segments if timestamps else None,
            provider=self.descriptor,
        )


@pytest.fixture(autouse=True)
def _fake_provider():
    FakeSTT.calls = 0
    FakeSTT.fail_category = None
    register_stt_provider("fake-test", FakeSTT, replace=True)
    yield
    unregister_stt_provider("fake-test")


def _ctx(tmp_path, provider: str = "fake-test") -> ToolContext:
    ws = tmp_path / "workspace"
    ws.mkdir(exist_ok=True)
    return ToolContext(
        session_id="s1",
        working_dir=str(ws),
        storage_path=str(tmp_path),
        allowed_paths=[str(ws)],
        extras={"stt": {"provider": provider, "api_url": "http://stt.local", "model": "m"}},
    )


def _mk_audio(ctx: ToolContext, name: str = "회의록.wav", content: bytes = b"RIFFfake-wav-bytes") -> str:
    from pathlib import Path

    p = Path(ctx.working_dir) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return name


def _run(tool, input, ctx):
    return asyncio.run(tool.execute(input, ctx))


# ── gate ──────────────────────────────────────────────────────────────


def test_family_registered_and_gated():
    """All three tools exist in the built-in maps and share the gate."""
    assert BUILT_IN_TOOL_FEATURES["audio"] == ["AudioTranscribe", "AudioListFiles", "AudioInfo"]
    for name in BUILT_IN_TOOL_FEATURES["audio"]:
        tool = BUILT_IN_TOOL_CLASSES[name]()
        assert tool.required_config_keys() == ["feature:stt_enabled"], name


def test_gate_drops_family_without_feature_token():
    """EFFECT PROOF: without feature:stt_enabled the tools are removed
    from the registry (never reach the model); with it they stay."""
    from geny_executor.core.pipeline import _gate_unconfigured_tools
    from geny_executor.tools.registry import ToolRegistry

    for satisfied, expect in ((set(), False), ({"feature:stt_enabled"}, True)):
        reg = ToolRegistry()
        reg.register(AudioTranscribeTool())
        reg.register(AudioListFilesTool())
        _gate_unconfigured_tools(reg, satisfied, report=None)
        assert (reg.get("AudioTranscribe") is not None) is expect
        assert (reg.get("AudioListFiles") is not None) is expect


# ── transcription + sidecar cache ─────────────────────────────────────


def test_transcribe_writes_sidecar_and_returns_text(tmp_path):
    ctx = _ctx(tmp_path)
    name = _mk_audio(ctx)
    res = _run(AudioTranscribeTool(), {"path": name}, ctx)
    assert not res.is_error
    assert "안녕하세요 테스트입니다" in res.content
    assert "cached=no" in res.content
    sidecar = tmp_path / "workspace" / f"{name}.transcript.json"
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["text"] == "안녕하세요 테스트입니다"
    assert data["provider"] == "fake/stt-test"
    assert len(data["source_sha256"]) == 64


def test_sidecar_cache_prevents_repeat_stt_calls(tmp_path):
    """EFFECT PROOF: the second call is served from the sidecar — the
    provider is NOT called again (measured call count)."""
    ctx = _ctx(tmp_path)
    name = _mk_audio(ctx)
    _run(AudioTranscribeTool(), {"path": name}, ctx)
    assert FakeSTT.calls == 1
    res2 = _run(AudioTranscribeTool(), {"path": name}, ctx)
    assert FakeSTT.calls == 1, "cache hit must not touch the STT model"
    assert "cached=yes" in res2.content
    assert "안녕하세요 테스트입니다" in res2.content


def test_cache_invalidated_when_audio_changes(tmp_path):
    """EFFECT PROOF: changing the audio bytes invalidates the sidecar
    (sha-bound), so stale transcripts can never be served."""
    ctx = _ctx(tmp_path)
    name = _mk_audio(ctx, content=b"RIFF-take-one")
    _run(AudioTranscribeTool(), {"path": name}, ctx)
    from pathlib import Path

    Path(ctx.working_dir, name).write_bytes(b"RIFF-take-two-different")
    _run(AudioTranscribeTool(), {"path": name}, ctx)
    assert FakeSTT.calls == 2, "changed audio must be re-transcribed"


def test_force_retranscribes(tmp_path):
    ctx = _ctx(tmp_path)
    name = _mk_audio(ctx)
    _run(AudioTranscribeTool(), {"path": name}, ctx)
    _run(AudioTranscribeTool(), {"path": name, "force": True}, ctx)
    assert FakeSTT.calls == 2


def test_timestamps_upgrade_bypasses_textonly_cache(tmp_path):
    """A cached text-only transcript can't satisfy a timestamps request —
    the tool re-transcribes with segments and caches the richer result."""
    ctx = _ctx(tmp_path)
    name = _mk_audio(ctx)
    _run(AudioTranscribeTool(), {"path": name}, ctx)
    res = _run(AudioTranscribeTool(), {"path": name, "timestamps": True}, ctx)
    assert FakeSTT.calls == 2
    assert "[segments]" in res.content and "안녕하세요" in res.content
    # …and now the segment-bearing sidecar serves timestamp requests too
    res3 = _run(AudioTranscribeTool(), {"path": name, "timestamps": True}, ctx)
    assert FakeSTT.calls == 2 and "cached=yes" in res3.content


# ── guards ────────────────────────────────────────────────────────────


def test_path_guard_blocks_escape_and_nonaudio(tmp_path):
    ctx = _ctx(tmp_path)
    (tmp_path / "secret.wav").write_bytes(b"outside-workspace")
    res = _run(AudioTranscribeTool(), {"path": "../secret.wav"}, ctx)
    assert res.is_error and "PATH_ESCAPE" in str(res.content)

    name = _mk_audio(ctx, name="문서.pdf")
    res2 = _run(AudioTranscribeTool(), {"path": name}, ctx)
    assert res2.is_error and "NOT_AUDIO" in str(res2.content)

    res3 = _run(AudioTranscribeTool(), {"path": "없는파일.wav"}, ctx)
    assert res3.is_error and "NOT_FOUND" in str(res3.content)
    assert FakeSTT.calls == 0, "guard failures must never reach the model"


def test_size_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_tools, "_MAX_AUDIO_BYTES", 10)
    ctx = _ctx(tmp_path)
    name = _mk_audio(ctx, content=b"x" * 100)
    res = _run(AudioTranscribeTool(), {"path": name}, ctx)
    assert res.is_error and "TOO_LARGE" in str(res.content)
    assert FakeSTT.calls == 0


def test_stt_error_categories_actionable(tmp_path):
    """EFFECT PROOF: provider failures surface as STT_<CATEGORY> with a
    next-step hint — never a bare stack trace."""
    ctx = _ctx(tmp_path)
    name = _mk_audio(ctx)
    FakeSTT.fail_category = "auth"
    res = _run(AudioTranscribeTool(), {"path": name}, ctx)
    assert res.is_error
    assert "STT_AUTH" in str(res.content) and "key/URL" in str(res.content)
    # no sidecar for failed transcriptions
    assert not (tmp_path / "workspace" / f"{name}.transcript.json").exists()


# ── discovery tools ───────────────────────────────────────────────────


def test_list_files_reports_transcription_state(tmp_path):
    ctx = _ctx(tmp_path)
    a = _mk_audio(ctx, "a.mp3")
    _mk_audio(ctx, "sub/b.flac")
    _mk_audio(ctx, "노트.txt", b"not audio")
    _run(AudioTranscribeTool(), {"path": a}, ctx)

    res = _run(AudioListFilesTool(), {}, ctx)
    assert "a.mp3" in res.content and "✓ transcribed" in res.content
    assert "sub/b.flac" in res.content and "· not transcribed" in res.content
    assert "노트.txt" not in res.content


def test_audio_info_reports_freshness(tmp_path):
    ctx = _ctx(tmp_path)
    name = _mk_audio(ctx)
    info0 = json.loads(_run(AudioInfoTool(), {"path": name}, ctx).content)
    assert info0["transcript"] == {"exists": False}

    _run(AudioTranscribeTool(), {"path": name}, ctx)
    info1 = json.loads(_run(AudioInfoTool(), {"path": name}, ctx).content)
    assert info1["transcript"]["exists"] and info1["transcript"]["fresh"]
    assert info1["within_transcribe_limit"] is True

    from pathlib import Path

    Path(ctx.working_dir, name).write_bytes(b"different bytes now")
    info2 = json.loads(_run(AudioInfoTool(), {"path": name}, ctx).content)
    assert info2["transcript"]["exists"] and not info2["transcript"]["fresh"]


# ── provider registry contract ────────────────────────────────────────


def test_registry_builtin_aliases_and_shadow_guard():
    for alias in ("openai_compatible", "openai", "whisper"):
        c = create_stt_client(alias, api_url="http://x", model="m")
        assert c.descriptor == "openai_compatible/m"
    with pytest.raises(ValueError, match="shadows a built-in"):
        register_stt_provider("whisper", FakeSTT)
    with pytest.raises(ValueError, match="available"):
        create_stt_client("no-such-provider")


def test_openai_compatible_wire_and_error_mapping(monkeypatch):
    """The built-in provider builds a correct multipart request and maps
    HTTP failures to actionable categories."""
    import httpx

    captured = {}

    class _FakeResp:
        def __init__(self, status, payload):
            self.status_code = status
            self._payload = payload
            self.text = json.dumps(payload)

        def json(self):
            return self._payload

    class _FakeClient:
        next_status = 200
        next_payload = {"text": " 전사 결과 ", "language": "ko", "duration": 2.5}

        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, *, data, files, headers):
            captured.update(url=url, data=data, files=files, headers=headers)
            return _FakeResp(_FakeClient.next_status, _FakeClient.next_payload)

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)

    client = create_stt_client(
        "whisper", api_url="http://stt:8001/", model="whisper-large-v3", api_key="sk-x",
    )
    res = asyncio.run(client.transcribe(b"bytes", mime_type="audio/wav", language="ko"))
    assert res.text == "전사 결과" and res.language == "ko" and res.duration_seconds == 2.5
    assert captured["url"] == "http://stt:8001/v1/audio/transcriptions"
    assert captured["data"]["model"] == "whisper-large-v3"
    assert captured["data"]["language"] == "ko"
    assert captured["headers"]["Authorization"] == "Bearer sk-x"
    assert captured["files"]["file"][2] == "audio/wav"

    for status, category in ((401, "auth"), (429, "quota"), (500, "transient"), (400, "invalid")):
        _FakeClient.next_status = status
        with pytest.raises(STTError) as e:
            asyncio.run(client.transcribe(b"bytes", mime_type="audio/wav"))
        assert e.value.category == category, status
