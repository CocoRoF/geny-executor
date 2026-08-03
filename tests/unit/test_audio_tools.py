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


# ── audit round: malformed sidecars, cache economics, concurrency ─────


def _write_raw_sidecar(ctx, name, payload):
    from pathlib import Path

    sc = Path(ctx.working_dir) / f"{name}.transcript.json"
    sc.write_text(json.dumps(payload), encoding="utf-8")
    return sc


def test_malformed_sidecars_never_crash(tmp_path):
    """EFFECT PROOF: hand-edited / foreign-schema sidecars (an expected
    input — they sync between PCs) read as cache-miss, never a stack
    trace, and the file is re-transcribed cleanly."""
    ctx = _ctx(tmp_path)
    name = _mk_audio(ctx)
    import hashlib as _h
    from pathlib import Path

    sha = _h.sha256(Path(ctx.working_dir, name).read_bytes()).hexdigest()
    # fully invalid → cache MISS (re-transcribe), never a crash
    invalid_payloads = [
        ["not", "a", "dict"],
        "just a string",
        {"text": 5, "source_sha256": sha},
        {"text": "ok", "source_sha256": "short-sha"},
    ]
    for payload in invalid_payloads:
        FakeSTT.calls = 0
        _write_raw_sidecar(ctx, name, payload)
        res = _run(AudioTranscribeTool(), {"path": name}, ctx)
        assert not res.is_error, f"crashed on {payload!r}: {res.content}"
        assert FakeSTT.calls == 1, f"invalid sidecar must be a cache MISS: {payload!r}"

    # valid core (text+sha) with junk extras → junk dropped, text served
    # from cache (no re-billing), and formatting never crashes
    messy_payloads = [
        {"text": "ok", "source_sha256": sha, "duration_seconds": "3.0",
         "segments": {"weird": 1}},
        {"text": "ok", "source_sha256": sha, "segments": [None, "str", {"start": "x"}]},
    ]
    for payload in messy_payloads:
        FakeSTT.calls = 0
        _write_raw_sidecar(ctx, name, payload)
        res = _run(AudioTranscribeTool(), {"path": name}, ctx)
        assert not res.is_error, f"crashed on {payload!r}: {res.content}"
        assert FakeSTT.calls == 0 and "cached=yes" in res.content, payload

    # AudioInfo reports malformed instead of crashing
    _write_raw_sidecar(ctx, name, ["broken"])
    info = json.loads(_run(AudioInfoTool(), {"path": name}, ctx).content)
    assert info["transcript"]["exists"] and info["transcript"].get("malformed")


def test_string_duration_sidecar_partially_coerced(tmp_path):
    """A sidecar with a coercible-but-wrong-typed duration is treated as
    cache-miss (strict schema) — and formatting never crashes."""
    ctx = _ctx(tmp_path)
    name = _mk_audio(ctx)
    res = _run(AudioTranscribeTool(), {"path": name}, ctx)
    assert "duration=3.0s" in res.content  # runtime float path formats fine


def test_timestamps_cache_hits_on_flag_not_segments(tmp_path):
    """EFFECT PROOF: a server that returns NO segments (silent audio /
    minimal server) must not cause unbounded re-billing — the sidecar's
    timestamps flag satisfies later timestamps requests."""

    class NoSegSTT(FakeSTT):
        async def transcribe(self, audio, *, mime_type, language=None, timestamps=False):
            FakeSTT.calls += 1
            return STTResult(text="무음에 가까움", provider=self.descriptor, segments=None)

    register_stt_provider("noseg-test", NoSegSTT, replace=True)
    try:
        ctx = _ctx(tmp_path, provider="noseg-test")
        name = _mk_audio(ctx)
        _run(AudioTranscribeTool(), {"path": name, "timestamps": True}, ctx)
        assert FakeSTT.calls == 1
        res2 = _run(AudioTranscribeTool(), {"path": name, "timestamps": True}, ctx)
        assert FakeSTT.calls == 1, "no-segment result must still cache timestamps runs"
        assert "cached=yes" in res2.content
    finally:
        unregister_stt_provider("noseg-test")


def test_sidecar_write_failure_keeps_paid_transcript(tmp_path, monkeypatch):
    """EFFECT PROOF: a failed cache write (disk full) must not discard
    the paid STT result — the transcript returns with a warning."""
    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(audio_tools, "_write_sidecar", _boom)
    ctx = _ctx(tmp_path)
    name = _mk_audio(ctx)
    res = _run(AudioTranscribeTool(), {"path": name}, ctx)
    assert not res.is_error
    assert "안녕하세요 테스트입니다" in res.content
    assert "cache could not be saved" in res.content


def test_concurrent_transcribes_one_paid_call(tmp_path):
    """EFFECT PROOF: two simultaneous transcribes of one file collapse to
    ONE provider call (per-file lock; second becomes a cache hit)."""
    ctx = _ctx(tmp_path)
    name = _mk_audio(ctx)

    async def both():
        return await asyncio.gather(
            AudioTranscribeTool().execute({"path": name}, ctx),
            AudioTranscribeTool().execute({"path": name}, ctx),
        )

    r1, r2 = asyncio.run(both())
    assert not r1.is_error and not r2.is_error
    assert FakeSTT.calls == 1, "concurrent same-file transcribes must dedupe"
    assert sum("cached=yes" in str(r.content) for r in (r1, r2)) == 1


def test_list_prunes_heavy_dirs_and_reports_truncation(tmp_path):
    ctx = _ctx(tmp_path)
    _mk_audio(ctx, "node_modules/dep/조용한노래.mp3")
    _mk_audio(ctx, "정상.mp3")
    res = _run(AudioListFilesTool(), {}, ctx)
    assert "정상.mp3" in res.content
    assert "node_modules" not in res.content, "heavy trees must be pruned"

    for i in range(205):
        _mk_audio(ctx, f"많음/f{i:03d}.wav")
    res2 = _run(AudioListFilesTool(), {}, ctx)
    assert "truncated at 200" in res2.content, "silent truncation forbidden"


def test_mp4_no_longer_advertised(tmp_path):
    ctx = _ctx(tmp_path)
    name = _mk_audio(ctx, "영상.mp4")
    res = _run(AudioTranscribeTool(), {"path": name}, ctx)
    assert res.is_error and "NOT_AUDIO" in str(res.content)
    lst = _run(AudioListFilesTool(), {}, ctx)
    assert "영상.mp4" not in lst.content


def test_provider_contract_hardening(monkeypatch):
    """Missing-text payloads are schema errors (never cached silence);
    null segments tolerated; builder returning junk is rejected."""
    import httpx

    class _FakeResp:
        def __init__(self, payload):
            self.status_code = 200
            self._payload = payload
            self.text = json.dumps(payload)

        def json(self):
            return self._payload

    class _FakeClient:
        payload = {"transcription": "wrong-field"}

        def __init__(self, **kw): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, **kw): return _FakeResp(_FakeClient.payload)

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    client = create_stt_client("whisper", api_url="http://x", model="m")

    with pytest.raises(STTError) as e:
        asyncio.run(client.transcribe(b"b", mime_type="audio/wav"))
    assert e.value.category == "invalid" and "text" in str(e.value)

    _FakeClient.payload = {"text": "ok", "segments": [None, "str", {"start": 1, "end": 2, "text": "세그"}]}
    res = asyncio.run(client.transcribe(b"b", mime_type="audio/wav", timestamps=True))
    assert res.text == "ok" and len(res.segments) == 1 and res.segments[0].text == "세그"

    register_stt_provider("junk-builder", lambda **k: None, replace=True)
    try:
        with pytest.raises(TypeError, match="does not implement STTProvider"):
            create_stt_client("junk-builder")
    finally:
        unregister_stt_provider("junk-builder")
