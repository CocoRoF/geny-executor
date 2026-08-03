"""Built-in audio tools — the workspace's speech-to-text bridge.

The model has no audio content block (its media vocabulary is
text/image/document), so audio that lands in the workspace is opaque to
it. This family is the bridge: ``AudioTranscribe`` turns a workspace
audio file into TEXT via the host-wired STT provider
(``ctx.extras["stt"]`` → :func:`geny_executor.audio.stt.create_stt_client`).

Framework contract (what makes this more than a one-shot call):
  * **Sidecar cache** — every transcription is persisted next to the
    audio as ``<file>.transcript.json`` (text, segments, language,
    provider, source sha256). Re-calls return the cache without touching
    the STT service; the cache invalidates when the audio's sha changes.
    Because the sidecar is an ordinary workspace file it automatically
    joins the rest of the ecosystem: Read/Grep/doc tools can consume it,
    it lands in memory, and multi-PC workspace sync shares it.
  * **Gate** — the whole family is hidden behind ``feature:stt_enabled``
    (host sets it only when a usable provider is configured). No dead
    tools: if you can see AudioTranscribe, it works.

Tools:
  * ``AudioTranscribe`` (core) — path → transcript text (+timestamps opt)
  * ``AudioListFiles``  (deferred) — what audio exists / what's transcribed
  * ``AudioInfo``       (deferred) — size/format probe before committing
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from geny_executor.audio.stt import STTError, STTResult, create_stt_client
from geny_executor.tools.base import Tool, ToolCapabilities, ToolResult
from geny_executor.tools.built_in._path_guard import resolve_and_validate

_STT_FEATURE_KEY = "feature:stt_enabled"

#: formats the reference decoder (librosa/ffmpeg-family) reliably handles
AUDIO_EXTENSIONS = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".webm": "audio/webm",
    ".flac": "audio/flac",
}

_MAX_AUDIO_BYTES = 50 * 1024 * 1024  # P1 cap; long-audio chunking is a later phase
_SIDECAR_SUFFIX = ".transcript.json"

#: category → what the agent should tell the user / do next
_CATEGORY_HINTS = {
    "auth": "the STT endpoint rejected the credentials — ask the operator to check the STT provider key/URL",
    "quota": "the STT endpoint is rate/size limited — retry later or use a smaller file",
    "transient": "the STT service had a temporary failure — retrying once is reasonable",
    "invalid": "the audio file could not be decoded — check the file format/contents",
    "unknown": "an unexpected STT failure occurred",
}


def _err(code: str, message: str) -> ToolResult:
    return ToolResult(content={"error": {"code": code, "message": message}}, is_error=True)


def _stt_config(context: Any) -> Optional[Dict[str, Any]]:
    extras = getattr(context, "extras", None) or {}
    cfg = extras.get("stt")
    if isinstance(cfg, dict) and (cfg.get("api_url") or cfg.get("provider")):
        return cfg
    return None


def _build_provider(cfg: Dict[str, Any]):
    kwargs = {k: v for k, v in cfg.items() if k not in ("provider", "language") and v is not None}
    return create_stt_client(cfg.get("provider") or "openai_compatible", **kwargs)


def _resolve_audio_path(context: Any, path: str):
    """Path-guard *path* into the workspace; must be an existing audio file."""
    working_dir = getattr(context, "working_dir", "") or ""
    allowed = getattr(context, "allowed_paths", None)
    try:
        target = resolve_and_validate(path, working_dir, allowed)
    except PermissionError as exc:
        return None, _err("PATH_ESCAPE", str(exc))
    except ValueError as exc:
        return None, _err("BAD_PATH", str(exc))
    if not target.exists() or not target.is_file():
        return None, _err("NOT_FOUND", f"No such audio file: {path}")
    if target.suffix.lower() not in AUDIO_EXTENSIONS:
        return None, _err(
            "NOT_AUDIO",
            f"'{target.name}' is not a supported audio format "
            f"({', '.join(sorted(AUDIO_EXTENSIONS))}).",
        )
    return target, None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sidecar_path(audio: Path) -> Path:
    return audio.with_name(audio.name + _SIDECAR_SUFFIX)


def _load_sidecar(audio: Path, source_sha: str) -> Optional[dict]:
    """Return the cached transcript IFF it matches the current audio bytes."""
    sc = _sidecar_path(audio)
    if not sc.exists():
        return None
    try:
        data = json.loads(sc.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if data.get("source_sha256") != source_sha:
        return None  # audio changed since transcription → stale
    return data


def _write_sidecar(audio: Path, source_sha: str, result: STTResult) -> dict:
    data = result.to_dict()
    data["source_sha256"] = source_sha
    data["source_file"] = audio.name
    data["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    sc = _sidecar_path(audio)
    tmp = sc.with_suffix(sc.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(sc)
    return data


def _format_transcript(data: dict, *, cached: bool, include_segments: bool) -> str:
    lines: List[str] = []
    meta = []
    if data.get("language"):
        meta.append(f"language={data['language']}")
    if data.get("duration_seconds") is not None:
        meta.append(f"duration={data['duration_seconds']:.1f}s")
    meta.append(f"provider={data.get('provider', '?')}")
    meta.append("cached=yes" if cached else "cached=no")
    lines.append(f"[transcript: {data.get('source_file', '?')} · {' · '.join(meta)}]")
    text = (data.get("text") or "").strip()
    lines.append(text if text else "(no speech detected)")
    if include_segments and data.get("segments"):
        lines.append("")
        lines.append("[segments]")
        for seg in data["segments"]:
            lines.append(f"{seg['start']:8.2f}–{seg['end']:8.2f}  {seg['text']}")
    lines.append("")
    lines.append(f"(transcript saved: {data.get('source_file', '')}{_SIDECAR_SUFFIX})")
    return "\n".join(lines)


class _AudioToolBase(Tool):
    """Shared feature gate for the audio family."""

    def required_config_keys(self) -> List[str]:
        # Host gate — hidden until the host wires a usable STT provider.
        return [_STT_FEATURE_KEY]


class AudioTranscribeTool(_AudioToolBase):
    """Transcribe a workspace audio file to text (cached in a sidecar)."""

    @property
    def name(self) -> str:
        return "AudioTranscribe"

    @property
    def description(self) -> str:
        return (
            "Transcribe a workspace audio file (wav/mp3/m4a/ogg/webm/flac) to "
            "text with the configured STT model. Results are cached next to "
            "the file as <name>.transcript.json and reused until the audio "
            "changes; set force=true to re-transcribe. timestamps=true adds "
            "timed segments. Max 50MB."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Audio file path (relative to the workspace).",
                },
                "language": {
                    "type": "string",
                    "description": "ISO language hint (e.g. 'ko', 'en'). Omit for auto-detect.",
                },
                "timestamps": {
                    "type": "boolean",
                    "description": "Include timed segments (default false).",
                },
                "force": {
                    "type": "boolean",
                    "description": "Ignore the cached transcript and call the STT model again.",
                },
            },
            "required": ["path"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(read_only=False, concurrency_safe=True, idempotent=True)

    async def execute(self, input: Dict[str, Any], context: Any) -> ToolResult:
        cfg = _stt_config(context)
        if cfg is None:
            # unreachable when the gate works; kept for defense in depth
            return _err("STT_NOT_CONFIGURED", "No STT provider is configured for this session.")

        target, err = _resolve_audio_path(context, str(input.get("path", "")))
        if err:
            return err
        assert target is not None

        size = target.stat().st_size
        if size > _MAX_AUDIO_BYTES:
            return _err(
                "TOO_LARGE",
                f"'{target.name}' is {size // (1024*1024)}MB (max "
                f"{_MAX_AUDIO_BYTES // (1024*1024)}MB). Split the audio first.",
            )

        include_segments = bool(input.get("timestamps"))
        source_sha = await asyncio.to_thread(_sha256, target)

        if not input.get("force"):
            cached = _load_sidecar(target, source_sha)
            if cached is not None and (not include_segments or cached.get("segments")):
                return ToolResult(content=_format_transcript(
                    cached, cached=True, include_segments=include_segments,
                ))

        try:
            provider = _build_provider(cfg)
        except (ValueError, TypeError) as exc:
            return _err("STT_MISCONFIGURED", f"STT provider config invalid: {exc}")

        audio_bytes = await asyncio.to_thread(target.read_bytes)
        mime = AUDIO_EXTENSIONS[target.suffix.lower()]
        language = input.get("language") or cfg.get("language") or None

        try:
            result = await provider.transcribe(
                audio_bytes,
                mime_type=mime,
                language=language,
                timestamps=include_segments,
            )
        except STTError as exc:
            hint = _CATEGORY_HINTS.get(exc.category, _CATEGORY_HINTS["unknown"])
            return _err(f"STT_{exc.category.upper()}", f"{exc} — {hint}")

        data = await asyncio.to_thread(_write_sidecar, target, source_sha, result)
        return ToolResult(content=_format_transcript(
            data, cached=False, include_segments=include_segments,
        ))


class AudioListFilesTool(_AudioToolBase):
    """List workspace audio files and their transcription state."""

    @property
    def name(self) -> str:
        return "AudioListFiles"

    @property
    def description(self) -> str:
        return (
            "List audio files in the workspace (wav/mp3/m4a/ogg/webm/flac) "
            "with size and whether a transcript sidecar already exists. Use "
            "before AudioTranscribe to see what can be transcribed."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Subdirectory to search (default: whole workspace).",
                },
            },
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(read_only=True, concurrency_safe=True, idempotent=True)

    async def execute(self, input: Dict[str, Any], context: Any) -> ToolResult:
        working_dir = getattr(context, "working_dir", "") or ""
        allowed = getattr(context, "allowed_paths", None)
        base = str(input.get("path") or ".")
        try:
            root = resolve_and_validate(base, working_dir, allowed)
        except (PermissionError, ValueError) as exc:
            return _err("BAD_PATH", str(exc))
        if not root.is_dir():
            return _err("NOT_FOUND", f"No such directory: {base}")

        def _scan() -> List[dict]:
            out: List[dict] = []
            wd = Path(working_dir).resolve() if working_dir else root
            for p in sorted(root.rglob("*")):
                if not p.is_file() or p.suffix.lower() not in AUDIO_EXTENSIONS:
                    continue
                try:
                    rel = str(p.resolve().relative_to(wd))
                except ValueError:
                    rel = str(p)
                out.append({
                    "path": rel,
                    "size_bytes": p.stat().st_size,
                    "transcribed": _sidecar_path(p).exists(),
                })
                if len(out) >= 200:
                    break
            return out

        files = await asyncio.to_thread(_scan)
        if not files:
            return ToolResult(content="No audio files found in the workspace.")
        lines = [f"{len(files)} audio file(s):"]
        for f in files:
            mb = f["size_bytes"] / (1024 * 1024)
            mark = "✓ transcribed" if f["transcribed"] else "· not transcribed"
            lines.append(f"  {f['path']}  ({mb:.1f}MB, {mark})")
        return ToolResult(content="\n".join(lines))


class AudioInfoTool(_AudioToolBase):
    """Probe one audio file (size/format/cache state) before transcribing."""

    @property
    def name(self) -> str:
        return "AudioInfo"

    @property
    def description(self) -> str:
        return (
            "Inspect a workspace audio file: size, format, and whether a "
            "cached transcript exists (with its language/duration). Cheap — "
            "use to decide whether AudioTranscribe is worth the cost."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Audio file path."},
            },
            "required": ["path"],
        }

    def capabilities(self, input: Dict[str, Any]) -> ToolCapabilities:
        return ToolCapabilities(read_only=True, concurrency_safe=True, idempotent=True)

    async def execute(self, input: Dict[str, Any], context: Any) -> ToolResult:
        target, err = _resolve_audio_path(context, str(input.get("path", "")))
        if err:
            return err
        assert target is not None
        size = target.stat().st_size
        info: Dict[str, Any] = {
            "file": target.name,
            "format": target.suffix.lower().lstrip("."),
            "mime_type": AUDIO_EXTENSIONS[target.suffix.lower()],
            "size_mb": round(size / (1024 * 1024), 2),
            "within_transcribe_limit": size <= _MAX_AUDIO_BYTES,
        }
        sc = _sidecar_path(target)
        if sc.exists():
            try:
                data = json.loads(sc.read_text(encoding="utf-8"))
                current = await asyncio.to_thread(_sha256, target)
                info["transcript"] = {
                    "exists": True,
                    "fresh": data.get("source_sha256") == current,
                    "language": data.get("language"),
                    "duration_seconds": data.get("duration_seconds"),
                    "provider": data.get("provider"),
                    "chars": len(data.get("text") or ""),
                }
            except (OSError, ValueError):
                info["transcript"] = {"exists": True, "fresh": False}
        else:
            info["transcript"] = {"exists": False}
        return ToolResult(content=json.dumps(info, ensure_ascii=False, indent=1))


AUDIO_TOOL_CLASSES: Dict[str, type] = {
    "AudioTranscribe": AudioTranscribeTool,
    "AudioListFiles": AudioListFilesTool,
    "AudioInfo": AudioInfoTool,
}
