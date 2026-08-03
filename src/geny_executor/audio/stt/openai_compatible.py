"""Built-in STT provider #1 — the OpenAI-compatible transcription API.

``POST {api_url}/v1/audio/transcriptions`` (multipart) is the de-facto
standard surface: OpenAI itself, vLLM-served Whisper (Geny's
``whisper-stt`` container), Groq, faster-whisper servers … one client
covers them all, which is why this is the only transport the executor
ships. Anything more exotic plugs in via ``register_stt_provider``.

Uses ``httpx`` (already a core dependency) — no new extras.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from geny_executor.audio.stt.provider import (
    STTError,
    STTResult,
    STTSegment,
)

_DEFAULT_TIMEOUT = 300.0  # long files legitimately take minutes


def _categorize(status: int) -> str:
    if status in (401, 403):
        return "auth"
    if status in (413, 429):
        return "quota"
    if status >= 500:
        return "transient"
    if status >= 400:
        return "invalid"
    return "unknown"


class OpenAICompatibleSTT:
    """STTProvider over the OpenAI-compatible transcription endpoint."""

    def __init__(
        self,
        *,
        api_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = _DEFAULT_TIMEOUT,
        temperature: float = 0.0,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        if not api_url:
            raise ValueError("api_url is required")
        if not model:
            raise ValueError("model is required")
        self._api_url = api_url.rstrip("/")
        self._model = model
        self._api_key = api_key or None
        self._timeout = float(timeout or _DEFAULT_TIMEOUT)
        self._temperature = temperature
        self._extra_headers = dict(extra_headers or {})

    @property
    def descriptor(self) -> str:
        return f"openai_compatible/{self._model}"

    async def transcribe(
        self,
        audio: bytes,
        *,
        mime_type: str,
        language: Optional[str] = None,
        timestamps: bool = False,
    ) -> STTResult:
        if not audio:
            raise STTError("audio payload is empty", category="invalid")

        headers = dict(self._extra_headers)
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        data: Dict[str, Any] = {
            "model": self._model,
            "response_format": "verbose_json" if timestamps else "json",
            "temperature": str(self._temperature),
        }
        if language:
            data["language"] = language

        ext = (mime_type.rsplit("/", 1)[-1] or "bin").split(";")[0]
        files = {"file": (f"audio.{ext}", audio, mime_type or "application/octet-stream")}

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._api_url}/v1/audio/transcriptions",
                    data=data,
                    files=files,
                    headers=headers,
                )
        except httpx.TimeoutException as exc:
            raise STTError(f"STT request timed out: {exc}", category="transient") from exc
        except httpx.HTTPError as exc:
            raise STTError(f"STT transport error: {exc}", category="transient") from exc
        except Exception as exc:  # noqa: BLE001 — e.g. httpx.InvalidURL is NOT an HTTPError
            raise STTError(f"STT request could not be built: {exc}", category="invalid") from exc

        if resp.status_code != 200:
            body = resp.text[:300]
            raise STTError(
                f"STT endpoint returned HTTP {resp.status_code}: {body}",
                category=_categorize(resp.status_code),
            )

        try:
            payload = resp.json()
        except ValueError:
            # some servers honour response_format=text regardless
            payload = {"text": resp.text}

        segments = None
        raw_segments = payload.get("segments") if isinstance(payload, dict) else None
        if isinstance(raw_segments, list):
            segments = []
            for seg in raw_segments:
                if not isinstance(seg, dict):
                    continue  # null / string entries from lax servers
                try:
                    segments.append(STTSegment(
                        start=float(seg.get("start", 0.0)),
                        end=float(seg.get("end", 0.0)),
                        text=str(seg.get("text", "")).strip(),
                    ))
                except (TypeError, ValueError):
                    continue

        if isinstance(payload, dict) and "text" not in payload:
            raise STTError(
                "STT endpoint returned JSON without a 'text' field "
                f"(keys: {sorted(payload)[:8]}) — not an OpenAI-compatible "
                "transcription response",
                category="invalid",
            )
        text = str(payload.get("text", "")) if isinstance(payload, dict) else str(payload)
        duration = payload.get("duration") if isinstance(payload, dict) else None
        return STTResult(
            text=text.strip(),
            language=(payload.get("language") if isinstance(payload, dict) else None),
            duration_seconds=float(duration) if isinstance(duration, (int, float)) else None,
            segments=segments,
            provider=self.descriptor,
        )
