"""STT provider interface — the swappable speech-to-text backend.

Mirrors the embedding-client contract (the package's canonical
"strong interface" seam): a deliberately minimal ``@runtime_checkable``
Protocol, a small result dataclass, and a categorized error type so
tools can turn failures into user-actionable messages.

The host wires a concrete provider per session through
``ToolContext.extras["stt"]`` (serializable config — provider name +
endpoint + model), resolved via :func:`geny_executor.audio.stt.registry.
create_stt_client`. Hosts with bespoke engines register them with
:func:`register_stt_provider` and address them by name from plain
config — no client-instance injection needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, runtime_checkable


@dataclass
class STTSegment:
    """One timed span of the transcript (``timestamps=True`` results)."""

    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "text": self.text}


@dataclass
class STTResult:
    """A completed transcription."""

    text: str
    language: Optional[str] = None
    duration_seconds: Optional[float] = None
    segments: Optional[List[STTSegment]] = None
    #: Provider descriptor that produced this result ("provider/model").
    provider: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out: dict = {"text": self.text, "provider": self.provider}
        if self.language:
            out["language"] = self.language
        if self.duration_seconds is not None:
            out["duration_seconds"] = self.duration_seconds
        if self.segments is not None:
            out["segments"] = [s.to_dict() for s in self.segments]
        return out


#: Failure classes a caller can act on (embedding EmbeddingError parity).
STT_ERROR_CATEGORIES = ("auth", "quota", "transient", "invalid", "unknown")


class STTError(Exception):
    """Categorized STT failure.

    ``category`` ∈ :data:`STT_ERROR_CATEGORIES`:
      * ``auth`` — key/endpoint rejected the request (401/403)
      * ``quota`` — rate/size limits (413/429)
      * ``transient`` — network / 5xx / timeout; retry may succeed
      * ``invalid`` — the audio itself is unusable (bad format, empty)
      * ``unknown`` — anything else
    """

    def __init__(self, message: str, *, category: str = "unknown") -> None:
        super().__init__(message)
        self.category = category if category in STT_ERROR_CATEGORIES else "unknown"


@runtime_checkable
class STTProvider(Protocol):
    """Minimal speech-to-text backend contract.

    Implementations MUST:
      * be safe to call concurrently (or serialize internally);
      * raise :class:`STTError` (never a bare transport exception) so the
        tool layer can classify failures;
      * treat ``language=None`` as auto-detect.
    """

    async def transcribe(
        self,
        audio: bytes,
        *,
        mime_type: str,
        language: Optional[str] = None,
        timestamps: bool = False,
    ) -> STTResult:
        """Transcribe ``audio`` and return the result."""
        ...

    @property
    def descriptor(self) -> str:
        """Stable identity, e.g. ``"openai_compatible/whisper-large-v3"``."""
        ...
