"""Default artifact emitters for Stage 14: Emit."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from geny_executor.core.state import PipelineState
from geny_executor.stages.s14_emit.interface import Emitter
from geny_executor.stages.s14_emit.types import EmitResult


class TextEmitter(Emitter):
    """Emit plain text response — default emitter."""

    def __init__(self, callback: Optional[Callable[[str], Any]] = None):
        self._callback = callback

    @property
    def name(self) -> str:
        return "text"

    async def emit(self, state: PipelineState) -> EmitResult:
        text = state.final_text
        if self._callback and text:
            result = self._callback(text)
            if hasattr(result, "__await__"):
                await result
        return EmitResult(emitted=True, channels=["text"])


class CallbackEmitter(Emitter):
    """Emit via a generic callback with full state access."""

    def __init__(self, callback: Callable[[PipelineState], Any]):
        self._callback = callback

    @property
    def name(self) -> str:
        return "callback"

    async def emit(self, state: PipelineState) -> EmitResult:
        result = self._callback(state)
        if hasattr(result, "__await__"):
            await result
        return EmitResult(emitted=True, channels=["callback"])


class VTuberEmitter(Emitter):
    """Emit for VTuber system — emotion extraction + avatar state."""

    def __init__(
        self,
        emotion_callback: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    ):
        self._emotion_callback = emotion_callback

    @property
    def name(self) -> str:
        return "vtuber"

    async def emit(self, state: PipelineState) -> EmitResult:
        text = state.final_text
        emotion = self._extract_emotion(text, state)

        if self._emotion_callback:
            result = self._emotion_callback(text, emotion)
            if hasattr(result, "__await__"):
                await result

        state.metadata["last_emotion"] = emotion
        return EmitResult(emitted=True, channels=["vtuber"], metadata={"emotion": emotion})

    def _extract_emotion(self, text: str, state: PipelineState) -> Dict[str, Any]:
        emotion_keywords = {
            "happy": ["기뻐", "좋아", "하하", "ㅎㅎ", "😊", "😄", "!!", "awesome", "great"],
            "sad": ["슬퍼", "아쉽", "😢", "😭", "sorry", "unfortunately"],
            "excited": ["대박", "와!", "놀라", "😲", "🎉", "amazing", "wow"],
            "thinking": ["음...", "글쎄", "생각해", "🤔", "hmm", "let me think"],
            "neutral": [],
        }

        text_lower = text.lower()
        scores: Dict[str, float] = {}
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            scores[emotion] = score

        best = max(scores, key=scores.get) if any(scores.values()) else "neutral"
        return {
            "primary": best,
            "confidence": min(scores.get(best, 0) / 3.0, 1.0),
            "scores": scores,
        }


class TTSEmitter(Emitter):
    """Emit text to TTS engine for audio synthesis."""

    def __init__(self, tts_callback: Optional[Callable[[str], Any]] = None):
        self._tts_callback = tts_callback

    @property
    def name(self) -> str:
        return "tts"

    async def emit(self, state: PipelineState) -> EmitResult:
        text = state.final_text
        if not text:
            return EmitResult(emitted=False, channels=["tts"])

        if self._tts_callback:
            result = self._tts_callback(text)
            if hasattr(result, "__await__"):
                await result

        return EmitResult(emitted=True, channels=["tts"])
