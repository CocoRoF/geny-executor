"""Emit stage — emitter strategies (Level 2)."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from geny_executor.core.stage import Strategy
from geny_executor.core.state import PipelineState

logger = logging.getLogger(__name__)


@dataclass
class EmitResult:
    """Result of emission."""

    emitted: bool = True
    channels: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Emitter(Strategy, ABC):
    """Level 2 strategy: how to emit results to external consumers."""

    @abstractmethod
    async def emit(self, state: PipelineState) -> EmitResult:
        """Emit pipeline results. Return emission result."""
        ...


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
    """Emit for VTuber system — emotion extraction + avatar state.

    Extracts emotion from response text and updates VTuber state.
    This is a placeholder for integration with Geny VTuber pipeline.
    """

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
        return EmitResult(
            emitted=True,
            channels=["vtuber"],
            metadata={"emotion": emotion},
        )

    def _extract_emotion(self, text: str, state: PipelineState) -> Dict[str, Any]:
        """Rule-based emotion extraction. Override for ML-based."""
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

    def __init__(
        self,
        tts_callback: Optional[Callable[[str], Any]] = None,
    ):
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


class EmitterChain:
    """Chain of emitters — runs all in sequence."""

    def __init__(self, emitters: Optional[List[Emitter]] = None):
        self._emitters = emitters or []

    def add(self, emitter: Emitter) -> None:
        self._emitters.append(emitter)

    async def emit_all(self, state: PipelineState) -> List[EmitResult]:
        results = []
        for emitter in self._emitters:
            try:
                result = await emitter.emit(state)
                results.append(result)
            except Exception as e:
                logger.warning("Emitter %s failed: %s", emitter.name, e)
                results.append(EmitResult(emitted=False, channels=[emitter.name],
                                          metadata={"error": str(e)}))
        return results

    @property
    def emitters(self) -> List[Emitter]:
        return list(self._emitters)
