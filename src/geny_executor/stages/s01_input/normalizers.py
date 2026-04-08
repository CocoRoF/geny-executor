"""Input normalizers — Level 2 strategies for input preprocessing."""

from __future__ import annotations

import unicodedata
from abc import abstractmethod
from typing import Any, Dict, List

from geny_executor.core.stage import Strategy
from geny_executor.stages.s01_input.types import NormalizedInput


class InputNormalizer(Strategy):
    """Base interface for input normalization."""

    @abstractmethod
    def normalize(self, raw_input: Any) -> NormalizedInput:
        """Transform raw input into NormalizedInput."""
        ...


class DefaultNormalizer(InputNormalizer):
    """Standard normalizer — trim, unicode normalize."""

    @property
    def name(self) -> str:
        return "default"

    @property
    def description(self) -> str:
        return "Standard trimming and unicode normalization"

    def normalize(self, raw_input: Any) -> NormalizedInput:
        if isinstance(raw_input, NormalizedInput):
            return raw_input

        if isinstance(raw_input, str):
            text = self._normalize_text(raw_input)
            return NormalizedInput(text=text, raw_input=raw_input)

        if isinstance(raw_input, dict):
            text = self._normalize_text(str(raw_input.get("text", raw_input.get("content", ""))))
            return NormalizedInput(
                text=text,
                metadata=raw_input.get("metadata", {}),
                raw_input=raw_input,
            )

        # Fallback: stringify
        text = self._normalize_text(str(raw_input))
        return NormalizedInput(text=text, raw_input=raw_input)

    def _normalize_text(self, text: str) -> str:
        text = text.strip()
        text = unicodedata.normalize("NFC", text)
        return text


class MultimodalNormalizer(InputNormalizer):
    """Multimodal normalizer — handles images and files."""

    @property
    def name(self) -> str:
        return "multimodal"

    @property
    def description(self) -> str:
        return "Handles text, images, and file attachments"

    def normalize(self, raw_input: Any) -> NormalizedInput:
        if isinstance(raw_input, NormalizedInput):
            return raw_input

        if isinstance(raw_input, str):
            return NormalizedInput(
                text=raw_input.strip(),
                raw_input=raw_input,
            )

        if isinstance(raw_input, dict):
            text = str(raw_input.get("text", raw_input.get("content", ""))).strip()
            images: List[Dict[str, Any]] = []
            files: List[Dict[str, Any]] = []

            for item in raw_input.get("images", []):
                if isinstance(item, dict):
                    images.append(self._make_image_block(item))

            for item in raw_input.get("files", []):
                if isinstance(item, dict):
                    files.append(item)

            return NormalizedInput(
                text=text,
                images=images,
                files=files,
                metadata=raw_input.get("metadata", {}),
                raw_input=raw_input,
            )

        return NormalizedInput(text=str(raw_input).strip(), raw_input=raw_input)

    def _make_image_block(self, image: Dict[str, Any]) -> Dict[str, Any]:
        """Convert to Anthropic image content block format."""
        if "type" in image and image["type"] == "image":
            return image

        media_type = image.get("media_type", "image/png")
        if "base64" in image:
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image["base64"],
                },
            }
        if "url" in image:
            return {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": image["url"],
                },
            }
        return image
