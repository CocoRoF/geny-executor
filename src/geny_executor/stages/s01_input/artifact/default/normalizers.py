"""Default artifact normalizers for Stage 1: Input."""

from __future__ import annotations

import unicodedata
from typing import Any, Dict, List

from geny_executor.stages.s01_input.interface import InputNormalizer
from geny_executor.stages.s01_input.types import NormalizedInput


def _normalize_text(text: str) -> str:
    text = text.strip()
    text = unicodedata.normalize("NFC", text)
    return text


class DefaultNormalizer(InputNormalizer):
    """Standard normalizer — trim, unicode normalize.

    Also routes multimodal inputs (``images`` / ``files`` / ``attachments``
    keys in a dict input) into ``MultimodalNormalizer`` so that the default
    behaviour transparently supports attachments without callers needing to
    explicitly switch normalizers.
    """

    @property
    def name(self) -> str:
        return "default"

    @property
    def description(self) -> str:
        return "Standard trimming and unicode normalization (multimodal-aware)"

    def normalize(self, raw_input: Any) -> NormalizedInput:
        if isinstance(raw_input, NormalizedInput):
            return raw_input

        if isinstance(raw_input, str):
            return NormalizedInput(text=_normalize_text(raw_input), raw_input=raw_input)

        if isinstance(raw_input, dict):
            # Auto-delegate to MultimodalNormalizer when attachments present.
            if any(k in raw_input for k in ("images", "files", "attachments")):
                normalized = MultimodalNormalizer().normalize(raw_input)
                normalized.text = _normalize_text(normalized.text)
                return normalized
            text = _normalize_text(str(raw_input.get("text", raw_input.get("content", ""))))
            return NormalizedInput(
                text=text,
                metadata=raw_input.get("metadata", {}),
                raw_input=raw_input,
            )

        return NormalizedInput(text=_normalize_text(str(raw_input)), raw_input=raw_input)


class MultimodalNormalizer(InputNormalizer):
    """Multimodal normalizer — handles images and files.

    Accepts inputs in any of the following shapes for ``images`` / ``files``:

    1. **Anthropic content block** (canonical, returned as-is):
       ``{"type": "image", "source": {"type": "base64"|"url", ...}}``
    2. **Lenient client form** (from Geny backend / executor-web HTTP):
       ``{"kind": "image", "mime_type": "image/png", "data": "<b64>"}`` or
       ``{"kind": "image", "mime_type": "image/png", "url": "https://..."}``
    3. **Legacy short form**:
       ``{"media_type": "image/png", "base64": "..."}`` or
       ``{"media_type": "image/png", "url": "..."}``

    All forms are normalized into Anthropic-style content blocks before
    storage. See :mod:`geny_executor.stages.s01_input.types` for the
    canonical schema.
    """

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

            # Generic ``attachments`` array — auto-route by ``kind``
            for item in raw_input.get("attachments", []) or []:
                if not isinstance(item, dict):
                    continue
                kind = (item.get("kind") or item.get("type") or "").lower()
                if kind in ("image", "img"):
                    images.append(self._make_image_block(item))
                else:
                    files.append(self._make_file_block(item))

            for item in raw_input.get("images", []) or []:
                if isinstance(item, dict):
                    images.append(self._make_image_block(item))

            for item in raw_input.get("files", []) or []:
                if isinstance(item, dict):
                    files.append(self._make_file_block(item))

            return NormalizedInput(
                text=text,
                images=images,
                files=files,
                metadata=raw_input.get("metadata", {}),
                raw_input=raw_input,
            )

        return NormalizedInput(text=str(raw_input).strip(), raw_input=raw_input)

    def _make_image_block(self, image: Dict[str, Any]) -> Dict[str, Any]:
        """Convert any accepted shape into an Anthropic image content block."""
        # Already canonical
        if image.get("type") == "image" and isinstance(image.get("source"), dict):
            return image

        media_type = (
            image.get("mime_type")
            or image.get("media_type")
            or image.get("mimeType")
            or "image/png"
        )
        data = image.get("data") or image.get("base64") or image.get("b64")
        url = image.get("url")

        block: Dict[str, Any] = {"type": "image"}
        if data:
            block["source"] = {
                "type": "base64",
                "media_type": media_type,
                "data": data,
            }
        elif url:
            block["source"] = {"type": "url", "url": url}
        else:
            # Malformed input — preserve original for diagnostics
            return image

        # Provenance metadata for downstream stages (memory dehydration etc.)
        meta: Dict[str, Any] = {}
        for k in ("name", "size", "sha256", "attachment_id"):
            if image.get(k) is not None:
                meta[k] = image[k]
        if meta:
            block["_meta"] = meta
        return block

    def _make_file_block(self, file: Dict[str, Any]) -> Dict[str, Any]:
        """Convert any accepted shape into a canonical file block.

        TODO (P1+): PDF 등을 Anthropic ``document`` 블록으로 직접 매핑.
        텍스트 추출 / OCR / 청크 분할 등 본격적인 파일 파이프라인 구현.
        지금은 metadata 만 보존하고 ``to_message_content()`` 에서 메타데이터
        텍스트로 노출된다.
        """
        return {
            "type": "file",
            "name": file.get("name") or file.get("filename"),
            "mime_type": (
                file.get("mime_type")
                or file.get("media_type")
                or file.get("mimeType")
                or "application/octet-stream"
            ),
            "url": file.get("url"),
            "data": file.get("data") or file.get("base64"),
            "size": file.get("size"),
            "sha256": file.get("sha256"),
            "attachment_id": file.get("attachment_id"),
        }
