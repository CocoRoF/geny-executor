"""Input stage data types.

Canonical attachment representation
-----------------------------------
``NormalizedInput.images`` / ``files`` 는 **Anthropic-style content block**
딕셔너리들의 리스트다. 이는 파이프라인 전체에서의 canonical form 이며,
다른 LLM provider (OpenAI / Google) 로 보낼 때는 ``s06_api._translate``
가 provider-native 형식으로 변환한다.

Image block 예:
    {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "<b64>"},
    }
또는
    {
        "type": "image",
        "source": {"type": "url", "url": "https://example.com/x.png"},
    }

File block (Anthropic ``document`` 와 호환되지만 우선 placeholder 로 처리):
    {
        "type": "file",  # TODO: Anthropic ``document`` 블록으로 마이그레이션
        "name": "report.pdf",
        "mime_type": "application/pdf",
        "url": "https://...",            # 또는 source.data (base64)
        "size": 123456,
    }

호출자가 인풋 dict 를 줄 때는 더 관대한 alias 키 (``kind``, ``data``,
``mime_type`` 등) 를 써도 되고, ``MultimodalNormalizer`` 가 위 canonical
형태로 변환해 보관한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class NormalizedInput:
    """Validated and normalized user input."""

    text: str
    role: str = "user"

    # Multimodal content (Anthropic-style canonical content blocks)
    images: List[Dict[str, Any]] = field(default_factory=list)
    files: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    source: str = "user"  # "user", "system", "agent", "broadcast"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Original raw input (before normalization)
    raw_input: Optional[Any] = None

    def has_attachments(self) -> bool:
        return bool(self.images) or bool(self.files)

    def to_message_content(self) -> Any:
        """Convert to canonical (Anthropic-style) message content.

        Returns ``str`` if there is no attachment, otherwise a list of
        content blocks where image / file blocks come first followed by a
        text block. The text block is always included (possibly empty)
        when attachments exist so providers that strictly require text
        alongside media still work.
        """
        if not self.has_attachments():
            return self.text

        blocks: List[Dict[str, Any]] = []
        for img in self.images:
            blocks.append(img)
        for f in self.files:
            # TODO: PDF 등을 Anthropic ``document`` block 으로 매핑하고
            # 본문 추출 (PyPDF, OCR 등) 후 text 블록으로 인라인하는 경로
            # 추가. 현재는 메타데이터를 텍스트로 노출만 한다.
            name = f.get("name") or f.get("filename") or "unnamed"
            mime = f.get("mime_type") or f.get("media_type") or "application/octet-stream"
            blocks.append(
                {
                    "type": "text",
                    "text": f"[attached file: {name} ({mime})]",
                }
            )
        # 텍스트 블록은 항상 마지막에 (빈 문자열도 허용 — provider 가 결정)
        blocks.append({"type": "text", "text": self.text or ""})
        return blocks
