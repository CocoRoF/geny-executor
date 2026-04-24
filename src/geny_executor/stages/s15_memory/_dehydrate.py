"""Multimodal content dehydration for memory persistence.

이미지 content block 의 base64 raw payload 는 한 장당 수 MB 이므로,
LTM persistence (JSON 파일) / STM (in-memory dict) 에 그대로 직렬화하면
디스크/메모리가 빠르게 비대화된다.

이 모듈은 메시지를 ``persistence.save()`` 또는 ``provider.record_turn()``
으로 넘기기 직전에 base64 payload 만 떼어낸 dehydrated 카피를 만드는
헬퍼를 제공한다. ``state.messages`` (in-memory) 자체는 손대지 않으므로
같은 턴 안에서 LLM 재호출이 일어나도 이미지는 그대로 보존된다.

Dehydrated image block 형태:

    {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": null,
                   "_dehydrated": True},
        "_meta": {"name": "...", "size": 12345, "sha256": "..."},
    }

URL source 는 작으므로 그대로 둔다. ``recall`` 시점에 이미지가 필요한
경우 ``_meta.attachment_id`` 를 통해 외부 storage 에서 다시 hydrate 하는
것은 호출자 책임 (이번 단계의 스코프 밖, TODO).
"""

from __future__ import annotations

from typing import Any, Dict, List


def _dehydrate_block(block: Dict[str, Any]) -> Dict[str, Any]:
    btype = block.get("type")
    if btype == "image":
        source = block.get("source") or {}
        if source.get("type") == "base64" and source.get("data"):
            new_source = {**source, "data": None, "_dehydrated": True}
            return {**block, "source": new_source}
        return block
    if btype == "file":
        # File 의 inline data 도 동일하게 떼어낸다. URL/메타는 보존.
        if block.get("data"):
            return {**block, "data": None, "_dehydrated": True}
        return block
    return block


def dehydrate_content(content: Any) -> Any:
    """Return a dehydrated copy of canonical message content."""
    if not isinstance(content, list):
        return content
    return [_dehydrate_block(b) if isinstance(b, dict) else b for b in content]


def dehydrate_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    content = msg.get("content")
    if not isinstance(content, list):
        return msg
    return {**msg, "content": dehydrate_content(content)}


def dehydrate_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return a shallow copy of ``messages`` with media payloads stripped."""
    return [dehydrate_message(m) for m in messages]
